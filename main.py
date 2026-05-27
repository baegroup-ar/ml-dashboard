import os
import httpx
import secrets
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeTimedSerializer, BadSignature
import bcrypt
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

ML_CLIENT_ID = os.environ["ML_CLIENT_ID"]
ML_CLIENT_SECRET = os.environ["ML_CLIENT_SECRET"]
APP_URL = os.environ.get("APP_URL", "http://localhost:8000")
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))
DATABASE_URL = os.environ["DATABASE_URL"].replace("postgres://", "postgresql+pg8000://").replace("postgresql://", "postgresql+pg8000://")
ADMIN_EMAIL = os.environ["ADMIN_EMAIL"]
ADMIN_PASSWORD = os.environ["ADMIN_PASSWORD"]

ML_AUTH_URL = "https://auth.mercadolibre.com.ar/authorization"
ML_TOKEN_URL = "https://api.mercadolibre.com/oauth/token"
ML_API_URL = "https://api.mercadolibre.com"
SHIPPING_LOGIC_VERSION = "v8-gross-shipping-bonif"

serializer = URLSafeTimedSerializer(SECRET_KEY)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                is_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ml_accounts (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                ml_user_id TEXT NOT NULL,
                nickname TEXT NOT NULL,
                access_token TEXT NOT NULL,
                refresh_token TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(user_id, ml_user_id)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS shipment_cost_cache (
                shipping_id BIGINT PRIMARY KEY,
                cost NUMERIC(10,2) NOT NULL,
                buyer_cost NUMERIC(10,2) DEFAULT NULL,
                logistic_type VARCHAR(50) DEFAULT NULL,
                cached_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("ALTER TABLE shipment_cost_cache ADD COLUMN IF NOT EXISTS buyer_cost NUMERIC(10,2) DEFAULT NULL"))
        conn.execute(text("ALTER TABLE shipment_cost_cache ALTER COLUMN buyer_cost DROP NOT NULL"))
        conn.execute(text("ALTER TABLE shipment_cost_cache ADD COLUMN IF NOT EXISTS logistic_type VARCHAR(50) DEFAULT NULL"))
        conn.execute(text("ALTER TABLE shipment_cost_cache ADD COLUMN IF NOT EXISTS bonificacion NUMERIC(10,2) DEFAULT NULL"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        # Invalida el caché cuando cambia la lógica de cálculo de envío.
        current = conn.execute(text("SELECT value FROM app_meta WHERE key='shipping_logic_version'")).fetchone()
        if not current or current[0] != SHIPPING_LOGIC_VERSION:
            conn.execute(text("DELETE FROM shipment_cost_cache"))
            conn.execute(text(
                "INSERT INTO app_meta (key, value) VALUES ('shipping_logic_version', :v)"
                " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()"
            ), {"v": SHIPPING_LOGIC_VERSION})
        existing = conn.execute(text("SELECT id FROM users WHERE email=:e"), {"e": ADMIN_EMAIL}).fetchone()
        if not existing:
            hashed = bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode()
            conn.execute(text(
                "INSERT INTO users (email, password_hash, name, is_admin) VALUES (:e,:h,:n,:a)"
            ), {"e": ADMIN_EMAIL, "h": hashed, "n": "Administrador", "a": True})
        conn.commit()


@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def db_fetchone(query, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        row = result.fetchone()
        return dict(row._mapping) if row else None


def db_fetchall(query, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        return [dict(r._mapping) for r in result.fetchall()]


def db_execute(query, params=None):
    with engine.connect() as conn:
        conn.execute(text(query), params or {})
        conn.commit()


def get_session_user_id(request: Request) -> Optional[int]:
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        return serializer.loads(token, max_age=86400 * 7)
    except BadSignature:
        return None


def get_user(user_id: int):
    return db_fetchone("SELECT * FROM users WHERE id=:id", {"id": user_id})


async def refresh_ml_token(account_id: int) -> Optional[str]:
    acc = db_fetchone("SELECT * FROM ml_accounts WHERE id=:id", {"id": account_id})
    if not acc:
        return None
    if datetime.utcnow() < acc["expires_at"] - timedelta(minutes=5):
        return acc["access_token"]
    async with httpx.AsyncClient() as client:
        resp = await client.post(ML_TOKEN_URL, data={
            "grant_type": "refresh_token",
            "client_id": ML_CLIENT_ID,
            "client_secret": ML_CLIENT_SECRET,
            "refresh_token": acc["refresh_token"],
        })
        if resp.status_code != 200:
            return None
        tokens = resp.json()
        new_expires = datetime.utcnow() + timedelta(seconds=tokens["expires_in"])
        db_execute(
            "UPDATE ml_accounts SET access_token=:at, refresh_token=:rt, expires_at=:ea WHERE id=:id",
            {"at": tokens["access_token"], "rt": tokens.get("refresh_token", acc["refresh_token"]),
             "ea": new_expires, "id": account_id}
        )
        return tokens["access_token"]


def db_get_cached_shipping(shipping_ids: list) -> dict:
    if not shipping_ids:
        return {}
    result = {}
    chunk = 500
    for i in range(0, len(shipping_ids), chunk):
        batch = shipping_ids[i:i+chunk]
        placeholders = ",".join(f":id{j}" for j in range(len(batch)))
        params = {f"id{j}": sid for j, sid in enumerate(batch)}
        rows = db_fetchall(
            f"SELECT shipping_id, cost, buyer_cost, bonificacion FROM shipment_cost_cache WHERE shipping_id IN ({placeholders})",
            params,
        )
        for row in rows:
            if row["buyer_cost"] is not None and row["bonificacion"] is not None:
                result[row["shipping_id"]] = {
                    "seller": float(row["cost"]),
                    "buyer": float(row["buyer_cost"]),
                    "bonificacion": float(row["bonificacion"]),
                }
    return result


def db_save_shipping_costs(costs: dict):
    for sid, c in costs.items():
        seller = float(c.get("seller", 0))
        buyer = float(c.get("buyer", 0))
        bonif = float(c.get("bonificacion", 0))
        db_execute(
            "INSERT INTO shipment_cost_cache (shipping_id, cost, buyer_cost, bonificacion)"
            " VALUES (:sid, :cost, :buyer_cost, :bonif)"
            " ON CONFLICT (shipping_id) DO UPDATE SET"
            " cost = EXCLUDED.cost, buyer_cost = EXCLUDED.buyer_cost, bonificacion = EXCLUDED.bonificacion",
            {"sid": sid, "cost": seller, "buyer_cost": buyer, "bonif": bonif},
        )


def amount_value(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("amount", "value", "cost"):
            if value.get(key) is not None:
                return amount_value(value.get(key))
        return 0.0
    if isinstance(value, list):
        return sum(amount_value(v) for v in value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def discount_total(discounts) -> float:
    total = 0.0
    for discount in discounts or []:
        total += amount_value(discount.get("promoted_amount"))
    return total


def to_ar(dt_str: str) -> tuple:
    """Convierte datetime ISO de ML a fecha/hora Argentina (UTC-3).
    ML puede devolver -03:00 (ya local) o +00:00 (UTC) según el endpoint."""
    if not dt_str:
        return "", ""
    try:
        dt_naive = datetime.strptime(dt_str[:19], "%Y-%m-%dT%H:%M:%S")
        # Extraer offset saltando milisegundos opcionales: "...T12:16:00.000-03:00"
        tail = dt_str[19:].lstrip(".0123456789")  # quita ".000" si existe
        offset_min = 0
        if len(tail) >= 6 and tail[0] in ("+", "-"):
            sign = -1 if tail[0] == "-" else 1
            h, m = int(tail[1:3]), int(tail[4:6])
            offset_min = sign * (h * 60 + m)
        # dt_naive está en el timezone del offset → convertir a UTC → luego a AR (UTC-3)
        dt_ar = dt_naive - timedelta(minutes=offset_min) - timedelta(hours=3)
        return dt_ar.strftime("%Y-%m-%d"), dt_ar.strftime("%H:%M")
    except Exception:
        return dt_str[:10], ""


async def get_shipping_cost(client, shipping_id, headers) -> dict:
    """Lee el desglose de envío combinando /shipments/{id} y /shipments/{id}/costs.

    /costs sólo expone el costo NETO del vendedor (0 cuando el comprador paga),
    así que para que el panel coincida con lo que muestra ML para cada venta
    usamos /shipments para Ing./Costo Envío y /costs únicamente para la
    bonificación que ML otorga al vendedor.

    Reglas:
      - Flex (home_delivery): Ing. = cost del comprador, Costo = 0,
        Bonif. = senders[0].compensation o, si no viene, list_cost − cost.
      - Colecta/Full con comprador pagando envío (cost > 0):
        Ing. = Costo = cost (ML descuenta al vendedor el mismo monto).
      - Colecta/Full con Envío Gratis (cost = 0):
        Ing. = 0, Costo = senders[0].cost (descuento ya aplicado) o,
        si /costs no responde, 50% del list_cost.
    """
    r_ship_task = asyncio.create_task(
        client.get(f"{ML_API_URL}/shipments/{shipping_id}", headers=headers)
    )
    costs_headers = {**headers, "X-Costs-New": "true", "x-format-new": "true"}
    r_costs_task = asyncio.create_task(
        client.get(f"{ML_API_URL}/shipments/{shipping_id}/costs", headers=costs_headers)
    )
    r_ship = await r_ship_task
    r_costs = await r_costs_task

    if r_ship.status_code != 200:
        return {"seller": 0.0, "buyer": 0.0, "bonificacion": 0.0}

    ship = r_ship.json()
    so = ship.get("shipping_option") or {}
    logistic = ship.get("logistic") or {}
    logistic_type = ship.get("logistic_type") or logistic.get("type") or ""
    mode = ship.get("mode") or logistic.get("mode") or ""
    option_name = str(so.get("name") or so.get("shipping_method_type") or "").lower()
    cost = amount_value(so.get("cost"))
    list_cost = amount_value(so.get("list_cost"))
    base_cost = amount_value(so.get("base_cost") or ship.get("base_cost"))
    is_flex = (
        logistic_type in {"self_service", "home_delivery"}
        or mode == "self_service"
        or "flex" in option_name
    )

    # Extraer datos del endpoint /costs si está disponible
    costs_sender_cost = 0.0
    costs_gross_amount = 0.0
    receiver_cost = 0.0
    sender_discount = 0.0
    compensation = 0.0
    if r_costs.status_code == 200:
        cd = r_costs.json()
        costs_gross_amount = amount_value(cd.get("gross_amount"))
        receiver = cd.get("receiver") or {}
        receiver_cost = amount_value(receiver.get("cost"))
        senders = cd.get("senders") or []
        if senders:
            s0 = senders[0]
            costs_sender_cost = amount_value(s0.get("cost"))
            sender_discount = discount_total(s0.get("discounts"))
            compensation = amount_value(s0.get("compensation"))

    buyer_cost_from_costs = receiver_cost if receiver_cost > 0 else cost

    def gross_candidate(net_cost: float, bonif: float = 0.0) -> float:
        if net_cost > 0 and bonif > 0:
            return net_cost + bonif
        if costs_gross_amount > 0:
            return costs_gross_amount
        if list_cost > 0:
            return list_cost
        if base_cost > 0:
            return base_cost
        return net_cost or cost

    def derive_bonif(net_cost: float, gross_cost: float) -> float:
        if compensation > 0:
            return compensation
        if sender_discount > 0:
            return sender_discount
        if gross_cost > net_cost > 0:
            return gross_cost - net_cost
        return 0.0

    if is_flex:
        # Flex: ML descuenta el list_cost y compensa con la bonificación.
        buyer_cost = buyer_cost_from_costs
        seller_net_cost = costs_sender_cost or list_cost or base_cost or cost
        seller_cost = gross_candidate(seller_net_cost, compensation or sender_discount)
        bonificacion = derive_bonif(seller_net_cost, seller_cost)
        if bonificacion == 0 and buyer_cost == 0 and seller_net_cost > 0 and seller_cost == seller_net_cost:
            seller_cost = round(seller_net_cost / 0.9, 2)
            bonificacion = seller_cost - seller_net_cost
    elif logistic_type in {"xd_drop_off", "drop_off", "cross_docking", "fulfillment"}:
        # Colecta / Full
        if cost > 0:
            buyer_cost = cost
            seller_cost = cost
            bonificacion = 0.0
        else:
            buyer_cost = 0.0
            seller_net_cost = costs_sender_cost
            if costs_sender_cost > 0:
                seller_cost = gross_candidate(seller_net_cost, compensation or sender_discount)
            else:
                discount = so.get("discount") or {}
                if discount.get("promoted_amount") is not None:
                    seller_net_cost = amount_value(discount["promoted_amount"])
                    seller_cost = gross_candidate(seller_net_cost, compensation or sender_discount)
                elif discount.get("rate"):
                    seller_net_cost = list_cost * (1.0 - amount_value(discount["rate"]))
                    seller_cost = list_cost
                else:
                    seller_net_cost = (list_cost or base_cost) * 0.5
                    seller_cost = list_cost or base_cost or seller_net_cost
            bonificacion = derive_bonif(seller_net_cost, seller_cost)
    else:
        # logistic_type vacío o desconocido (común en ventas canceladas).
        buyer_cost = buyer_cost_from_costs
        seller_net_cost = costs_sender_cost or base_cost or list_cost or cost
        seller_cost = gross_candidate(seller_net_cost, compensation or sender_discount)
        bonificacion = derive_bonif(seller_net_cost, seller_cost)

    return {
        "seller": round(seller_cost, 2),
        "buyer": round(buyer_cost, 2),
        "bonificacion": round(bonificacion, 2),
    }


# ── Auth ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if get_session_user_id(request):
        return RedirectResponse("/dashboard")
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/", response_class=HTMLResponse)
async def do_login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = db_fetchone("SELECT * FROM users WHERE email=:e", {"e": email.lower().strip()})
    if not user or not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Email o contraseña incorrectos"})
    session_token = serializer.dumps(user["id"])
    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie("session", session_token, max_age=86400 * 7, httponly=True)
    return response


@app.get("/logout")
async def logout():
    r = RedirectResponse("/")
    r.delete_cookie("session")
    return r


# ── Dashboard ───────────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse("/")
    user = get_user(user_id)
    accounts = db_fetchall("SELECT id, nickname, ml_user_id FROM ml_accounts WHERE user_id=:uid ORDER BY id", {"uid": user_id})
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user, "accounts": accounts})


# ── ML OAuth ────────────────────────────────────────────────────

@app.get("/ml/connect")
async def ml_connect(request: Request):
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse("/")
    state = secrets.token_urlsafe(16)
    url = (f"{ML_AUTH_URL}?response_type=code&client_id={ML_CLIENT_ID}"
           f"&redirect_uri={APP_URL}/ml/callback&state={state}")
    r = RedirectResponse(url)
    r.set_cookie("oauth_state", state, max_age=600, httponly=True)
    return r


@app.get("/ml/callback")
async def ml_callback(request: Request, code: str, state: str):
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse("/")
    if request.cookies.get("oauth_state") != state:
        raise HTTPException(400, "Estado OAuth inválido")
    async with httpx.AsyncClient() as client:
        resp = await client.post(ML_TOKEN_URL, data={
            "grant_type": "authorization_code",
            "client_id": ML_CLIENT_ID,
            "client_secret": ML_CLIENT_SECRET,
            "code": code,
            "redirect_uri": f"{APP_URL}/ml/callback",
        })
        if resp.status_code != 200:
            raise HTTPException(400, f"Error ML: {resp.text}")
        tokens = resp.json()
    ml_user_id = str(tokens["user_id"])
    nickname = tokens.get("nickname", ml_user_id)
    expires_at = datetime.utcnow() + timedelta(seconds=tokens["expires_in"])
    db_execute("""
        INSERT INTO ml_accounts (user_id, ml_user_id, nickname, access_token, refresh_token, expires_at)
        VALUES (:uid,:mlid,:nick,:at,:rt,:ea)
        ON CONFLICT (user_id, ml_user_id) DO UPDATE
        SET access_token=:at, refresh_token=:rt, expires_at=:ea, nickname=:nick
    """, {"uid": user_id, "mlid": ml_user_id, "nick": nickname,
          "at": tokens["access_token"], "rt": tokens["refresh_token"], "ea": expires_at})
    r = RedirectResponse("/dashboard", status_code=303)
    r.delete_cookie("oauth_state")
    return r


@app.post("/ml/disconnect/{account_id}")
async def ml_disconnect(request: Request, account_id: int):
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(401)
    db_execute("DELETE FROM ml_accounts WHERE id=:id AND user_id=:uid", {"id": account_id, "uid": user_id})
    return RedirectResponse("/dashboard", status_code=303)


# ── API datos ───────────────────────────────────────────────────

@app.get("/api/orders/{account_id}")
async def api_orders(request: Request, account_id: int,
                     date_from: Optional[str] = None, date_to: Optional[str] = None):
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(401)
    acc = db_fetchone("SELECT * FROM ml_accounts WHERE id=:id AND user_id=:uid", {"id": account_id, "uid": user_id})
    if not acc:
        raise HTTPException(404)
    token = await refresh_ml_token(account_id)
    if not token:
        raise HTTPException(502)
    headers = {"Authorization": f"Bearer {token}"}

    today = datetime.utcnow().date()
    df = date_from or str(today - timedelta(days=365))
    dt = date_to or str(today)
    # Limitar rango a 1 año máximo
    if (datetime.strptime(dt, "%Y-%m-%d") - datetime.strptime(df, "%Y-%m-%d")).days > 365:
        df = str(datetime.strptime(dt, "%Y-%m-%d").date() - timedelta(days=365))

    # Convertir fechas AR (UTC-3) a UTC para el filtro: medianoche AR = 03:00 UTC del mismo día
    # Se resta 24h al inicio para capturar órdenes cuyo date_created fue el día anterior
    # pero cuyo pago (date_approved) cayó dentro del rango solicitado.
    from_utc = datetime.strptime(df, "%Y-%m-%d") + timedelta(hours=3) - timedelta(hours=24)
    to_utc   = datetime.strptime(dt, "%Y-%m-%d") + timedelta(hours=27)  # 23:59:59 AR = ~03:00 UTC día siguiente
    search_params: dict = {
        "seller": acc["ml_user_id"],
        "sort": "date_desc",
        "limit": 50,
        "order.date_created.from": from_utc.strftime("%Y-%m-%dT%H:%M:%S.000-00:00"),
        "order.date_created.to":   to_utc.strftime("%Y-%m-%dT%H:%M:%S.000-00:00"),
    }

    async with httpx.AsyncClient(timeout=60) as client:
        # Primera página para saber el total
        r = await client.get(f"{ML_API_URL}/orders/search", headers=headers, params={**search_params, "offset": 0})
        if r.status_code != 200:
            raise HTTPException(502)
        first = r.json()
        all_results = list(first.get("results", []))
        total = first.get("paging", {}).get("total", 0)

        # Resto de páginas en paralelo
        offsets = list(range(50, total, 50))
        if offsets:
            page_sem = asyncio.Semaphore(10)
            async def fetch_page(off: int):
                async with page_sem:
                    rp = await client.get(
                        f"{ML_API_URL}/orders/search", headers=headers,
                        params={**search_params, "offset": off},
                    )
                    return rp.json().get("results", []) if rp.status_code == 200 else []
            pages = await asyncio.gather(*[fetch_page(off) for off in offsets])
            for page in pages:
                all_results.extend(page)

        # Costos de envío: primero desde cache, luego API solo para los que faltan
        all_sids = [(o.get("shipping") or {}).get("id") for o in all_results]
        unique_sids = [sid for sid in dict.fromkeys(s for s in all_sids if s)]
        cost_cache = db_get_cached_shipping(unique_sids)

        uncached = [sid for sid in unique_sids if sid not in cost_cache]
        if uncached:
            ship_sem = asyncio.Semaphore(15)
            async def fetch_ship(sid):
                async with ship_sem:
                    return sid, await get_shipping_cost(client, sid, headers)
            new_costs = dict(await asyncio.gather(*[fetch_ship(sid) for sid in uncached]))
            db_save_shipping_costs(new_costs)
            cost_cache.update(new_costs)

    empty_ship = {"seller": 0.0, "buyer": 0.0, "bonificacion": 0.0}

    # ── Paso 1: construir lista raw por orden individual ──────────
    raw_list = []
    for o, sid in zip(all_results, all_sids):
        a = float(o.get("total_amount", 0))
        estado = o.get("status", "")
        payments = o.get("payments", [])
        pay_str = ""
        if payments:
            pay_str = payments[0].get("date_approved", "") or payments[0].get("date_created", "")
        fecha, hora = to_ar(pay_str if pay_str else o.get("date_created", ""))
        if fecha and not (df <= fecha <= dt):
            continue
        comision = round(sum(float(i.get("sale_fee", 0)) for i in o.get("order_items", [])), 2)
        ship_info = cost_cache.get(sid, empty_ship) if sid else empty_ship
        items = []
        for i in o.get("order_items", []):
            sku = (i.get("item", {}).get("seller_sku") or "").strip()
            qty = int(i.get("quantity", 1))
            items.append({
                "sku": sku,
                "titulo": i.get("item", {}).get("title", "?"),
                "monto": round(float(i.get("unit_price", 0)) * qty, 2),
                "comision": round(float(i.get("sale_fee", 0)), 2),
                "cantidad": qty,
            })
        raw_list.append({
            "id": o.get("id"),
            "pack_id": o.get("pack_id"),
            "fecha": fecha,
            "hora": hora,
            "monto": round(a, 2),
            "comision": comision,
            "envio": ship_info["seller"],
            "shipping_buyer": round(ship_info["buyer"], 2),
            "bonificacion": round(ship_info.get("bonificacion", 0), 2),
            "coupon_amt": round(float((o.get("coupon") or {}).get("amount", 0)), 2),
            "estado": estado,
            "items": items,
        })

    # ── Paso 2: agrupar packs (mismo pack_id → un solo envío) ─────
    orders = []
    pack_map: dict = {}
    for raw in raw_list:
        pid = raw["pack_id"]
        if pid:
            if pid not in pack_map:
                pack_map[pid] = {
                    "id": pid, "venta_id": pid,
                    "fecha": raw["fecha"], "hora": raw["hora"],
                    "monto": 0.0, "comision": 0.0,
                    "envio": raw["envio"],            # un envío por pack
                    "shipping_buyer": raw["shipping_buyer"],
                    "bonificacion": raw["bonificacion"],
                    "coupon_total": 0.0,
                    "estado": raw["estado"],
                    "is_pack": True, "items": [],
                }
            p = pack_map[pid]
            p["monto"] = round(p["monto"] + raw["monto"], 2)
            p["comision"] = round(p["comision"] + raw["comision"], 2)
            p["coupon_total"] = round(p["coupon_total"] + raw["coupon_amt"], 2)
            p["items"].extend(raw["items"])
        else:
            # Ing. Envío = sólo lo que paga el comprador por envío (sin cupones).
            ingreso_envio = raw["shipping_buyer"]
            sku_col = " / ".join(i["sku"] or i["titulo"] for i in raw["items"])
            ganancia = round(
                raw["monto"] + ingreso_envio + raw["bonificacion"]
                + raw["coupon_amt"] - raw["comision"] - raw["envio"], 2
            )
            orders.append({
                "id": raw["id"], "venta_id": raw["id"],
                "fecha": raw["fecha"], "hora": raw["hora"],
                "producto": sku_col,
                "monto": raw["monto"], "comision": raw["comision"],
                "ingreso_envio": ingreso_envio,
                "bonificacion": raw["bonificacion"],
                "envio": raw["envio"],
                "ganancia": ganancia,
                "estado": raw["estado"],
                "is_pack": False, "items": raw["items"],
            })

    for pid, p in pack_map.items():
        ingreso_envio = p["shipping_buyer"]
        p["ingreso_envio"] = ingreso_envio
        p["ganancia"] = round(
            p["monto"] + ingreso_envio + p["bonificacion"]
            + p["coupon_total"] - p["comision"] - p["envio"], 2
        )
        if len(p["items"]) == 1:
            p["is_pack"] = False
            p["producto"] = p["items"][0]["sku"] or p["items"][0]["titulo"]
        else:
            p["producto"] = f"Paquete ({len(p['items'])} productos)"
        del p["shipping_buyer"], p["coupon_total"]
        orders.append(p)

    orders.sort(key=lambda x: (x.get("fecha") or "", x.get("hora") or ""), reverse=True)

    # ── Paso 3: agregados diarios y top productos ─────────────────
    daily: dict = {}
    products: dict = {}
    for order in orders:
        if order.get("fecha") and order.get("estado") == "paid":
            f = order["fecha"]
            daily.setdefault(f, {"ventas": 0, "ingresos": 0, "ganancia": 0})
            daily[f]["ventas"] += 1
            daily[f]["ingresos"] += order["monto"]
            daily[f]["ganancia"] += order["ganancia"]
            for item in order.get("items", []):
                t = item.get("sku") or item.get("titulo", "Sin título")
                products.setdefault(t, {"cantidad": 0, "ingresos": 0})
                products[t]["cantidad"] += item.get("cantidad", 1)
                products[t]["ingresos"] += item.get("monto", 0)

    top = sorted(products.items(), key=lambda x: x[1]["ingresos"], reverse=True)[:5]
    return {
        "orders": orders,
        "daily": dict(sorted(daily.items())),
        "top_products": [{"nombre": k, **v} for k, v in top],
        "ultima_actualizacion": datetime.utcnow().isoformat(),
    }


# ── Admin ────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse("/")
    user = get_user(user_id)
    if not user["is_admin"]:
        raise HTTPException(403)
    users = db_fetchall("""
        SELECT u.id, u.email, u.name, u.is_admin, u.created_at, COUNT(m.id) as ml_count
        FROM users u LEFT JOIN ml_accounts m ON m.user_id = u.id
        GROUP BY u.id ORDER BY u.created_at DESC
    """)
    success = request.query_params.get("success")
    return templates.TemplateResponse("admin.html", {"request": request, "users": users, "error": None, "success": success})


@app.post("/admin/users/create", response_class=HTMLResponse)
async def admin_create_user(request: Request, name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse("/")
    user = get_user(user_id)
    if not user["is_admin"]:
        raise HTTPException(403)
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        db_execute(
            "INSERT INTO users (email, password_hash, name) VALUES (:e,:h,:n)",
            {"e": email.lower().strip(), "h": hashed, "n": name}
        )
        return RedirectResponse("/admin?success=1", status_code=303)
    except Exception:
        users = db_fetchall("""
            SELECT u.id, u.email, u.name, u.is_admin, u.created_at, COUNT(m.id) as ml_count
            FROM users u LEFT JOIN ml_accounts m ON m.user_id = u.id
            GROUP BY u.id ORDER BY u.created_at DESC
        """)
        return templates.TemplateResponse("admin.html", {"request": request, "users": users, "error": "El email ya existe", "success": None})


@app.post("/admin/users/delete/{target_id}")
async def admin_delete_user(request: Request, target_id: int):
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(401)
    user = get_user(user_id)
    if not user["is_admin"]:
        raise HTTPException(403)
    if target_id == user_id:
        raise HTTPException(400, "No podés eliminarte a vos mismo")
    db_execute("DELETE FROM users WHERE id=:id", {"id": target_id})
    return RedirectResponse("/admin", status_code=303)
