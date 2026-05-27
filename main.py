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
                cached_at TIMESTAMP DEFAULT NOW()
            )
        """))
        # Migraciones para bases existentes
        conn.execute(text("""
            ALTER TABLE shipment_cost_cache ADD COLUMN IF NOT EXISTS buyer_cost NUMERIC(10,2) DEFAULT NULL
        """))
        # Registros viejos tienen buyer_cost=0 por DEFAULT; marcarlos NULL para que se re-fetcheen
        conn.execute(text("""
            UPDATE shipment_cost_cache SET buyer_cost = NULL WHERE buyer_cost = 0
        """))
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
            f"SELECT shipping_id, cost, buyer_cost FROM shipment_cost_cache WHERE shipping_id IN ({placeholders})",
            params,
        )
        for row in rows:
            # Solo se considera "en cache" si ya tiene buyer_cost (no NULL)
            if row["buyer_cost"] is not None:
                result[row["shipping_id"]] = {"seller": float(row["cost"]), "buyer": float(row["buyer_cost"])}
    return result


def db_save_shipping_costs(costs: dict):
    for sid, c in costs.items():
        seller = c["seller"] if isinstance(c, dict) else float(c)
        buyer = c["buyer"] if isinstance(c, dict) else 0.0
        db_execute(
            "INSERT INTO shipment_cost_cache (shipping_id, cost, buyer_cost) VALUES (:sid, :cost, :buyer_cost)"
            " ON CONFLICT (shipping_id) DO UPDATE SET buyer_cost = EXCLUDED.buyer_cost",
            {"sid": sid, "cost": seller, "buyer_cost": buyer},
        )


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
    r = await client.get(f"{ML_API_URL}/shipments/{shipping_id}", headers=headers)
    if r.status_code != 200:
        return {"seller": 0.0, "buyer": 0.0}
    data = r.json()
    seller_cost = data.get("base_cost") or 0
    buyer_cost = (data.get("shipping_option") or {}).get("cost") or 0
    return {
        "seller": round(float(seller_cost), 2),
        "buyer": round(float(buyer_cost), 2),
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

    orders = []
    daily: dict = {}
    products: dict = {}
    empty_ship = {"seller": 0.0, "buyer": 0.0}
    for o, sid in zip(all_results, all_sids):
        a = o.get("total_amount", 0)
        estado = o.get("status", "")
        # Usar fecha de pago aprobado para mostrar; caer en date_created si no hay pago
        payments = o.get("payments", [])
        pay_str = ""
        if payments:
            pay_str = payments[0].get("date_approved", "") or payments[0].get("date_created", "")
        fecha, hora = to_ar(pay_str if pay_str else o.get("date_created", ""))
        # Filtro client-side sobre fecha de pago: solo dentro del rango AR solicitado
        if fecha and not (df <= fecha <= dt):
            continue
        comision = round(sum(item.get("sale_fee", 0) for item in o.get("order_items", [])), 2)
        ship_info = cost_cache.get(sid, empty_ship) if sid else empty_ship
        envio = ship_info["seller"]
        ingreso_envio = ship_info["buyer"]
        ganancia = round(a + ingreso_envio - comision - envio, 2)
        orders.append({
            "id": o.get("id"),
            "venta_id": o.get("pack_id") or o.get("id"),
            "fecha": fecha,
            "hora": hora,
            "producto": ", ".join(i.get("item", {}).get("title", "?") for i in o.get("order_items", [])),
            "monto": round(a, 2),
            "comision": comision,
            "ingreso_envio": ingreso_envio,
            "envio": envio,
            "ganancia": ganancia,
            "estado": estado,
        })
        if fecha and estado == "paid":
            daily.setdefault(fecha, {"ventas": 0, "ingresos": 0, "ganancia": 0})
            daily[fecha]["ventas"] += 1
            daily[fecha]["ingresos"] += a
            daily[fecha]["ganancia"] += ganancia
            for item in o.get("order_items", []):
                t = item.get("item", {}).get("title", "Sin título")
                products.setdefault(t, {"cantidad": 0, "ingresos": 0})
                products[t]["cantidad"] += item.get("quantity", 1)
                products[t]["ingresos"] += item.get("unit_price", 0) * item.get("quantity", 1)
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
