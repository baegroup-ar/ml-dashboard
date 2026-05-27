import os
import httpx
import secrets
import asyncio
import json
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
SHIPPING_LOGIC_VERSION = "v15-net-cost-envio-gratis"

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
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS order_snapshot_cache (
                account_id INTEGER NOT NULL REFERENCES ml_accounts(id) ON DELETE CASCADE,
                order_id TEXT NOT NULL,
                paid_date DATE NOT NULL,
                paid_time TEXT,
                payload JSONB NOT NULL,
                details_complete BOOLEAN DEFAULT FALSE,
                updated_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (account_id, order_id)
            )
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_order_snapshot_cache_account_date ON order_snapshot_cache (account_id, paid_date)"))
        # Invalida el caché cuando cambia la lógica de cálculo de envío.
        current = conn.execute(text("SELECT value FROM app_meta WHERE key='shipping_logic_version'")).fetchone()
        if not current or current[0] != SHIPPING_LOGIC_VERSION:
            conn.execute(text("DELETE FROM shipment_cost_cache"))
            # También limpia el snapshot de órdenes para recalcular bonificaciones.
            conn.execute(text("DELETE FROM order_snapshot_cache"))
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


def db_fetch_order_snapshots(account_id: int, date_from: str, date_to: str) -> list:
    rows = db_fetchall("""
        SELECT payload FROM order_snapshot_cache
        WHERE account_id=:aid AND paid_date BETWEEN :df AND :dt
        ORDER BY paid_date DESC, paid_time DESC
    """, {"aid": account_id, "df": date_from, "dt": date_to})
    orders = []
    for row in rows:
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        orders.append(payload)
    return orders


def db_save_order_snapshots(account_id: int, orders: list, details_complete: bool):
    if not orders:
        return
    params_list = [
        {
            "aid": account_id,
            "oid": str(order["id"]),
            "paid_date": order["fecha"],
            "paid_time": order.get("hora") or "",
            "payload": json.dumps(order),
            "complete": details_complete,
        }
        for order in orders
    ]
    # Una sola conexión + executemany en vez de N conexiones.
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO order_snapshot_cache
                (account_id, order_id, paid_date, paid_time, payload, details_complete, updated_at)
            VALUES (:aid, :oid, :paid_date, :paid_time, CAST(:payload AS JSONB), :complete, NOW())
            ON CONFLICT (account_id, order_id) DO UPDATE SET
                paid_date=EXCLUDED.paid_date,
                paid_time=EXCLUDED.paid_time,
                payload=EXCLUDED.payload,
                details_complete=EXCLUDED.details_complete,
                updated_at=NOW()
        """), params_list)
        conn.commit()


def build_dashboard_payload(orders: list, details_complete: bool):
    orders = sorted(orders, key=lambda x: (x.get("fecha") or "", x.get("hora") or ""), reverse=True)


    daily: dict = {}
    products: dict = {}
    for order in orders:
        if order.get("fecha") and order.get("estado") == "paid":
            f = order["fecha"]
            daily.setdefault(f, {"ventas": 0, "ingresos": 0, "ganancia": 0})
            daily[f]["ventas"] += 1
            daily[f]["ingresos"] += order.get("monto", 0)
            daily[f]["ganancia"] += order.get("ganancia", 0)
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
        "details_complete": details_complete,
        "ultima_actualizacion": datetime.utcnow().isoformat(),
    }


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
    if not costs:
        return
    params_list = [
        {
            "sid": sid,
            "cost": float(c.get("seller", 0)),
            "buyer_cost": float(c.get("buyer", 0)),
            "bonif": float(c.get("bonificacion", 0)),
        }
        for sid, c in costs.items()
    ]
    # Una sola conexión + executemany en vez de N conexiones.
    with engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO shipment_cost_cache (shipping_id, cost, buyer_cost, bonificacion)"
            " VALUES (:sid, :cost, :buyer_cost, :bonif)"
            " ON CONFLICT (shipping_id) DO UPDATE SET"
            " cost = EXCLUDED.cost, buyer_cost = EXCLUDED.buyer_cost, bonificacion = EXCLUDED.bonificacion"
        ), params_list)
        conn.commit()


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


def order_ids_from_sales_info(row: dict) -> list:
    ids = []
    for sale in row.get("sales_info") or []:
        oid = sale.get("order_id")
        if oid is not None:
            ids.append(str(oid))
    if row.get("order_id") is not None:
        ids.append(str(row.get("order_id")))
    return ids


def billing_amount(row: dict, key: str, default=0.0) -> float:
    charge = row.get("charge_info") or {}
    return amount_value(charge.get(key, row.get(key, default)))


def billing_text(row: dict, key: str) -> str:
    charge = row.get("charge_info") or {}
    return str(charge.get(key, row.get(key, "")) or "").lower()


def apply_billing_row(target: dict, row: dict):
    charge = row.get("charge_info") or {}
    discount = row.get("discount_info") or {}
    shipping = row.get("shipping_info") or {}
    detail_type = billing_text(row, "detail_type").upper()
    sub_type = billing_text(row, "detail_sub_type").upper()
    concept_type = billing_text(row, "concept_type").upper()
    transaction_detail = billing_text(row, "transaction_detail")
    detail_amount = abs(billing_amount(row, "detail_amount"))
    amount_without_discount = abs(amount_value(discount.get("charge_amount_without_discount")))
    discount_amount = abs(amount_value(discount.get("discount_amount")))

    # Una fila BONUS NUNCA se suma a "envio" — sólo a "bonificacion".
    # Si no excluimos BONUS de is_shipping, su detail_amount se sumaba
    # también a envio y duplicaba el costo (ej. #2000016639025652).
    is_bonus = detail_type == "BONUS"
    is_shipping_concept = (
        concept_type == "SHIPPING"
        or sub_type in {"CXD", "CFF", "BXD", "BFF"}
        or "envío" in transaction_detail
        or "envio" in transaction_detail
        or "shipping" in transaction_detail
    )
    is_shipping = (not is_bonus) and is_shipping_concept
    is_sale_charge = (
        (not is_shipping) and (not is_bonus)
        and detail_type == "CHARGE"
        and (sub_type.startswith("CV") or "cargo por venta" in transaction_detail or "cargo por vender" in transaction_detail)
    )
    if is_sale_charge:
        target["comision"] += detail_amount
        target["has_comision"] = True
    if is_shipping:
        target["envio"] += amount_without_discount or detail_amount
        target["bonificacion"] += discount_amount
        target["has_shipping"] = True
        receiver_cost = amount_value(shipping.get("receiver_shipping_cost"))
        if receiver_cost > 0:
            target["ingreso_envio"] = max(target["ingreso_envio"], receiver_cost)
    elif is_bonus and is_shipping_concept:
        target["bonificacion"] += detail_amount
        target["has_shipping"] = True


def period_key_from_ml_date(value: str) -> str:
    if not value:
        return ""
    return f"{value[:7]}-01"


def flex_order_id(row: dict) -> str:
    shipping = row.get("shipping_info") or {}
    order = shipping.get("order") or {}
    oid = order.get("order_id") or row.get("order_id")
    return str(oid) if oid is not None else ""


def apply_flex_billing_row(target: dict, row: dict):
    charge = row.get("charge_info") or {}
    shipping = row.get("shipping_info") or {}
    detail_type = str(charge.get("detail_type") or row.get("detail_type") or "").upper()
    transaction_detail = str(charge.get("transaction_detail") or row.get("transaction_detail") or "").lower()
    amount = abs(amount_value(charge.get("detail_amount", row.get("detail_amount"))))
    receiver_cost = amount_value(shipping.get("receiver_shipping_cost"))
    if receiver_cost > 0:
        target["ingreso_envio"] = max(target["ingreso_envio"], receiver_cost)
    if not amount:
        return
    is_cancellation = "anulaci" in transaction_detail or "cancellation" in transaction_detail
    is_bonus = detail_type == "BONUS" or ("bonific" in transaction_detail and not is_cancellation)
    if is_bonus:
        target["bonificacion"] += amount
        target["has_flex"] = True
    elif is_cancellation or detail_type == "CHARGE":
        target["envio"] += amount
        target["has_flex_charge"] = True


async def fetch_flex_billing_by_order(client, headers, order_periods: dict) -> dict:
    if not order_periods:
        return {}
    parsed = {}
    period_map = {}
    for oid, key in order_periods.items():
        if key:
            period_map.setdefault(key, []).append(str(oid))

    document_types = ("BILL", "CREDIT_NOTE", "DEBIT_NOTE")
    for key, ids in period_map.items():
        unique_ids = list(dict.fromkeys(ids))
        for i in range(0, len(unique_ids), 100):
            batch = unique_ids[i:i+100]
            for document_type in document_types:
                params = {
                    "document_type": document_type,
                    "order_ids": ",".join(batch),
                    "limit": 1000,
                }
                r = await client.get(
                    f"{ML_API_URL}/billing/integration/periods/key/{key}/group/ML/flex/details",
                    headers=headers,
                    params=params,
                )
                if r.status_code not in (200, 206):
                    continue
                for row in r.json().get("results", []):
                    oid = flex_order_id(row)
                    if not oid:
                        continue
                    parsed.setdefault(oid, {
                        "envio": 0.0, "bonificacion": 0.0, "ingreso_envio": 0.0,
                        "has_flex": False, "has_flex_charge": False,
                    })
                    apply_flex_billing_row(parsed[oid], row)
    return {
        oid: {
            k: (round(v, 2) if isinstance(v, (int, float)) else v)
            for k, v in values.items()
        }
        for oid, values in parsed.items()
    }


async def fetch_billing_by_order(client, headers, seller_id, order_ids: list) -> dict:
    if not order_ids:
        return {}
    parsed = {}
    unique_ids = list(dict.fromkeys(str(oid) for oid in order_ids if oid))
    for i in range(0, len(unique_ids), 100):
        batch = unique_ids[i:i+100]
        params = {
            "order_ids": ",".join(batch),
            "seller_id": seller_id,
            "limit": 150,
        }
        r = await client.get(f"{ML_API_URL}/billing/integration/group/ML/order/details", headers=headers, params=params)
        if r.status_code not in (200, 206):
            continue
        data = r.json()
        for row in data.get("results", []):
            for oid in order_ids_from_sales_info(row):
                parsed.setdefault(oid, {
                    "comision": 0.0, "envio": 0.0, "bonificacion": 0.0, "ingreso_envio": 0.0,
                    "has_comision": False, "has_shipping": False,
                })
                apply_billing_row(parsed[oid], row)
    return {
        oid: {
            k: (round(v, 2) if isinstance(v, (int, float)) else v)
            for k, v in values.items()
        }
        for oid, values in parsed.items()
    }


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
    option_discount = so.get("discount") or {}
    option_discount_amount = amount_value(option_discount.get("promoted_amount"))
    option_discount_rate = amount_value(option_discount.get("rate"))
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
        if option_discount_amount > 0:
            return option_discount_amount
        return net_cost or cost

    def derive_bonif(net_cost: float, gross_cost: float) -> float:
        if compensation > 0:
            return compensation
        if sender_discount > 0:
            return sender_discount
        if option_discount_amount > 0:
            return option_discount_amount
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
            bonificacion = seller_cost
            seller_cost = 0.0
    elif logistic_type in {"xd_drop_off", "drop_off", "cross_docking", "fulfillment"}:
        # Colecta / Full
        if cost > 0:
            buyer_cost = cost
            seller_cost = cost
            bonificacion = 0.0
        else:
            # Envío Gratis: el Costo Envío que muestra ML es el monto NETO que
            # le descuenta al vendedor (senders[0].cost en /shipments/{id}/costs),
            # NO el gross_amount (precio de lista antes del descuento).
            buyer_cost = 0.0
            if costs_sender_cost > 0:
                seller_cost = costs_sender_cost
            elif option_discount_amount > 0:
                seller_cost = option_discount_amount
            elif option_discount_rate:
                seller_cost = list_cost * (1.0 - option_discount_rate)
            else:
                seller_cost = (list_cost or base_cost) * 0.5
            # Bonificación: la que expone /costs si está; si no, asumimos que
            # ML bonifica al vendedor el mismo monto que le cobra (envío
            # gratis netea a $0 igual que en el panel de ML).
            bonificacion = compensation if compensation > 0 else seller_cost
    else:
        # logistic_type vacío o desconocido (común en ventas canceladas).
        buyer_cost = buyer_cost_from_costs
        seller_net_cost = costs_sender_cost or base_cost or list_cost or cost
        seller_cost = gross_candidate(seller_net_cost, compensation or sender_discount)
        bonificacion = derive_bonif(seller_net_cost, seller_cost)
        # Envío gratis sin tipo identificado: tratar como colecta envío gratis
        # y reflejar la bonificación equivalente al costo de envío.
        if bonificacion == 0 and buyer_cost == 0 and seller_cost > 0:
            bonificacion = seller_cost

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
                     date_from: Optional[str] = None, date_to: Optional[str] = None,
                     refresh: bool = False, fast: bool = False):
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(401)
    acc = db_fetchone("SELECT * FROM ml_accounts WHERE id=:id AND user_id=:uid", {"id": account_id, "uid": user_id})
    if not acc:
        raise HTTPException(404)

    today = datetime.utcnow().date()
    df = date_from or str(today - timedelta(days=365))
    dt = date_to or str(today)
    # Limitar rango a 1 año máximo
    if (datetime.strptime(dt, "%Y-%m-%d") - datetime.strptime(df, "%Y-%m-%d")).days > 365:
        df = str(datetime.strptime(dt, "%Y-%m-%d").date() - timedelta(days=365))

    # Siempre servimos el caché de Postgres si existe (lectura local <100ms).
    # Sólo vamos contra la API de ML cuando se pide refresh explícito o no
    # hay caché. El auto-refresh del modo Hoy ya manda refresh=1 cada 2 min,
    # y el botón Actualizar también lo fuerza.
    cached_orders = db_fetch_order_snapshots(account_id, df, dt)
    if cached_orders and not refresh:
        return build_dashboard_payload(cached_orders, details_complete=True)

    token = await refresh_ml_token(account_id)
    if not token:
        # Si tenemos caché pero no token, igual servimos el caché.
        if cached_orders:
            return build_dashboard_payload(cached_orders, details_complete=True)
        raise HTTPException(502)
    headers = {"Authorization": f"Bearer {token}"}

    from_utc = datetime.strptime(df, "%Y-%m-%d") + timedelta(hours=3) - timedelta(hours=24)
    to_utc   = datetime.strptime(dt, "%Y-%m-%d") + timedelta(hours=27)  # 23:59:59 AR = ~03:00 UTC día siguiente
    search_params: dict = {
        "seller": acc["ml_user_id"],
        "sort": "date_desc",
        "limit": 50,
        "order.date_created.from": from_utc.strftime("%Y-%m-%dT%H:%M:%S.000-00:00"),
        "order.date_created.to":   to_utc.strftime("%Y-%m-%dT%H:%M:%S.000-00:00"),
    }

    async with httpx.AsyncClient(timeout=90) as client:
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
        cost_cache = {}
        billing_by_order = {}
        flex_billing_by_order = {}
        new_shipping_costs: dict = {}
        if not fast:
            unique_sids = [sid for sid in dict.fromkeys(s for s in all_sids if s)]
            cost_cache = db_get_cached_shipping(unique_sids)
            uncached = [sid for sid in unique_sids if sid not in cost_cache]

            # Bajar envíos faltantes, billing normal y billing flex en paralelo.
            ship_sem = asyncio.Semaphore(25)
            async def fetch_ship(sid):
                async with ship_sem:
                    return sid, await get_shipping_cost(client, sid, headers)

            async def gather_ships():
                if not uncached:
                    return {}
                return dict(await asyncio.gather(*[fetch_ship(sid) for sid in uncached]))

            order_ids_for_billing = [str(o.get("id")) for o in all_results if o.get("id")]
            order_period_map = {
                str(o.get("id")): period_key_from_ml_date(o.get("date_created", ""))
                for o in all_results
                if o.get("id")
            }

            # Si alguno de los billing endpoints falla, no rompemos la respuesta;
            # caemos al cálculo desde /shipments + /shipments/.../costs.
            results = await asyncio.gather(
                gather_ships(),
                fetch_billing_by_order(client, headers, acc["ml_user_id"], order_ids_for_billing),
                fetch_flex_billing_by_order(client, headers, order_period_map),
                return_exceptions=True,
            )
            new_shipping_costs = results[0] if not isinstance(results[0], Exception) else {}
            billing_by_order = results[1] if not isinstance(results[1], Exception) else {}
            flex_billing_by_order = results[2] if not isinstance(results[2], Exception) else {}
            cost_cache.update(new_shipping_costs)

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
        order_id = str(o.get("id"))
        billing = billing_by_order.get(order_id, {})
        flex_billing = flex_billing_by_order.get(order_id, {})
        comision = round(sum(
            float(i.get("sale_fee", 0)) * int(i.get("quantity", 1))
            for i in o.get("order_items", [])
        ), 2)
        ship_info = cost_cache.get(sid, empty_ship) if sid else empty_ship
        if billing.get("has_shipping"):
            envio = billing.get("envio", 0)
            ingreso_envio = billing.get("ingreso_envio", 0)
            bonificacion = billing.get("bonificacion", 0)
        else:
            envio = ship_info["seller"]
            ingreso_envio = ship_info["buyer"]
            bonificacion = ship_info.get("bonificacion", 0)
        if flex_billing.get("has_flex") or flex_billing.get("has_flex_charge"):
            bonificacion = flex_billing.get("bonificacion", bonificacion)
            ingreso_envio = flex_billing.get("ingreso_envio", ingreso_envio) or ingreso_envio
            if flex_billing.get("has_flex_charge"):
                envio = flex_billing.get("envio", envio)
            elif abs(envio - bonificacion) < 0.01:
                envio = 0.0
        if billing.get("has_comision"):
            comision = billing.get("comision", comision)
        # Fallback envío gratis: cuando el comprador no pagó envío y nos quedó
        # un costo de envío sin bonificación identificada, ML bonifica al
        # vendedor el mismo monto (ej. #2000013189550281 → Bonif = $6.490).
        if ingreso_envio == 0 and envio > 0 and bonificacion == 0:
            bonificacion = envio
        items = []
        for i in o.get("order_items", []):
            sku = (i.get("item", {}).get("seller_sku") or "").strip()
            qty = int(i.get("quantity", 1))
            items.append({
                "sku": sku,
                "titulo": i.get("item", {}).get("title", "?"),
                "monto": round(float(i.get("unit_price", 0)) * qty, 2),
                "comision": round(float(i.get("sale_fee", 0)) * qty, 2),
                "cantidad": qty,
            })
        raw_list.append({
            "id": o.get("id"),
            "pack_id": o.get("pack_id"),
            "fecha": fecha,
            "hora": hora,
            "monto": round(a, 2),
            "comision": comision,
            "envio": round(envio, 2),
            "shipping_buyer": round(ingreso_envio, 2),
            "bonificacion": round(bonificacion, 2),
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
    if not fast:
        # Persistir caches en background para no demorar la respuesta al cliente.
        if new_shipping_costs:
            asyncio.create_task(asyncio.to_thread(db_save_shipping_costs, new_shipping_costs))
        asyncio.create_task(
            asyncio.to_thread(db_save_order_snapshots, account_id, orders, True)
        )
    return build_dashboard_payload(orders, details_complete=not fast)


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
        "details_complete": not fast,
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


# ── Debug ───────────────────────────────────────────────────────

@app.get("/api/debug/order/{order_id}")
async def debug_order(request: Request, order_id: str):
    """Devuelve TODA la data cruda que devuelve ML para una orden puntual.

    Probá todos los endpoints relevantes para entender de dónde sacar
    la bonificación cuando no aparece en el panel:
      - /orders/{id}
      - /shipments/{shipping_id}
      - /shipments/{shipping_id}/costs
      - /billing/integration/group/ML/order/details (por seller_id)
      - /billing/integration/periods/key/{period}/group/ML/flex/details
      - /orders/{id}/discounts
      - /orders/{id}/feedback
    """
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(401)
    accs = db_fetchall("SELECT * FROM ml_accounts WHERE user_id=:uid ORDER BY id", {"uid": user_id})
    if not accs:
        raise HTTPException(404, "No tenés cuentas de ML conectadas")

    out: dict = {"order_id": order_id, "tries": []}
    for acc in accs:
        token = await refresh_ml_token(acc["id"])
        if not token:
            continue
        headers = {"Authorization": f"Bearer {token}"}
        costs_headers = {**headers, "X-Costs-New": "true", "x-format-new": "true"}
        async with httpx.AsyncClient(timeout=30) as client:
            entry: dict = {"account": acc["nickname"], "account_id": acc["id"]}
            r_order = await client.get(f"{ML_API_URL}/orders/{order_id}", headers=headers)
            entry["order_status"] = r_order.status_code
            if r_order.status_code != 200:
                entry["order_error"] = r_order.text[:500]
                out["tries"].append(entry)
                continue
            order = r_order.json()
            entry["order"] = order
            shipping_id = (order.get("shipping") or {}).get("id")
            entry["shipping_id"] = shipping_id

            tasks = {}
            if shipping_id:
                tasks["shipment"] = client.get(f"{ML_API_URL}/shipments/{shipping_id}", headers=headers)
                tasks["shipment_costs"] = client.get(f"{ML_API_URL}/shipments/{shipping_id}/costs", headers=costs_headers)
            tasks["billing"] = client.get(
                f"{ML_API_URL}/billing/integration/group/ML/order/details",
                headers=headers,
                params={"order_ids": order_id, "seller_id": acc["ml_user_id"], "limit": 150},
            )
            period_key = period_key_from_ml_date(order.get("date_created", ""))
            if period_key:
                tasks["flex_billing_BILL"] = client.get(
                    f"{ML_API_URL}/billing/integration/periods/key/{period_key}/group/ML/flex/details",
                    headers=headers,
                    params={"document_type": "BILL", "order_ids": order_id, "limit": 1000},
                )
                tasks["flex_billing_CREDIT"] = client.get(
                    f"{ML_API_URL}/billing/integration/periods/key/{period_key}/group/ML/flex/details",
                    headers=headers,
                    params={"document_type": "CREDIT_NOTE", "order_ids": order_id, "limit": 1000},
                )
                # También probar SIN document_type
                tasks["flex_billing_NO_TYPE"] = client.get(
                    f"{ML_API_URL}/billing/integration/periods/key/{period_key}/group/ML/flex/details",
                    headers=headers,
                    params={"order_ids": order_id, "limit": 1000},
                )
            tasks["discounts"] = client.get(f"{ML_API_URL}/orders/{order_id}/discounts", headers=headers)
            tasks["feedback"] = client.get(f"{ML_API_URL}/orders/{order_id}/feedback", headers=headers)

            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, res in zip(tasks.keys(), results):
                if isinstance(res, Exception):
                    entry[name] = {"error": str(res)}
                else:
                    try:
                        entry[f"{name}_status"] = res.status_code
                        entry[name] = res.json() if res.status_code in (200, 206) else res.text[:1000]
                    except Exception as e:
                        entry[name] = {"parse_error": str(e), "text": res.text[:500]}

            out["tries"].append(entry)
            return out
    return out


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
