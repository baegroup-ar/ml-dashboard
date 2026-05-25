import os
import httpx
import secrets
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeTimedSerializer, BadSignature

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

ML_CLIENT_ID = os.environ["ML_CLIENT_ID"]
ML_CLIENT_SECRET = os.environ["ML_CLIENT_SECRET"]
APP_URL = os.environ.get("APP_URL", "http://localhost:8000")
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))

ML_AUTH_URL = "https://auth.mercadolibre.com.ar/authorization"
ML_TOKEN_URL = "https://api.mercadolibre.com/oauth/token"
ML_API_URL = "https://api.mercadolibre.com"

serializer = URLSafeTimedSerializer(SECRET_KEY)
token_store: dict = {}


def get_session_user(request: Request) -> Optional[str]:
    session_token = request.cookies.get("session")
    if not session_token:
        return None
    try:
        user_id = serializer.loads(session_token, max_age=86400 * 7)
        return user_id if user_id in token_store else None
    except BadSignature:
        return None


async def refresh_token_if_needed(user_id: str):
    data = token_store.get(user_id)
    if not data:
        return False
    if datetime.utcnow() < data["expires_at"] - timedelta(minutes=5):
        return True
    async with httpx.AsyncClient() as client:
        resp = await client.post(ML_TOKEN_URL, data={
            "grant_type": "refresh_token",
            "client_id": ML_CLIENT_ID,
            "client_secret": ML_CLIENT_SECRET,
            "refresh_token": data["refresh_token"],
        })
        if resp.status_code != 200:
            return False
        tokens = resp.json()
        token_store[user_id]["access_token"] = tokens["access_token"]
        token_store[user_id]["refresh_token"] = tokens.get("refresh_token", data["refresh_token"])
        token_store[user_id]["expires_at"] = datetime.utcnow() + timedelta(seconds=tokens["expires_in"])
        return True


async def get_order_fees(client: httpx.AsyncClient, order_id: str, headers: dict) -> dict:
    """Obtiene comisiones y costo de envío real de una orden."""
    fees_resp = await client.get(f"{ML_API_URL}/orders/{order_id}", headers=headers)
    if fees_resp.status_code != 200:
        return {"comision": 0, "envio": 0}
    data = fees_resp.json()

    # Comisiones ML (marketplace_fee)
    comision = 0
    for fee in data.get("fees", []):
        comision += abs(fee.get("amount", 0))

    # Costo de envío
    envio = 0
    shipping = data.get("shipping", {}) or {}
    shipping_id = shipping.get("id")
    if shipping_id:
        ship_resp = await client.get(f"{ML_API_URL}/shipments/{shipping_id}", headers=headers)
        if ship_resp.status_code == 200:
            ship_data = ship_resp.json()
            base_cost = ship_data.get("shipping_option", {}) or {}
            envio = base_cost.get("cost", 0) or 0

    return {"comision": round(comision, 2), "envio": round(envio, 2)}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user_id = get_session_user(request)
    if user_id:
        return RedirectResponse("/dashboard")
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/login")
async def login():
    state = secrets.token_urlsafe(16)
    redirect_uri = f"{APP_URL}/callback"
    url = (
        f"{ML_AUTH_URL}"
        f"?response_type=code"
        f"&client_id={ML_CLIENT_ID}"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
    )
    response = RedirectResponse(url)
    response.set_cookie("oauth_state", state, max_age=600, httponly=True)
    return response


@app.get("/callback")
async def callback(request: Request, code: str, state: str):
    stored_state = request.cookies.get("oauth_state")
    if stored_state != state:
        raise HTTPException(400, "Estado OAuth inválido")

    async with httpx.AsyncClient() as client:
        resp = await client.post(ML_TOKEN_URL, data={
            "grant_type": "authorization_code",
            "client_id": ML_CLIENT_ID,
            "client_secret": ML_CLIENT_SECRET,
            "code": code,
            "redirect_uri": f"{APP_URL}/callback",
        })
        if resp.status_code != 200:
            raise HTTPException(400, f"Error al obtener token: {resp.text}")
        tokens = resp.json()

    user_id = str(tokens["user_id"])
    token_store[user_id] = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires_at": datetime.utcnow() + timedelta(seconds=tokens["expires_in"]),
        "nickname": tokens.get("nickname", user_id),
    }

    session_token = serializer.dumps(user_id)
    response = RedirectResponse("/dashboard")
    response.set_cookie("session", session_token, max_age=86400 * 7, httponly=True)
    response.delete_cookie("oauth_state")
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse("/")
    response.delete_cookie("session")
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user_id = get_session_user(request)
    if not user_id:
        return RedirectResponse("/")
    nickname = token_store[user_id]["nickname"]
    return templates.TemplateResponse("dashboard.html", {"request": request, "nickname": nickname})


@app.get("/api/summary")
async def api_summary(request: Request):
    user_id = get_session_user(request)
    if not user_id or not await refresh_token_if_needed(user_id):
        raise HTTPException(401, "No autorizado")

    token = token_store[user_id]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=30) as client:
        date_from = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00.000-00:00")
        orders_resp = await client.get(
            f"{ML_API_URL}/orders/search",
            headers=headers,
            params={
                "seller": user_id,
                "order.status": "paid",
                "order.date_created.from": date_from,
                "limit": 50,
                "sort": "date_desc",
            }
        )
        if orders_resp.status_code != 200:
            raise HTTPException(502, "Error al obtener órdenes")
        orders_data = orders_resp.json()
        results = orders_data.get("results", [])

        # Obtener fees reales en paralelo (máx 10 a la vez para no saturar la API)
        semaphore = asyncio.Semaphore(10)
        async def fetch_fees(order_id):
            async with semaphore:
                return await get_order_fees(client, order_id, headers)

        fees_list = await asyncio.gather(*[fetch_fees(str(o["id"])) for o in results])

    total_ventas = len(results)
    ingresos_brutos = sum(o.get("total_amount", 0) for o in results)
    total_comisiones = sum(f["comision"] for f in fees_list)
    total_envios = sum(f["envio"] for f in fees_list)
    ganancia_neta = ingresos_brutos - total_comisiones - total_envios

    # Daily breakdown últimos 7 días
    daily = {}
    for o, fees in zip(results, fees_list):
        date_str = o.get("date_created", "")[:10]
        if date_str:
            daily.setdefault(date_str, {"ventas": 0, "ingresos": 0, "ganancia": 0})
            amount = o.get("total_amount", 0)
            daily[date_str]["ventas"] += 1
            daily[date_str]["ingresos"] += amount
            daily[date_str]["ganancia"] += amount - fees["comision"] - fees["envio"]

    # Top productos
    products = {}
    for o in results:
        for item in o.get("order_items", []):
            title = item.get("item", {}).get("title", "Sin título")
            qty = item.get("quantity", 1)
            price = item.get("unit_price", 0)
            products.setdefault(title, {"cantidad": 0, "ingresos": 0})
            products[title]["cantidad"] += qty
            products[title]["ingresos"] += price * qty

    top_products = sorted(products.items(), key=lambda x: x[1]["ingresos"], reverse=True)[:5]

    return {
        "resumen": {
            "total_ventas": total_ventas,
            "ingresos_brutos": round(ingresos_brutos, 2),
            "total_comisiones": round(total_comisiones, 2),
            "total_envios": round(total_envios, 2),
            "ganancia_neta": round(ganancia_neta, 2),
            "margen_promedio": round((ganancia_neta / ingresos_brutos * 100) if ingresos_brutos else 0, 1),
        },
        "daily": dict(sorted(daily.items())[-7:]),
        "top_products": [{"nombre": k, **v} for k, v in top_products],
        "ultima_actualizacion": datetime.utcnow().isoformat(),
    }


@app.get("/api/orders")
async def api_orders(request: Request, limit: int = 20):
    user_id = get_session_user(request)
    if not user_id or not await refresh_token_if_needed(user_id):
        raise HTTPException(401, "No autorizado")

    token = token_store[user_id]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{ML_API_URL}/orders/search",
            headers=headers,
            params={"seller": user_id, "sort": "date_desc", "limit": limit},
        )
        if resp.status_code != 200:
            raise HTTPException(502, "Error al obtener órdenes")
        data = resp.json()
        results = data.get("results", [])

        semaphore = asyncio.Semaphore(10)
        async def fetch_fees(order_id):
            async with semaphore:
                return await get_order_fees(client, order_id, headers)

        fees_list = await asyncio.gather(*[fetch_fees(str(o["id"])) for o in results])

    orders = []
    for o, fees in zip(results, fees_list):
        amount = o.get("total_amount", 0)
        items = [i.get("item", {}).get("title", "?") for i in o.get("order_items", [])]
        orders.append({
            "id": o.get("id"),
            "fecha": o.get("date_created", "")[:10],
            "producto": ", ".join(items),
            "monto": round(amount, 2),
            "comision": fees["comision"],
            "envio": fees["envio"],
            "ganancia": round(amount - fees["comision"] - fees["envio"], 2),
            "estado": o.get("status", ""),
        })

    return {"orders": orders}
