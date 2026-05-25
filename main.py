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
import asyncpg

ML_CLIENT_ID = os.environ["ML_CLIENT_ID"]
ML_CLIENT_SECRET = os.environ["ML_CLIENT_SECRET"]
APP_URL = os.environ.get("APP_URL", "http://localhost:8000")
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))
DATABASE_URL = os.environ["DATABASE_URL"]
ADMIN_EMAIL = os.environ["ADMIN_EMAIL"]
ADMIN_PASSWORD = os.environ["ADMIN_PASSWORD"]

ML_AUTH_URL = "https://auth.mercadolibre.com.ar/authorization"
ML_TOKEN_URL = "https://api.mercadolibre.com/oauth/token"
ML_API_URL = "https://api.mercadolibre.com"

serializer = URLSafeTimedSerializer(SECRET_KEY)
db_pool = None


@asynccontextmanager
async def lifespan(app):
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, ssl="require")
    await init_db()
    yield
    await db_pool.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


async def init_db():
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                is_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
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
        """)
        # Crear admin si no existe
        existing = await conn.fetchrow("SELECT id FROM users WHERE email=$1", ADMIN_EMAIL)
        if not existing:
            hashed = bcrypt.hashpw(ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode()
            await conn.execute(
                "INSERT INTO users (email, password_hash, name, is_admin) VALUES ($1,$2,$3,$4)",
                ADMIN_EMAIL, hashed, "Administrador", True
            )


def get_session_user_id(request: Request) -> Optional[int]:
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        return serializer.loads(token, max_age=86400 * 7)
    except BadSignature:
        return None


async def get_user(user_id: int):
    async with db_pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM users WHERE id=$1", user_id)


async def refresh_ml_token(account_id: int) -> Optional[str]:
    async with db_pool.acquire() as conn:
        acc = await conn.fetchrow("SELECT * FROM ml_accounts WHERE id=$1", account_id)
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
            await conn.execute(
                "UPDATE ml_accounts SET access_token=$1, refresh_token=$2, expires_at=$3 WHERE id=$4",
                tokens["access_token"],
                tokens.get("refresh_token", acc["refresh_token"]),
                new_expires,
                account_id
            )
            return tokens["access_token"]


async def get_order_fees(client, order_id, headers):
    fees_resp = await client.get(f"{ML_API_URL}/orders/{order_id}", headers=headers)
    if fees_resp.status_code != 200:
        return {"comision": 0, "envio": 0}
    data = fees_resp.json()
    comision = sum(abs(f.get("amount", 0)) for f in data.get("fees", []))
    envio = 0
    shipping_id = (data.get("shipping") or {}).get("id")
    if shipping_id:
        sr = await client.get(f"{ML_API_URL}/shipments/{shipping_id}", headers=headers)
        if sr.status_code == 200:
            envio = (sr.json().get("shipping_option") or {}).get("cost", 0) or 0
    return {"comision": round(comision, 2), "envio": round(envio, 2)}


# ── Auth ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if get_session_user_id(request):
        return RedirectResponse("/dashboard")
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/", response_class=HTMLResponse)
async def do_login(request: Request, email: str = Form(...), password: str = Form(...)):
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE email=$1", email.lower().strip())
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
    user = await get_user(user_id)
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch("SELECT id, nickname, ml_user_id FROM ml_accounts WHERE user_id=$1 ORDER BY id", user_id)
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "accounts": accounts,
    })


# ── ML OAuth ────────────────────────────────────────────────────

@app.get("/ml/connect")
async def ml_connect(request: Request):
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse("/")
    state = secrets.token_urlsafe(16)
    url = (
        f"{ML_AUTH_URL}?response_type=code"
        f"&client_id={ML_CLIENT_ID}"
        f"&redirect_uri={APP_URL}/ml/callback"
        f"&state={state}"
    )
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
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO ml_accounts (user_id, ml_user_id, nickname, access_token, refresh_token, expires_at)
            VALUES ($1,$2,$3,$4,$5,$6)
            ON CONFLICT (user_id, ml_user_id) DO UPDATE
            SET access_token=$4, refresh_token=$5, expires_at=$6, nickname=$3
        """, user_id, ml_user_id, nickname, tokens["access_token"], tokens["refresh_token"], expires_at)
    r = RedirectResponse("/dashboard", status_code=303)
    r.delete_cookie("oauth_state")
    return r


@app.post("/ml/disconnect/{account_id}")
async def ml_disconnect(request: Request, account_id: int):
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(401)
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM ml_accounts WHERE id=$1 AND user_id=$2", account_id, user_id)
    return RedirectResponse("/dashboard", status_code=303)


# ── API datos ───────────────────────────────────────────────────

@app.get("/api/summary/{account_id}")
async def api_summary(request: Request, account_id: int):
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(401)
    async with db_pool.acquire() as conn:
        acc = await conn.fetchrow("SELECT * FROM ml_accounts WHERE id=$1 AND user_id=$2", account_id, user_id)
    if not acc:
        raise HTTPException(404)
    token = await refresh_ml_token(account_id)
    if not token:
        raise HTTPException(502, "No se pudo renovar el token de ML")
    headers = {"Authorization": f"Bearer {token}"}
    ml_user_id = acc["ml_user_id"]
    async with httpx.AsyncClient(timeout=30) as client:
        date_from = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00.000-00:00")
        r = await client.get(f"{ML_API_URL}/orders/search", headers=headers, params={
            "seller": ml_user_id, "order.status": "paid",
            "order.date_created.from": date_from, "limit": 50, "sort": "date_desc",
        })
        if r.status_code != 200:
            raise HTTPException(502, "Error al obtener órdenes")
        results = r.json().get("results", [])
        sem = asyncio.Semaphore(10)
        async def fetch(oid):
            async with sem:
                return await get_order_fees(client, oid, headers)
        fees_list = await asyncio.gather(*[fetch(str(o["id"])) for o in results])

    ingresos = sum(o.get("total_amount", 0) for o in results)
    comisiones = sum(f["comision"] for f in fees_list)
    envios = sum(f["envio"] for f in fees_list)
    neta = ingresos - comisiones - envios
    daily = {}
    for o, f in zip(results, fees_list):
        d = o.get("date_created", "")[:10]
        if d:
            daily.setdefault(d, {"ventas": 0, "ingresos": 0, "ganancia": 0})
            a = o.get("total_amount", 0)
            daily[d]["ventas"] += 1
            daily[d]["ingresos"] += a
            daily[d]["ganancia"] += a - f["comision"] - f["envio"]
    products = {}
    for o in results:
        for item in o.get("order_items", []):
            t = item.get("item", {}).get("title", "Sin título")
            products.setdefault(t, {"cantidad": 0, "ingresos": 0})
            products[t]["cantidad"] += item.get("quantity", 1)
            products[t]["ingresos"] += item.get("unit_price", 0) * item.get("quantity", 1)
    top = sorted(products.items(), key=lambda x: x[1]["ingresos"], reverse=True)[:5]
    return {
        "resumen": {
            "total_ventas": len(results),
            "ingresos_brutos": round(ingresos, 2),
            "total_comisiones": round(comisiones, 2),
            "total_envios": round(envios, 2),
            "ganancia_neta": round(neta, 2),
            "margen_promedio": round((neta / ingresos * 100) if ingresos else 0, 1),
        },
        "daily": dict(sorted(daily.items())[-7:]),
        "top_products": [{"nombre": k, **v} for k, v in top],
        "ultima_actualizacion": datetime.utcnow().isoformat(),
    }


@app.get("/api/orders/{account_id}")
async def api_orders(request: Request, account_id: int, limit: int = 20):
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(401)
    async with db_pool.acquire() as conn:
        acc = await conn.fetchrow("SELECT * FROM ml_accounts WHERE id=$1 AND user_id=$2", account_id, user_id)
    if not acc:
        raise HTTPException(404)
    token = await refresh_ml_token(account_id)
    if not token:
        raise HTTPException(502)
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{ML_API_URL}/orders/search", headers=headers,
            params={"seller": acc["ml_user_id"], "sort": "date_desc", "limit": limit})
        if r.status_code != 200:
            raise HTTPException(502)
        results = r.json().get("results", [])
        sem = asyncio.Semaphore(10)
        async def fetch(oid):
            async with sem:
                return await get_order_fees(client, oid, headers)
        fees_list = await asyncio.gather(*[fetch(str(o["id"])) for o in results])
    orders = []
    for o, f in zip(results, fees_list):
        a = o.get("total_amount", 0)
        orders.append({
            "id": o.get("id"),
            "fecha": o.get("date_created", "")[:10],
            "producto": ", ".join(i.get("item", {}).get("title", "?") for i in o.get("order_items", [])),
            "monto": round(a, 2),
            "comision": f["comision"],
            "envio": f["envio"],
            "ganancia": round(a - f["comision"] - f["envio"], 2),
            "estado": o.get("status", ""),
        })
    return {"orders": orders}


# ── Admin ────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse("/")
    user = await get_user(user_id)
    if not user["is_admin"]:
        raise HTTPException(403, "Acceso denegado")
    async with db_pool.acquire() as conn:
        users = await conn.fetch("""
            SELECT u.*, COUNT(m.id) as ml_count
            FROM users u
            LEFT JOIN ml_accounts m ON m.user_id = u.id
            GROUP BY u.id ORDER BY u.created_at DESC
        """)
    return templates.TemplateResponse("admin.html", {"request": request, "users": users, "error": None, "success": None})


@app.post("/admin/users/create", response_class=HTMLResponse)
async def admin_create_user(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
):
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse("/")
    user = await get_user(user_id)
    if not user["is_admin"]:
        raise HTTPException(403)
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users (email, password_hash, name) VALUES ($1,$2,$3)",
                email.lower().strip(), hashed, name
            )
        return RedirectResponse("/admin?success=1", status_code=303)
    except Exception:
        async with db_pool.acquire() as conn:
            users = await conn.fetch("""
                SELECT u.*, COUNT(m.id) as ml_count
                FROM users u LEFT JOIN ml_accounts m ON m.user_id = u.id
                GROUP BY u.id ORDER BY u.created_at DESC
            """)
        return templates.TemplateResponse("admin.html", {"request": request, "users": users, "error": "El email ya existe", "success": None})


@app.post("/admin/users/delete/{target_id}")
async def admin_delete_user(request: Request, target_id: int):
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(401)
    user = await get_user(user_id)
    if not user["is_admin"]:
        raise HTTPException(403)
    if target_id == user_id:
        raise HTTPException(400, "No podés eliminarte a vos mismo")
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM users WHERE id=$1", target_id)
    return RedirectResponse("/admin", status_code=303)


@app.get("/admin/users/{target_id}/accounts")
async def admin_user_accounts(request: Request, target_id: int):
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(401)
    user = await get_user(user_id)
    if not user["is_admin"]:
        raise HTTPException(403)
    async with db_pool.acquire() as conn:
        accounts = await conn.fetch("SELECT * FROM ml_accounts WHERE user_id=$1", target_id)
        target_user = await conn.fetchrow("SELECT * FROM users WHERE id=$1", target_id)
    return JSONResponse([{
        "id": a["id"], "nickname": a["nickname"], "ml_user_id": a["ml_user_id"],
        "created_at": a["created_at"].isoformat()
    } for a in accounts])
