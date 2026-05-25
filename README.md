# Panel de ventas MercadoLibre

Panel en tiempo real con OAuth multi-usuario. Construido con FastAPI + Railway.

---

## Paso 1 — Crear la app en MercadoLibre Developers

1. Entrá a https://developers.mercadolibre.com.ar
2. Hacé click en **Crear aplicación**
3. Completá:
   - **Nombre**: Panel de ventas (o el que quieras)
   - **Descripción corta**: Panel de métricas de ventas
   - **Dominio**: (lo completás después con la URL de Railway)
   - **Redirect URI**: https://TU-APP.railway.app/callback
   - **Permisos necesarios**: `read`, `orders`, `offline_access`
4. Guardá y copiá el **App ID** (CLIENT_ID) y el **Secret Key** (CLIENT_SECRET)

---

## Paso 2 — Subir a GitHub

```bash
git init
git add .
git commit -m "Panel ML inicial"
git remote add origin https://github.com/TU-USUARIO/ml-dashboard.git
git push -u origin main
```

---

## Paso 3 — Deployar en Railway

1. Entrá a https://railway.app y creá un nuevo proyecto
2. Elegí **Deploy from GitHub repo** y seleccioná tu repositorio
3. Railway va a detectar el `Procfile` automáticamente

---

## Paso 4 — Variables de entorno en Railway

En Railway → tu proyecto → **Variables**, agregá:

| Variable | Valor |
|---|---|
| `ML_CLIENT_ID` | Tu App ID de ML Developers |
| `ML_CLIENT_SECRET` | Tu Secret Key de ML Developers |
| `APP_URL` | https://TU-APP.railway.app (sin slash final) |
| `SECRET_KEY` | Una cadena random larga (ej: generala en https://randomkeygen.com) |

---

## Paso 5 — Actualizar el Redirect URI

Una vez que Railway te dio la URL final:
1. Volvé a ML Developers → tu app
2. Actualizá el **Redirect URI** con: `https://TU-APP.railway.app/callback`
3. Guardá

---

## Uso

- Entrá a tu URL de Railway
- Hacé click en **Conectar con MercadoLibre**
- Autorizás la app una sola vez
- El panel carga tus ventas de los últimos 30 días y se actualiza cada 5 minutos

---

## Estructura del proyecto

```
ml-dashboard/
├── main.py              # FastAPI: OAuth + endpoints API
├── templates/
│   ├── login.html       # Página de inicio de sesión
│   └── dashboard.html   # Panel principal
├── static/              # Assets estáticos (CSS/JS extra)
├── requirements.txt
└── Procfile             # Comando de arranque para Railway
```

---

## Notas importantes

- Los tokens se guardan **en memoria** (se pierden si Railway reinicia el servidor).
- Para producción con múltiples usuarios, se recomienda agregar una base de datos PostgreSQL (Railway la ofrece gratis) y guardar los tokens ahí.
- La app solo pide permisos de **lectura** — nunca puede hacer compras ni modificar tu cuenta.
