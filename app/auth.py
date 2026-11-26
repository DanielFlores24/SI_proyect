"""Login por contraseña compartida y límite de preguntas por sesión.

La sesión vive en una cookie firmada con HMAC: no hay base de datos ni sesión
en el servidor, así que funciona igual aunque Cloud Run cambie de instancia.
Dentro de la cookie guardamos el número de preguntas ya usadas.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs

from dotenv import load_dotenv
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

PASSWORD = os.environ.get("CHAT_PASSWORD", "")
CLAVE = os.environ.get("CHAT_SECRET", "")
HORAS = int(os.environ.get("SESSION_HOURS", "8"))
LIMITE = int(os.environ.get("MAX_PREGUNTAS", "100"))
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "0") == "1"
COOKIE_NOMBRE = "inro_sesion"

router = APIRouter()


def activo() -> bool:
    """Sin contraseña configurada el chat queda abierto (solo para desarrollo)."""
    return bool(PASSWORD and CLAVE)


def _codificar(datos: dict) -> str:
    crudo = json.dumps(datos, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(crudo).decode("ascii").rstrip("=")


def firmar(datos: dict) -> str:
    carga = _codificar(datos)
    firma = hmac.new(CLAVE.encode("utf-8"), carga.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{carga}.{firma}"


def leer(token: str | None) -> dict | None:
    if not token or "." not in token:
        return None
    carga, firma = token.rsplit(".", 1)
    esperada = hmac.new(
        CLAVE.encode("utf-8"), carga.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(firma, esperada):
        return None
    try:
        relleno = "=" * (-len(carga) % 4)
        datos = json.loads(base64.urlsafe_b64decode(carga + relleno))
    except (ValueError, TypeError):
        return None
    if float(datos.get("exp", 0)) <= time.time():
        return None
    if int(datos.get("n", 0)) < 0:
        return None
    return datos


def datos_sesion(request: Request) -> dict | None:
    return leer(request.cookies.get(COOKIE_NOMBRE))


def poner_cookie(respuesta: Response, token: str) -> None:
    respuesta.set_cookie(
        COOKIE_NOMBRE,
        token,
        max_age=HORAS * 3600,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
        path="/",
    )


def registrar_uso(request: Request) -> tuple[str | None, bool]:
    """Devuelve (cookie nueva, permitido). Cuenta antes de llamar al modelo."""
    datos = datos_sesion(request)
    if not datos:
        return None, False
    usadas = int(datos.get("n", 0))
    if usadas >= LIMITE:
        return None, False
    datos["n"] = usadas + 1
    return firmar(datos), True


def _pagina_login(error: str = "") -> str:
    aviso = f'<p class="error">{error}</p>' if error else ""
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Entrar · Asistente INRO</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <div class="tarjeta">
    <h1>Asistente INRO</h1>
    <p class="tenue">CUCEI · Universidad de Guadalajara</p>
    {aviso}
    <form method="post" action="/login">
      <input type="password" name="password" placeholder="Contraseña del chat"
             autocomplete="current-password" required autofocus>
      <button type="submit">Entrar</button>
    </form>
    <p class="tenue nota">La sesión dura {HORAS} h y permite {LIMITE} preguntas.</p>
  </div>
</body>
</html>"""


@router.get("/login", include_in_schema=False)
def formulario_login(request: Request) -> Response:
    if datos_sesion(request):
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(_pagina_login())


@router.post("/login", include_in_schema=False)
async def entrar(request: Request) -> Response:
    cuerpo = (await request.body()).decode("utf-8", errors="replace")
    password = parse_qs(cuerpo).get("password", [""])[0]
    if not hmac.compare_digest(password, PASSWORD):
        return HTMLResponse(
            _pagina_login("Contraseña incorrecta. Intenta de nuevo."), status_code=401
        )
    respuesta = RedirectResponse("/", status_code=303)
    poner_cookie(respuesta, firmar({"exp": time.time() + HORAS * 3600, "n": 0}))
    return respuesta


@router.get("/logout", include_in_schema=False)
def salir() -> Response:
    respuesta = RedirectResponse("/login", status_code=303)
    respuesta.delete_cookie(COOKIE_NOMBRE, path="/")
    return respuesta
