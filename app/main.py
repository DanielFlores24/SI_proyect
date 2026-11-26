"""API del chat de Ingeniería Robótica.

Endpoints:
    GET  /health   estado del servicio
    POST /api/ask  {"pregunta": "...", "historial": [...]} -> {"respuesta": "...", "fuentes": [...]}
    GET  /          interfaz de chat (protegida por contraseña)
    GET  /login     formulario de acceso
    GET  /logout    cierra la sesión
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import auth, rag

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "static"
INDEX_HTML = ROOT / "app" / "templates" / "index.html"

app = FastAPI(title="Chat INRO", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.include_router(auth.router)


class Turno(BaseModel):
    pregunta: str
    respuesta: str


class Pregunta(BaseModel):
    pregunta: str = Field(min_length=1, max_length=1000)
    historial: list[Turno] = Field(default_factory=list)


@app.middleware("http")
async def proteger(request: Request, llamar):
    """El login y /health son públicos; el chat y la API necesitan sesión."""
    if auth.activo():
        ruta = request.url.path
        publicas = ruta in ("/login", "/health") or ruta.startswith("/static/")
        if not publicas and not auth.datos_sesion(request):
            if ruta.startswith("/api/"):
                return JSONResponse(
                    {"detail": "Inicia sesión en /login para usar el chat."},
                    status_code=401,
                )
            return RedirectResponse("/login", status_code=303)
    return await llamar(request)


@app.get("/health")
def health() -> dict:
    chunks, vectors = rag.cargar_indice()
    return {
        "estado": "ok",
        "fragmentos": len(chunks),
        "dimensiones": int(vectors.shape[1]),
        "modelo": rag.CHAT_MODEL,
        "protegido": auth.activo(),
        "limite_preguntas": auth.LIMITE,
    }


@app.post("/api/ask")
def ask(body: Pregunta, request: Request) -> JSONResponse:
    cookie: str | None = None
    if auth.activo():
        cookie, permitido = auth.registrar_uso(request)
        if not permitido:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Ya usaste {auth.LIMITE} preguntas en esta sesión. "
                    "Cierra sesión y vuelve a entrar para seguir."
                ),
            )
    try:
        historial = [turno.model_dump() for turno in body.historial]
        resultado = rag.responder(body.pregunta.strip(), historial)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error al consultar el modelo: {exc}")
    respuesta = JSONResponse(resultado)
    if cookie:
        auth.poner_cookie(respuesta, cookie)
    return respuesta


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)
