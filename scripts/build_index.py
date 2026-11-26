"""Convierte el corpus de data/dataIA/ en un índice de búsqueda semántica.

Salida:
    data/index/chunks.json      fragmentos + metadata (fuente, url, fecha, tipo)
    data/index/embeddings.npy   vectores de gemini-embedding-001 (768 dims)
    data/index/meta.json        modelo, dimensiones y fecha de construcción

Uso:
    python scripts/build_index.py --dry-run   # solo muestra estadísticas
    python scripts/build_index.py             # genera el índice
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataIA"
INDEX = ROOT / "data" / "index"

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
EMBED_LOCATION = os.environ.get("EMBEDDING_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
DIMENSIONES = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))

MAX_CHARS = 1200
OVERLAP = 150
BATCH = 24  # 24 fragmentos ≈ 8k tokens, por debajo del límite de 20k por petición
SIIAU_URL = "https://siiauescolar.siiau.udg.mx/wal/sspseca.consulta_oferta"


# --------------------------------------------------------------------------- #
# Lectura del corpus
# --------------------------------------------------------------------------- #
def leer_frontmatter(texto: str) -> tuple[dict, str]:
    """Los encabezados los escribimos como JSON, así que no hace falta PyYAML."""
    if not texto.startswith("---\n"):
        return {}, texto
    fin = texto.find("\n---\n", 4)
    if fin < 0:
        return {}, texto

    meta: dict = {}
    for linea in texto[4:fin].splitlines():
        if ":" not in linea:
            continue
        clave, valor = linea.split(":", 1)
        valor = valor.strip()
        try:
            meta[clave.strip()] = json.loads(valor)
        except json.JSONDecodeError:
            meta[clave.strip()] = valor.strip('"')

    return meta, texto[fin + 5 :]


def documentos() -> list[dict]:
    """Junta los Markdown del corpus. INDICE.md se excluye: es un resumen, no contenido."""
    archivos = [
        DATA / "mallas.md",
        DATA / "profesores_inro.md",
        *sorted((DATA / "fuentes").rglob("*.md")),
    ]

    salida = []
    for ruta in archivos:
        if not ruta.exists() or ruta.name == "INDICE.md":
            continue
        texto = ruta.read_text(encoding="utf-8")
        meta, cuerpo = leer_frontmatter(texto)
        if meta.get("titulo", "").startswith("Índice de fuentes"):
            continue

        # Los dos primeros archivos son derivados del sitio oficial / SIIAU.
        if ruta.name == "mallas.md":
            meta.setdefault("titulo", "Mallas curriculares 2018 y 2025")
            meta.setdefault("fuente", "Mallas oficiales INRO (PDF)")
            meta.setdefault("tipo", "oficial")
        elif ruta.name == "profesores_inro.md":
            meta.setdefault("titulo", "Profesores de Ingeniería Robótica")
            meta.setdefault("fuente", "SIIAU + PDF Acción U + MisProfesores")
            meta.setdefault("tipo", "mixto")
            # El listado por materia se indexa aparte desde el JSON (bloques limpios).
            corte = cuerpo.find("## Materias y profesores")
            if corte >= 0:
                cuerpo = cuerpo[:corte].rstrip()

        meta.setdefault("url", "")
        meta.setdefault("fecha_captura", "")
        meta.setdefault("tipo", "oficial")
        salida.append({"ruta": ruta, "meta": meta, "cuerpo": cuerpo})
    return salida


# --------------------------------------------------------------------------- #
# Fragmentación
# --------------------------------------------------------------------------- #
def dividir(texto: str) -> list[dict]:
    """Divide por encabezados y líneas, sin mezclar secciones en un mismo bloque."""
    bloques: list[dict] = []
    actual: list[str] = []
    largo = 0
    seccion_actual = ""

    def guardar() -> None:
        texto_bloque = "\n".join(actual).strip()
        if len(texto_bloque) > 30:
            bloques.append({"texto": texto_bloque, "seccion": seccion_actual})

    for linea in texto.splitlines():
        cabecera = re.match(r"^#{2,4}\s+(.+?)\s*$", linea)
        if cabecera:
            guardar()
            seccion_actual = cabecera.group(1)
            actual, largo = [linea], len(linea) + 1
            continue

        if actual and largo + len(linea) + 1 > MAX_CHARS:
            guardar()
            # Conservamos una cola para que el fragmento siguiente no empiece a ciegas.
            cola: list[str] = []
            usado = 0
            for l in reversed(actual):
                if usado + len(l) > OVERLAP:
                    break
                cola.insert(0, l)
                usado += len(l) + 1
            actual, largo = cola, usado

        actual.append(linea)
        largo += len(linea) + 1

    guardar()
    return bloques


def construir_chunks() -> list[dict]:
    chunks: list[dict] = []
    for doc in documentos():
        meta = doc["meta"]
        for parte in dividir(doc["cuerpo"]):
            contexto = meta.get("titulo", "")
            if parte["seccion"]:
                contexto += " › " + parte["seccion"]
            chunks.append(
                {
                    "id": f"c{len(chunks):04d}",
                    "texto": parte["texto"],
                    "texto_embedding": f"{contexto}\n{parte['texto']}",
                    "archivo": str(doc["ruta"].relative_to(ROOT)),
                    "titulo": meta.get("titulo", ""),
                    "fuente": meta.get("fuente", ""),
                    "url": meta.get("url", ""),
                    "fecha_captura": meta.get("fecha_captura", ""),
                    "tipo": meta.get("tipo", "oficial"),
                    "seccion": parte["seccion"],
                    "advertencia": meta.get("advertencia", ""),
                }
            )
    # Bloques estructurados: una materia = una respuesta completa y limpia.
    for bloque in chunks_profesores_por_materia():
        chunks.append({"id": f"c{len(chunks):04d}", **bloque})
    return chunks


def chunks_profesores_por_materia() -> list[dict]:
    """Crea un fragmento por materia desde la fuente estructurada de SIIAU."""
    ruta = DATA / "profesores_inro.json"
    if not ruta.exists():
        return []

    datos = json.loads(ruta.read_text(encoding="utf-8"))
    fecha = (datos.get("generado") or "")[:10]
    ciclos = ", ".join(datos.get("ciclos_consultados", []))
    salida: list[dict] = []

    for materia in datos.get("materias", []):
        profesores = sorted(materia.get("profesores", []))
        if not profesores:
            continue

        clave = materia["clave"]
        nombre = materia["nombre"]
        lista = "\n".join(f"- {nombre_prof}" for nombre_prof in profesores)
        texto = (
            f"Materia: {nombre} ({clave}).\n"
            f"Profesores registrados en SIIAU para los ciclos {ciclos}:\n{lista}"
        )
        embed = (
            f"Profesores por materia (SIIAU): {nombre} ({clave}). "
            + ", ".join(profesores)
            + f". Ciclos: {ciclos}."
        )

        salida.append(
            {
                "texto": texto,
                "texto_embedding": embed,
                "archivo": str(ruta.relative_to(ROOT)),
                "titulo": f"Profesores por materia — {nombre} ({clave})",
                "fuente": "SIIAU (oferta académica INRO, CUCEI)",
                "url": SIIAU_URL,
                "fecha_captura": fecha,
                "tipo": "oficial",
                "seccion": f"{nombre} ({clave})",
                "advertencia": "Listado observado en los ciclos consultados; la oferta puede cambiar por semestre.",
                "clave": clave,
                "nombre_materia": nombre,
                "estructurado": True,
            }
        )
    return salida


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def generar_vectores(chunks: list[dict]) -> np.ndarray:
    if not PROJECT:
        raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT en .env")

    client = genai.Client(vertexai=True, project=PROJECT, location=EMBED_LOCATION)
    config = types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=DIMENSIONES,
        auto_truncate=False,
    )

    vectores: list[list[float]] = []
    for i, lote in enumerate(batches(chunks, BATCH), 1):
        respuesta = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[c["texto_embedding"] for c in lote],
            config=config,
        )
        vectores.extend([e.values for e in respuesta.embeddings])
        print(f"  lote {i}: {len(vectores)}/{len(chunks)}", flush=True)
        time.sleep(0.2)

    matriz = np.asarray(vectores, dtype=np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    return matriz / np.maximum(normas, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="no llama a la API")
    args = parser.parse_args()

    chunks = construir_chunks()
    print(f"Documentos: {len({c['archivo'] for c in chunks})}")
    print(f"Fragmentos: {len(chunks)}")
    tipos: dict[str, int] = {}
    for c in chunks:
        tipos[c["tipo"]] = tipos.get(c["tipo"], 0) + 1
    print("Tipos:", tipos)

    if args.dry_run:
        print("\nEjemplo de fragmento:")
        print(json.dumps(chunks[0], ensure_ascii=False, indent=2)[:900])
        return

    INDEX.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerando embeddings con {EMBED_MODEL} en {EMBED_LOCATION}...")
    matriz = generar_vectores(chunks)

    # Guardamos solo los campos necesarios para responder y citar.
    publicos = [
        {k: v for k, v in c.items() if k != "texto_embedding"}
        for c in chunks
    ]
    (INDEX / "chunks.json").write_text(
        json.dumps(publicos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(INDEX / "embeddings.npy", matriz)
    (INDEX / "meta.json").write_text(
        json.dumps(
            {
                "generado": datetime.now().isoformat(timespec="seconds"),
                "modelo": EMBED_MODEL,
                "dimensiones": DIMENSIONES,
                "fragmentos": len(chunks),
                "estructurados": sum(1 for c in chunks if c.get("estructurado")),
                "max_chars": MAX_CHARS,
                "overlap": OVERLAP,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nOK {INDEX / 'chunks.json'}")
    print(f"OK {INDEX / 'embeddings.npy'}  {matriz.shape}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
