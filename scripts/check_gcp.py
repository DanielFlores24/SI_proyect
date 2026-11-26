"""Comprobación mínima de que Vertex AI funciona con nuestras credenciales.

Uso:
    python scripts/check_gcp.py

Si falla con "could not find default credentials", ejecuta:
    gcloud auth application-default login
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from google import genai
from google.genai import types

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
DEFAULT_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
EMBED_LOCATION = os.environ.get("EMBEDDING_LOCATION", DEFAULT_LOCATION)
CHAT_LOCATION = os.environ.get("GENERATION_LOCATION", DEFAULT_LOCATION)
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


def main() -> int:
    if not PROJECT:
        print("Falta GOOGLE_CLOUD_PROJECT. Revisa tu .env con las variables del README.md.")
        return 1

    embed_client = genai.Client(vertexai=True, project=PROJECT, location=EMBED_LOCATION)
    chat_client = genai.Client(vertexai=True, project=PROJECT, location=CHAT_LOCATION)

    print(f"Proyecto: {PROJECT}")
    print(f"Ubicaciones: embeddings={EMBED_LOCATION} | generación={CHAT_LOCATION}")
    print(f"Modelos: {EMBED_MODEL} / {CHAT_MODEL}")

    try:
        embed = embed_client.models.embed_content(
            model=EMBED_MODEL,
            contents=["¿Cuántos créditos tiene Control I?"],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
        dimensiones = len(embed.embeddings[0].values)
        print(f"OK embedding: {dimensiones} dimensiones")
    except Exception as exc:
        print(f"FALLÓ embedding: {type(exc).__name__}: {exc}")
        return 1

    try:
        respuesta = chat_client.models.generate_content(
            model=CHAT_MODEL,
            contents="Responde exactamente: OK",
        )
        print(f"OK generación: {respuesta.text.strip()}")
    except Exception as exc:
        print(f"FALLÓ generación: {type(exc).__name__}: {exc}")
        return 1

    print("\nTodo listo. Ya podemos construir el RAG.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
