"""Genera .env.yaml para Cloud Run a partir de .env.

Los secretos (CHAT_PASSWORD, CHAT_SECRET) se excluyen: en Cloud Run se pasan
con --set-secrets desde Secret Manager. COOKIE_SECURE se fuerza a 1 porque
Cloud Run sirve por HTTPS. El archivo resultante está ignorado por Git.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGEN = ROOT / ".env"
SALIDA = ROOT / ".env.yaml"
IGNORAR = {"CHAT_PASSWORD", "CHAT_SECRET", "PORT"}


def escapar(valor: str) -> str:
    return valor.replace("\\", "\\\\").replace('"', '\\"')


def main() -> None:
    if not ORIGEN.exists():
        raise SystemExit("Falta .env. Crea el archivo con las variables documentadas en README.md.")

    lineas = ["# Generado por scripts/make_env_yaml.py. No editar a mano."]
    for linea in ORIGEN.read_text(encoding="utf-8").splitlines():
        texto = linea.strip()
        if not texto or texto.startswith("#") or "=" not in texto:
            continue
        clave, valor = texto.split("=", 1)
        clave = clave.strip()
        if clave in IGNORAR:
            continue
        valor = valor.strip()
        if clave == "COOKIE_SECURE":
            valor = "1"
        lineas.append(f'{clave}: "{escapar(valor)}"')

    SALIDA.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Escribí {SALIDA.relative_to(ROOT)} con {len(lineas) - 1} variables.")


if __name__ == "__main__":
    main()
