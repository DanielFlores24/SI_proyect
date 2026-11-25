"""Consolida los profesores de Ingeniería en Robótica (CUCEI, UdeG).

Fuentes:
  1. SIIAU - Consulta pública de oferta académica (sin login), filtrada por
     carrera `INRO`, centro `D` (CUCEI) y varios ciclos escolares.
  2. PDF "Recomendaciones de Maestros - Acción U 2026A: INRO" (reseñas por
     materia y profesor).
  3. MisProfesores.com (calificaciones promedio de las páginas de CUCEI).

Salidas:
  data/dataIA/profesores_inro.json
  data/dataIA/profesores_inro.md

Uso:
  python scripts/scrape_profesores.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import defaultdict
from urllib.parse import quote
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DATA_IA = ROOT / "data" / "dataIA"
DOCS = ROOT / "data" / "docs"

SIIAU_URL = "https://siiauescolar.siiau.udg.mx/wal/sspseca.consulta_oferta"
CICLOS = ["202620", "202610", "202520", "202510", "202420", "202410"]

MISPROFES_URLS = [
    "https://www.misprofesores.com/escuelas/CUCEI_4047",
    "https://www.misprofesores.com/escuelas/CUCEI_11479",
    "https://www.misprofesores.com/escuelas/CUCEI_2727",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; proyecto-escolar/1.0)"}

KEY_RE = re.compile(r"^[A-Z]?\d{4,5}$")
SECCION_RE = re.compile(r"^[A-Z]\d{2}$")
NAME_RE = re.compile(
    r"^([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{4,70},\s*[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{2,70}):\s*$"
)
SUBJECT_RE = re.compile(r"^(.+?)\s*-\s*([A-Z]?\d{4,5})\s*$")


def norm(text: str) -> str:
    """Normaliza un nombre para poder comparar entre fuentes."""
    text = unicodedata.normalize("NFD", str(text or ""))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.upper().replace("Ñ", "N")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def name_key(text: str) -> str:
    """Tokens ordenados: 'APELLIDO, NOMBRE' y 'NOMBRE APELLIDO' dan lo mismo."""
    return " ".join(sorted(norm(text).split()))


def fetch(url: str, params: dict | None = None, encoding: str = "utf-8") -> str:
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    # SIIAU declara ISO-8859-1 y en realidad sirve Windows-1252 (Ñ = 0xD1).
    return r.content.decode(encoding, errors="replace")


# --------------------------------------------------------------------------- #
# SIIAU
# --------------------------------------------------------------------------- #
def scrape_siiau() -> tuple[dict, dict, list[str]]:
    """Devuelve (profesores, materias, ciclos_consultados)."""
    profesores: dict[str, dict] = {}
    materias: dict[str, dict] = {}
    ciclos_ok: list[str] = []

    for ciclo in CICLOS:
        try:
            html = fetch(
                SIIAU_URL,
                {"ciclop": ciclo, "cup": "D", "majrp": "INRO"},
                encoding="cp1252",
            )
        except Exception as exc:  # pragma: no cover - robustez ante caídas del sitio
            print(f"[SIIAU] {ciclo}: error {exc}")
            continue

        soup = BeautifulSoup(html, "lxml")
        tabla = soup.find("table")
        if not tabla:
            print(f"[SIIAU] {ciclo}: sin tabla")
            continue

        encontrados = 0
        for tr in tabla.find_all("tr"):
            celdas = tr.find_all("td", recursive=False)
            if len(celdas) < 9:
                continue
            vals = [c.get_text(" ", strip=True) for c in celdas[:7]]
            nrc, clave, materia, seccion = vals[:4]
            if not (nrc.isdigit() and KEY_RE.match(clave or "") and SECCION_RE.match(seccion or "")):
                continue

            profesores_fila: list[str] = []
            for td in celdas[8].find_all("td", class_="tdprofesor"):
                texto = td.get_text(" ", strip=True)
                if "," in texto and re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", texto):
                    if texto not in profesores_fila:
                        profesores_fila.append(texto)

            sesiones = []
            for fila in celdas[7].find_all("tr"):
                partes = [td.get_text(" ", strip=True) for td in fila.find_all("td")]
                if any(partes):
                    sesiones.append(" | ".join(p for p in partes if p))

            encontrados += len(profesores_fila)
            materias.setdefault(clave, {"clave": clave, "nombre": materia, "profesores": set()})

            for nombre in profesores_fila:
                key = name_key(nombre)
                prof = profesores.setdefault(
                    key,
                    {
                        "nombre": nombre,
                        "nombre_normalizado": norm(nombre),
                        "materias": {},
                        "ciclos": set(),
                        "secciones": 0,
                        "fuentes": ["SIIAU"],
                    },
                )
                # Preferir la variante con tildes si aparece en otro ciclo.
                if any(ord(c) > 127 for c in nombre) and not any(ord(c) > 127 for c in prof["nombre"]):
                    prof["nombre"] = nombre
                    prof["nombre_normalizado"] = norm(nombre)

                info = prof["materias"].setdefault(
                    clave, {"clave": clave, "materia": materia, "ciclos": set(), "secciones": 0}
                )
                info["ciclos"].add(ciclo)
                info["secciones"] += 1
                prof["ciclos"].add(ciclo)
                prof["secciones"] += 1
                materias[clave]["profesores"].add(nombre)

        if encontrados:
            ciclos_ok.append(ciclo)
        print(f"[SIIAU] {ciclo}: {encontrados} asignaciones profesor-sección")

    return profesores, materias, ciclos_ok


# --------------------------------------------------------------------------- #
# PDF de recomendaciones
# --------------------------------------------------------------------------- #
def extraer_texto_pdf(path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        # Algunos PDFs del proyecto tienen acentos descompuestos en el nombre;
        # pdftotext falla con esos bytes en la línea de comandos de Windows.
        tmpdir = Path(tempfile.gettempdir()) / "opencode"
        tmpdir.mkdir(parents=True, exist_ok=True)
        copia = tmpdir / "recomendaciones_temp.pdf"
        shutil.copy2(path, copia)
        salida = tmpdir / "recomendaciones_temp.txt"
        subprocess.run(
            [pdftotext, "-layout", str(copia), str(salida)],
            check=True,
            capture_output=True,
        )
        return salida.read_text(encoding="utf-8", errors="replace")

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Instala pdftotext o pypdf para leer el PDF de recomendaciones") from exc
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def scrape_recomendaciones() -> tuple[list[dict], dict[str, list[dict]]]:
    archivos = list(DOCS.glob("*Recomendaciones*.pdf"))
    if not archivos:
        print("[PDF] No se encontró el PDF de recomendaciones")
        return [], {}

    texto = extraer_texto_pdf(archivos[0])
    materias: list[dict] = []
    actual: dict | None = None
    profesor_actual: dict | None = None
    buffer: list[str] = []

    def cerrar() -> None:
        nonlocal profesor_actual, buffer
        if profesor_actual is not None:
            profesor_actual["resena"] = " ".join(buffer).strip()
        profesor_actual = None
        buffer = []

    for linea in texto.splitlines():
        limpia = linea.strip()
        m_sub = SUBJECT_RE.match(limpia)
        m_nom = NAME_RE.match(limpia)
        if m_sub and (" - " in limpia or re.search(r"\S-\s*[A-Z]?\d{4,5}$", limpia)):
            cerrar()
            nombre = m_sub.group(1).strip()
            anterior = materias[-1]["materia"] if materias else ""
            # El PDF parte algunos nombres de materia en dos renglones.
            if anterior.count("(") > anterior.count(")"):
                nombre = f"{anterior} {nombre})"
            elif nombre.startswith("(") and anterior:
                base = re.sub(r"\s*\([^)]*\)\s*$", "", anterior).strip()
                nombre = f"{base} {nombre}".strip()
            actual = {
                "materia": nombre,
                "clave": m_sub.group(2),
                "profesores": [],
            }
            materias.append(actual)
        elif m_nom and actual is not None:
            cerrar()
            profesor_actual = {"nombre": m_nom.group(1).strip(), "resena": ""}
            actual["profesores"].append(profesor_actual)
        elif profesor_actual is not None and limpia:
            if limpia.startswith("*") or "(c) 2002" in limpia:
                cerrar()
            else:
                buffer.append(limpia)
    cerrar()

    por_profesor: dict[str, list[dict]] = defaultdict(list)
    for materia in materias:
        for prof in materia["profesores"]:
            por_profesor[name_key(prof["nombre"])].append(
                {
                    "materia": materia["materia"],
                    "clave": materia["clave"],
                    "resena": prof["resena"],
                }
            )

    print(f"[PDF] {len(materias)} materias, {sum(len(m['profesores']) for m in materias)} reseñas")
    return materias, por_profesor


# --------------------------------------------------------------------------- #
# MisProfesores
# --------------------------------------------------------------------------- #
def scrape_misprofes() -> list[dict]:
    registros: list[dict] = []
    for url in MISPROFES_URLS:
        try:
            html = fetch(url)
        except Exception as exc:  # pragma: no cover
            print(f"[MisProfes] {url}: error {exc}")
            continue
        m = re.search(r"var\s+dataSet\s*=\s*(\[.*?\]);", html, re.S)
        if not m:
            print(f"[MisProfes] {url}: sin dataSet")
            continue
        datos = json.loads(m.group(1))
        for item in datos:
            nombre = f"{item.get('n', '')} {item.get('a', '')}".strip()
            if not nombre or not item.get("i"):
                continue
            promedio = item.get("c")
            slug = quote(nombre.replace("\\", "").replace(" ", "-"), safe="-_")
            registros.append(
                {
                    "nombre": nombre,
                    "nombre_normalizado": norm(nombre),
                    "departamento": (item.get("d") or "").strip(),
                    "calificaciones": int(item.get("m") or 0),
                    "promedio": round(float(promedio), 1) if promedio else None,
                    "url": f"https://www.misprofesores.com/profesores/{slug}_{item['i']}",
                }
            )
        print(f"[MisProfes] {url}: {len(datos)} registros")

    # Deduplicar por id de URL.
    vistos: set[str] = set()
    unicos = []
    for r in registros:
        if r["url"] in vistos:
            continue
        vistos.add(r["url"])
        unicos.append(r)
    return unicos


def match_misprofes(nombre: str, registros: list[dict]) -> dict | None:
    tokens = set(norm(nombre).split())
    if len(tokens) < 3:
        return None
    mejor = None
    mejor_score = 0
    for r in registros:
        rt = set(r["nombre_normalizado"].split())
        score = len(tokens & rt)
        if score > mejor_score:
            mejor, mejor_score = r, score
    # Exigir cuando menos apellido + nombre (2 tokens) y la mitad de los tokens.
    if mejor_score >= 2 and mejor_score >= len(tokens) * 0.6:
        return mejor
    return None


# --------------------------------------------------------------------------- #
# Consolidación y salidas
# --------------------------------------------------------------------------- #
def serializar(profesores: dict, materias: dict, recomendaciones: dict, misprofes: list[dict]) -> dict:
    resultado = []
    for key, prof in sorted(profesores.items(), key=lambda kv: kv[1]["nombre"]):
        recs = recomendaciones.get(key, [])
        mp = match_misprofes(prof["nombre"], misprofes)
        item = {
            "nombre": prof["nombre"],
            "nombre_normalizado": prof["nombre_normalizado"],
            "materias": [
                {
                    "clave": info["clave"],
                    "materia": info["materia"],
                    "ciclos": sorted(info["ciclos"]),
                    "secciones": info["secciones"],
                }
                for info in sorted(prof["materias"].values(), key=lambda x: x["materia"])
            ],
            "ciclos": sorted(prof["ciclos"]),
            "secciones_totales": prof["secciones"],
            "recomendaciones": recs,
            "misprofes": mp,
            "fuentes": ["SIIAU"] + (["PDF Acción U 2026A"] if recs else []) + (["MisProfesores"] if mp else []),
        }
        resultado.append(item)

    lista_materias = []
    for clave, info in sorted(materias.items(), key=lambda kv: kv[1]["nombre"]):
        lista_materias.append(
            {
                "clave": clave,
                "nombre": info["nombre"],
                "profesores": sorted(info["profesores"]),
            }
        )

    return {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "carrera": "Ingeniería en Robótica (CUCEI, UdeG)",
        "ciclos_consultados": CICLOS,
        "total_profesores": len(resultado),
        "total_materias": len(lista_materias),
        "profesores": resultado,
        "materias": lista_materias,
    }


def escribir_markdown(datos: dict, ruta: Path) -> None:
    lineas = [
        "# Profesores de Ingeniería en Robótica (CUCEI, UdeG)",
        "",
        f"Generado: {datos['generado']}  ",
        f"Ciclos consultados: {', '.join(datos['ciclos_consultados'])}  ",
        f"**{datos['total_profesores']} profesores** en {datos['total_materias']} materias.",
        "",
        "Fuentes: oferta académica pública de SIIAU (carrera `INRO`, centro `D`), "
        "recomendaciones de Acción U 2026A y MisProfesores.com.",
        "",
        "## Profesores",
        "",
        "| Profesor | Materias (clave) | Ciclos | Recomendación | MisProfes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p in datos["profesores"]:
        mats = ", ".join(f"{m['materia']} ({m['clave']})" for m in p["materias"])
        ciclos = ", ".join(p["ciclos"])
        rec = f"Sí ({len(p['recomendaciones'])})" if p["recomendaciones"] else "—"
        mp = p["misprofes"]
        mp_txt = f"{mp['promedio']}/10 ({mp['calificaciones']} calif.)" if mp and mp["promedio"] else "—"
        lineas.append(f"| {p['nombre']} | {mats} | {ciclos} | {rec} | {mp_txt} |")

    lineas += ["", "## Materias y profesores", ""]
    for m in datos["materias"]:
        lineas.append(f"### {m['nombre']} (`{m['clave']}`)")
        lineas.append("")
        if m["profesores"]:
            for nombre in m["profesores"]:
                lineas.append(f"- {nombre}")
        else:
            lineas.append("- _Sin profesores registrados en los ciclos consultados_")
        lineas.append("")

    ruta.write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    DATA_IA.mkdir(parents=True, exist_ok=True)

    profesores, materias, ciclos = scrape_siiau()
    recomendaciones_materias, recomendaciones = scrape_recomendaciones()
    misprofes = scrape_misprofes()

    datos = serializar(profesores, materias, recomendaciones, misprofes)
    datos["ciclos_consultados"] = ciclos
    datos["recomendaciones_pdf"] = recomendaciones_materias

    json_path = DATA_IA / "profesores_inro.json"
    md_path = DATA_IA / "profesores_inro.md"
    json_path.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    escribir_markdown(datos, md_path)

    print(f"\nOK {json_path}")
    print(f"OK {md_path}")
    print(f"Total: {datos['total_profesores']} profesores, {datos['total_materias']} materias")


if __name__ == "__main__":
    main()
