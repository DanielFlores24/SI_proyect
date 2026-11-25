"""Fuentes adicionales para el chat de Ingeniería en Robótica (CUCEI, UdeG).

Genera documentos con trazabilidad (fuente, URL, fecha de captura) en:
    data/dataIA/fuentes/

Fuentes:
  1. Sitio oficial de la carrera (cucei.udg.mx/carreras/robotica)
  2. Catálogo de materias de SIIAU (créditos, horas, prerrequisitos)
  3. Perfiles de académicos del CUCEI (investigadores SNII / PRODEP)
  4. Reseñas completas de MisProfesores para los profesores ya emparejados
  5. Bibliografía sugerida por materia vía Open Library (NO oficial)

Uso:
  python scripts/scrape_fuentes_extra.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_profesores import CICLOS, SIIAU_URL, name_key, norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_IA = ROOT / "data" / "dataIA"
FUENTES = DATA_IA / "fuentes"
BASE_INRO = "https://www.cucei.udg.mx/carreras/robotica/"
ACADEMICOS = "https://academicos.cucei.udg.mx"
OPENLIB = "https://openlibrary.org/search.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; proyecto-escolar/1.0)"}
HOY = datetime.now().strftime("%Y-%m-%d")

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(total=3, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504])
    ),
)


def get(url: str, *, encoding: str = "utf-8", params: dict | None = None, pause: float = 0.15) -> str | None:
    try:
        r = SESSION.get(url, params=params, timeout=35)
        r.raise_for_status()
        time.sleep(pause)
        return r.content.decode(encoding, errors="replace")
    except Exception as exc:
        print(f"  ! {url[:90]} -> {type(exc).__name__}")
        return None


def texto_limpio(texto: str) -> str:
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n\s*\n+", "\n\n", texto)
    return texto.strip()


def titulo_legible(texto: str) -> str:
    """Capitaliza sin arruinar números romanos ni siglas cortas."""
    especiales = {"I", "II", "III", "IV", "V", "XXI", "PDS", "IA", "HMI"}
    def Cap(m: re.Match) -> str:
        w = m.group(0)
        return w if w.upper() in especiales else w[0].upper() + w[1:].lower()
    return re.sub(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]+", Cap, texto)


def frontmatter(titulo: str, fuente: str, url: str, extra: dict | None = None, tipo: str = "apoyo") -> str:
    datos = {
        "titulo": titulo,
        "fuente": fuente,
        "url": url,
        "fecha_captura": HOY,
        "tipo": tipo,
    }
    datos.update(extra or {})
    lineas = ["---"]
    for k, v in datos.items():
        lineas.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    lineas += ["---", ""]
    return "\n".join(lineas)


# --------------------------------------------------------------------------- #
# 1. Sitio oficial de la carrera
# --------------------------------------------------------------------------- #
def scrape_sitio_oficial() -> list[dict]:
    print("\n[1/5] Sitio oficial INRO")
    directorio = FUENTES / "sitio_oficial"
    directorio.mkdir(parents=True, exist_ok=True)

    visitados: set[str] = set()
    pendientes = [BASE_INRO]
    documentos: list[dict] = []

    while pendientes and len(visitados) < 60:
        url = pendientes.pop(0).split("#")[0]
        if url in visitados:
            continue
        visitados.add(url)

        html = get(url)
        if not html:
            continue
        sopa = BeautifulSoup(html, "lxml")

        for a in sopa.find_all("a", href=True):
            href = urljoin(url, a["href"]).split("#")[0]
            if href.startswith(BASE_INRO) and href not in visitados and href not in pendientes:
                # Evitar archivos adjuntos y anclas de photography.
                if not re.search(r"\.(pdf|docx?|xlsx?|jpg|png)$", href, re.I):
                    pendientes.append(href)

        contenido = sopa.select_one(".content") or sopa.select_one("main") or sopa.body
        if contenido is None:
            continue
        for tag in contenido.select("script, style, nav"):
            tag.decompose()

        titulo = (sopa.find("h1").get_text(" ", strip=True) if sopa.find("h1") else sopa.title.get_text(" ", strip=True) if sopa.title else url)
        cuerpo = texto_limpio(contenido.get_text("\n", strip=True))
        if len(cuerpo) < 60:
            continue

        adjuntos = []
        for a in contenido.find_all("a", href=True):
            if re.search(r"\.(pdf|docx?|xlsx?)$", a["href"], re.I):
                adjuntos.append({"texto": a.get_text(" ", strip=True), "url": urljoin(url, a["href"])})

        slug = re.sub(r"[^a-z0-9]+", "-", url.replace(BASE_INRO, "").lower()).strip("-") or "inicio"
        archivo = directorio / f"{slug}.md"
        archivo.write_text(
            frontmatter(titulo, "Sitio oficial de Ingeniería Robótica (CUCEI)", url, {"adjuntos": adjuntos}, tipo="oficial")
            + f"# {titulo}\n\n{cuerpo}\n",
            encoding="utf-8",
        )
        documentos.append({"titulo": titulo, "url": url, "archivo": str(archivo.relative_to(ROOT)), "caracteres": len(cuerpo)})

    print(f"  {len(documentos)} páginas guardadas")
    return documentos


# --------------------------------------------------------------------------- #
# 2. Catálogo SIIAU por materia
# --------------------------------------------------------------------------- #
def scrape_catalogo() -> tuple[list[dict], list[dict]]:
    print("\n[2/5] Catálogo SIIAU")
    profesores = json.loads((DATA_IA / "profesores_inro.json").read_text(encoding="utf-8"))
    claves = {m["clave"]: m["nombre"] for m in profesores["materias"]}

    subclaves: dict[str, str] = {}
    for ciclo in CICLOS:
        html = get(SIIAU_URL, encoding="cp1252", params={"ciclop": ciclo, "cup": "D", "majrp": "INRO"})
        if not html:
            continue
        for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
            m = re.search(r"subclavep=([^&]+)", a["href"])
            clave = a.get_text(strip=True)
            if m and clave in claves:
                subclaves.setdefault(clave, m.group(1))

    registros: list[dict] = []
    for clave, nombre in claves.items():
        if clave not in subclaves:
            print(f"  ! sin subcódigo para {clave}")
            continue
        url = "https://siiauescolar.siiau.udg.mx/wal/scpcata.detmate"
        html = get(url, encoding="cp1252", params={"subclavep": subclaves[clave], "pEntra": "OAP"})
        if not html:
            continue
        sopa = BeautifulSoup(html, "lxml")
        campos: dict[str, str] = {}
        for tr in sopa.find_all("tr"):
            celdas = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            for i in range(0, len(celdas) - 1, 2):
                if celdas[i].endswith(":"):
                    campos[celdas[i][:-1].strip()] = celdas[i + 1].strip()

        if not campos.get("Clave"):
            continue
        registros.append(
            {
                "clave": clave,
                "nombre": titulo_legible(nombre),
                "area": campos.get("Area", ""),
                "creditos": campos.get("Créditos", ""),
                "horas_teoria": campos.get("Teoría", ""),
                "horas_practica": campos.get("Práctica", ""),
                "tipo": campos.get("Tipo", ""),
                "nivel": campos.get("Nivel", ""),
                "ciclo_inicio": campos.get("Ciclo de Inicio", ""),
                "estatus": campos.get("Estatus", ""),
                "prerrequisitos": campos.get("Prerrequisitos", ""),
                "correquisitos": campos.get("Correquisitos", ""),
                "departamento": campos.get("Departamento(s)", ""),
                "carreras": campos.get("Carrera(s)", ""),
                "url": url + "?subclavep=" + subclaves[clave] + "&pEntra=OAP",
            }
        )

    lineas = [frontmatter("Catálogo de materias INRO", "SIIAU - Catálogo público", "https://siiauescolar.siiau.udg.mx/wal/scpcata.menu", tipo="oficial")]
    lineas.append("# Catálogo de materias de Ingeniería Robótica\n")
    lineas.append(f"{len(registros)} unidades de aprendizaje. Fuente: SIIAU, capturado {HOY}.\n")
    lineas.append("| Clave | Materia | Créditos | Teoría | Práctica | Prerrequisitos | Departamento |")
    lineas.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(registros, key=lambda x: x["nombre"]):
        linea = "| " + " | ".join(
            str(r[k]).replace("|", "/") for k in ["clave", "nombre", "creditos", "horas_teoria", "horas_practica", "prerrequisitos", "departamento"]
        ) + " |"
        lineas.append(linea)
    lineas.append("")
    (FUENTES / "catalogo_materias.md").write_text("\n".join(lineas), encoding="utf-8")

    print(f"  {len(registros)} materias")
    return registros, [{"titulo": "Catálogo de materias INRO", "url": "https://siiauescolar.siiau.udg.mx/wal/scpcata.menu", "archivo": "data/dataIA/fuentes/catalogo_materias.md"}]


# --------------------------------------------------------------------------- #
# 3. Perfiles de académicos
# --------------------------------------------------------------------------- #
def descubrir_perfiles() -> list[dict]:
    vistos: dict[str, dict] = {}
    for seccion, max_paginas in [("investigadores-sni", 25), ("perfiles-prodep", 15)]:
        for pagina in range(1, max_paginas + 1):
            url = f"{ACADEMICOS}/{seccion}"
            html = get(url, params={"page": pagina}, pause=0.1)
            if not html:
                continue
            sopa = BeautifulSoup(html, "lxml")
            filas = 0
            for tr in sopa.find_all("tr"):
                a = tr.find("a", href=re.compile(r"/academicos/\d+"))
                if not a:
                    continue
                celdas = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
                vistos.setdefault(
                    a["href"],
                    {
                        "url": a["href"],
                        "nombre": a.get_text(" ", strip=True),
                        "datos": " | ".join(celdas[1:])[:600],
                        "lista": seccion,
                    },
                )
                filas += 1
            if filas == 0:
                break
    return list(vistos.values())


def scrape_academicos() -> tuple[list[dict], list[dict]]:
    print("\n[3/5] Perfiles de académicos")
    profesores = json.loads((DATA_IA / "profesores_inro.json").read_text(encoding="utf-8"))["profesores"]
    objetivos = {name_key(p["nombre"]): p["nombre"] for p in profesores}

    candidatos = descubrir_perfiles()
    print(f"  {len(candidatos)} perfiles descubiertos en listados públicos")

    encontrados = []
    for c in candidatos:
        k = name_key(c["nombre"])
        if k in objetivos:
            c["nombre_inro"] = objetivos[k]
            encontrados.append(c)
            continue
        # Coincidencia parcial por tokens (los listados invierten nombre/apellido).
        tok = set(k.split())
        for objetivo_k, objetivo_n in objetivos.items():
            ot = set(objetivo_k.split())
            if len(tok & ot) >= 3 and len(tok & ot) >= min(len(tok), len(ot)) * 0.75:
                c["nombre_inro"] = objetivo_n
                encontrados.append(c)
                break

    print(f"  {len(encontrados)} coincidencias con profesores de INRO")

    perfiles = []
    for c in encontrados:
        html = get(c["url"], pause=0.1)
        if not html:
            continue
        sopa = BeautifulSoup(html, "lxml")
        texto = texto_limpio(sopa.get_text("\n", strip=True))
        # Quitar encabezados de navegación repetidos.
        texto = re.sub(r"^(Éxito|Advertencia|Error|Perfiles PRODEP|Investigadores SNII|Cuerpos académicos|Estancias posdoctorales)\n", "", texto)
        i = texto.find("Departamento de adscripción")
        if i > 0:
            texto = texto[i:]
        pubs = len(re.findall(r"doi\.org|scopus\.com/inward", texto, re.I))
        perfiles.append(
            {
                "nombre": c.get("nombre_inro", c["nombre"]),
                "nombre_listado": c["nombre"],
                "url": c["url"],
                "datos_listado": c["datos"],
                "resumen": texto[:4000],
                "publicaciones_enlazadas": pubs,
            }
        )

    lineas = [frontmatter("Perfiles de académicos INRO", "Académicos CUCEI (listados públicos SNII/PRODEP)", f"{ACADEMICOS}/investigadores-sni", tipo="oficial")]
    lineas.append("# Perfiles de académicos que imparten INRO\n")
    lineas.append(f"{len(perfiles)} perfiles coincidentes. Fuente: Académicos CUCEI, capturado {HOY}.\n")
    for p in sorted(perfiles, key=lambda x: x["nombre"]):
        lineas.append(f"## {p['nombre']}\n")
        lineas.append(f"- Listado: {p['nombre_listado']}")
        lineas.append(f"- {p['datos_listado']}")
        lineas.append(f"- Perfil: {p['url']}")
        lineas.append(f"- Publicaciones enlazadas: {p['publicaciones_enlazadas']}\n")
        lineas.append(p["resumen"] + "\n")
    (FUENTES / "perfiles_academicos.md").write_text("\n".join(lineas), encoding="utf-8")

    print(f"  {len(perfiles)} perfiles guardados")
    return perfiles, [{"titulo": "Perfiles de académicos INRO", "url": f"{ACADEMICOS}/investigadores-sni", "archivo": "data/dataIA/fuentes/perfiles_academicos.md"}]


# --------------------------------------------------------------------------- #
# 4. Reseñas completas de MisProfesores
# --------------------------------------------------------------------------- #
def parsear_perfil_misprofes(html: str) -> dict:
    sopa = BeautifulSoup(html, "lxml")
    lineas = [l.strip() for l in sopa.get_text("\n", strip=True).splitlines() if l.strip()]

    def valor(label: str) -> str | None:
        for i, l in enumerate(lineas[:-1]):
            if l == label:
                return lineas[i + 1]
        return None

    etiquetas = []
    try:
        i = lineas.index("Etiquetas para este Profesor")
        for l in lineas[i + 1 : i + 60]:
            m = re.fullmatch(r"(.+?)\s*\((\d+)\)", l)
            if m:
                etiquetas.append({"etiqueta": m.group(1).strip(), "n": int(m.group(2))})
            elif l.startswith("Calificaciones de Estudiantes"):
                break
    except ValueError:
        pass

    try:
        j = lineas.index("Calificaciones de Estudiantes")
    except ValueError:
        j = 0

    bloques: list[list[str]] = []
    actual: list[str] = []
    for l in lineas[j + 1 :]:
        if re.fullmatch(r"\d{2}/[A-Za-z]{3}/\d{4}", l):
            if actual:
                bloques.append(actual)
            actual = [l]
        elif actual:
            actual.append(l)
    if actual:
        bloques.append(actual)

    resenas = []
    for b in bloques:
        texto = " ".join(b)
        nums = re.search(r"([\d.]+)\s+Calidad General\s+([\d.]+)\s+Facilidad", texto)
        resenas.append(
            {
                "fecha": b[0],
                "voto": next((x for x in b if x in ("BUENO", "REGULAR", "MALO")), None),
                "calidad": float(nums.group(1)) if nums else None,
                "facilidad": float(nums.group(2)) if nums else None,
                "texto": texto[:1500],
            }
        )

    return {
        "calidad_general": valor("Calidad General"),
        "recomiendan": valor("Lo Recomiendan"),
        "dificultad": valor("Nivel de Dificultad"),
        "etiquetas": etiquetas,
        "resenas": resenas,
    }


def scrape_resenas_misprofes() -> tuple[list[dict], list[dict]]:
    print("\n[4/5] Reseñas de MisProfesores")
    datos = json.loads((DATA_IA / "profesores_inro.json").read_text(encoding="utf-8"))
    objetivo = [p for p in datos["profesores"] if p["misprofes"]]
    print(f"  {len(objetivo)} fichas a consultar")

    registros = []
    for i, p in enumerate(objetivo, 1):
        html = get(p["misprofes"]["url"], pause=0.12)
        if not html:
            continue
        perfil = parsear_perfil_misprofes(html)
        registros.append(
            {
                "nombre": p["nombre"],
                "materias": [m["materia"] for m in p["materias"]],
                "url": p["misprofes"]["url"],
                **perfil,
            }
        )
        if i % 20 == 0:
            print(f"    {i}/{len(objetivo)}")

    lineas = [frontmatter("Reseñas detalladas de MisProfesores", "MisProfesores.com (apoyo, no oficial)", "https://www.misprofesores.com/escuelas/CUCEI_4047", tipo="apoyo")]
    lineas.append("# Reseñas detalladas de MisProfesores\n")
    lineas.append(f"{len(registros)} profesores. Opiniones de estudiantes, no información oficial. Capturado {HOY}.\n")
    for r in sorted(registros, key=lambda x: x["nombre"]):
        lineas.append(f"## {r['nombre']}\n")
        lineas.append(f"- Calidad general: {r['calidad_general']} | Recomiendan: {r['recomiendan']} | Dificultad: {r['dificultad']}")
        lineas.append(f"- Materias en INRO: {', '.join(r['materias']) or '—'}")
        lineas.append(f"- Perfil: {r['url']}")
        if r["etiquetas"]:
            lineas.append("- Etiquetas: " + ", ".join(f"{e['etiqueta']} ({e['n']})" for e in r["etiquetas"][:12]))
        lineas.append("")
        for res in r["resenas"][:12]:
            lineas.append(f"  - **{res['fecha']}** [{res['voto'] or '—'}] calidad {res['calidad']}, facilidad {res['facilidad']}: {res['texto']}")
        lineas.append("")
    (FUENTES / "resenas_misprofes.md").write_text("\n".join(lineas), encoding="utf-8")

    print(f"  {len(registros)} perfiles con {sum(len(r['resenas']) for r in registros)} reseñas")
    return registros, [{"titulo": "Reseñas detalladas de MisProfesores", "url": "https://www.misprofesores.com/escuelas/CUCEI_4047", "archivo": "data/dataIA/fuentes/resenas_misprofes.md"}]


# --------------------------------------------------------------------------- #
# 5. Bibliografía sugerida (Open Library)
# --------------------------------------------------------------------------- #
TEMAS = {
    "actuadores": ["electric motors and drives", "robotics actuators"],
    "algoritmia": ["introduction to algorithms", "data structures and algorithm analysis"],
    "analisis de problemas globales": ["global issues twenty first century"],
    "control": ["modern control engineering Ogata", "control systems engineering Nise"],
    "no lineal": ["nonlinear systems Khalil"],
    "diseno de interfaces": ["designing the user interface", "human computer interaction"],
    "electronica de potencia": ["power electronics Mohan"],
    "estadistica": ["probability and statistics for engineers and scientists"],
    "metodos matematicos i": ["discrete mathematics and its applications Rosen"],
    "metodos matematicos ii": ["calculus Stewart"],
    "metodos matematicos iii": ["differential equations Boyce DiPrima"],
    "modelado": ["system dynamics modeling and simulation"],
    "procesamiento digital": ["digital signal processing Oppenheim", "discrete time signal processing"],
    "programacion de sistemas embebidos": ["embedded systems programming"],
    "programacion de sistemas reconfigurables": ["reconfigurable computing FPGA"],
    "programacion": ["the C programming language Kernighan Ritchie"],
    "proyecto de sistemas": ["engineering project management"],
    "redes para circuitos": ["electronic devices and circuit theory Boylestad"],
    "robotica movil": ["introduction to autonomous mobile robots", "probabilistic robotics"],
    "sensores": ["sensors and signal conditioning", "measurement and instrumentation"],
    "sistemas inteligentes": ["artificial intelligence a modern approach", "machine learning Russell"],
}

# Consultas específicas para materias que comparten tema pero no libro.
POR_CLAVE = {
    "I9902": ["control systems engineering Nise"],
    "I9903": ["nonlinear systems Khalil", "nonlinear control systems Isidori"],
    "I9904": ["robotics modelling planning and control Siciliano"],
    "I9913": ["artificial intelligence a modern approach Russell"],
    "I9914": ["machine learning Mitchell", "artificial intelligence a modern approach Russell"],
    "I9915": ["deep learning Goodfellow", "pattern recognition Bishop"],
    "I9916": ["reinforcement learning Sutton Barto"],
    "I9891": ["digital signal processing Proakis Manolakis"],
    "I9905": ["system dynamics Ogata modeling simulation"],
    "V2455": ["global governance twenty first century"],
}


def consultar_openlib(query: str) -> list[dict]:
    html = get(OPENLIB, params={"q": query, "limit": 3, "fields": "key,title,author_name,first_publish_year,isbn"}, pause=0.12)
    if not html:
        return []
    try:
        docs = json.loads(html).get("docs", [])
    except json.JSONDecodeError:
        return []
    salida = []
    for d in docs:
        titulo = d.get("title", "")
        if re.search(r"solution|instructor|answer key|workbook|teacher'?s? (edition|manual)", titulo, re.I):
            continue
        isbn = (d.get("isbn") or [])
        salida.append(
            {
                "titulo": d.get("title", ""),
                "autores": d.get("author_name", [])[:3],
                "anio": d.get("first_publish_year"),
                "isbn": isbn[0] if isbn else None,
                "url": "https://openlibrary.org" + d["key"] if d.get("key") else None,
            }
        )
    return salida


def scrape_bibliografia() -> tuple[list[dict], list[dict]]:
    print("\n[5/5] Bibliografía sugerida")
    datos = json.loads((DATA_IA / "profesores_inro.json").read_text(encoding="utf-8"))
    registros = []
    for m in datos["materias"]:
        nombre = norm(m["nombre"]).lower()
        tema = None
        for clave_tema in sorted(TEMAS, key=len, reverse=True):
            if clave_tema in nombre:
                tema = clave_tema
                break
        if not tema:
            continue
        consultas = POR_CLAVE.get(m["clave"], TEMAS[tema][:2])
        libros = []
        for query in consultas[:2]:
            libros.extend(consultar_openlib(query))
            if len(libros) >= 3:
                break
        vistos, unicos = set(), []
        for l in libros:
            k = norm(l["titulo"])
            if k and k not in vistos:
                vistos.add(k)
                unicos.append(l)
        registros.append(
            {
                "clave": m["clave"],
                "materia": titulo_legible(m["nombre"]),
                "tema": tema,
                "libros": unicos[:3],
            }
        )

    lineas = [frontmatter("Bibliografía sugerida por materia", "Open Library (sugerencia NO oficial)", "https://openlibrary.org", {"advertencia": "No es la bibliografía oficial de la UdeG"}, tipo="apoyo")]
    lineas.append("# Bibliografía sugerida por materia\n")
    lineas.append("> **Importante:** esta lista es una sugerencia generada automáticamente a partir de Open Library. **No** corresponde a la bibliografía oficial de la UdeG. Para la bibliografía oficial consulta el programa de la asignatura con tu coordinación.\n")
    lineas.append(f"Capturado {HOY}.\n")
    for r in sorted(registros, key=lambda x: x["materia"]):
        lineas.append(f"## {r['materia']} (`{r['clave']}`)\n")
        if r["libros"]:
            for l in r["libros"]:
                autores = ", ".join(l["autores"]) if l["autores"] else "Autor no disponible"
                lineas.append(f"- **{l['titulo']}** — {autores} ({l['anio'] or 's/a'}) [Open Library]({l['url']})")
        else:
            lineas.append("- _Sin resultados_")
        lineas.append("")
    (FUENTES / "bibliografia_sugerida.md").write_text("\n".join(lineas), encoding="utf-8")

    print(f"  {len(registros)} materias con sugerencias")
    return registros, [{"titulo": "Bibliografía sugerida por materia", "url": "https://openlibrary.org", "archivo": "data/dataIA/fuentes/bibliografia_sugerida.md"}]


# --------------------------------------------------------------------------- #
def main() -> None:
    FUENTES.mkdir(parents=True, exist_ok=True)

    paginas = scrape_sitio_oficial()
    catalogo, doc_catalogo = scrape_catalogo()
    academicos, doc_academicos = scrape_academicos()
    resenas, doc_resenas = scrape_resenas_misprofes()
    bibliografia, doc_biblio = scrape_bibliografia()

    indice = [frontmatter("Índice de fuentes adicionales", "Generado localmente", "")]
    indice.append("# Fuentes adicionales para el chat INRO\n")
    indice.append(f"Generado: {datetime.now().isoformat(timespec='seconds')}\n")
    indice.append("Todos los archivos incluyen `fecha_captura`, `fuente` y `url` en el encabezado.")
    indice.append("El campo `tipo` distingue información **oficial** de información de **apoyo**.\n")
    indice.append("| Fuente | Documento | Registros | Tipo |")
    indice.append("| --- | --- | --- | --- |")
    indice.append(f"| Sitio oficial INRO | `sitio_oficial/` | {len(paginas)} páginas | oficial |")
    indice.append(f"| Catálogo SIIAU | `catalogo_materias.md` | {len(catalogo)} materias | oficial |")
    indice.append(f"| Académicos CUCEI | `perfiles_academicos.md` | {len(academicos)} perfiles | oficial |")
    indice.append(f"| MisProfesores | `resenas_misprofes.md` | {len(resenas)} profesores | apoyo |")
    indice.append(f"| Open Library | `bibliografia_sugerida.md` | {len(bibliografia)} materias | apoyo (NO oficial) |")
    (FUENTES / "INDICE.md").write_text("\n".join(indice) + "\n", encoding="utf-8")

    consolidado = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "sitio_oficial": paginas,
        "catalogo_materias": catalogo,
        "perfiles_academicos": academicos,
        "resenas_misprofes": resenas,
        "bibliografia_sugerida": bibliografia,
        "documentos": [doc_catalogo, doc_academicos, doc_resenas, doc_biblio],
    }
    (DATA_IA / "contexto_extra.json").write_text(json.dumps(consolidado, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nOK", FUENTES / "INDICE.md")
    print("OK", DATA_IA / "contexto_extra.json")


if __name__ == "__main__":
    main()
