"""Recuperación semántica y generación de respuestas con Gemini (Vertex AI).

El flujo es:
    pregunta -> embedding -> fragmentos más parecidos -> prompt con citas -> respuesta

El metadata `tipo` y `fecha_captura` viaja en cada fragmento para que el modelo
distinga información oficial de opiniones o sugerencias.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

INDEX_DIR = ROOT / "data" / "index"
DATA_DIR = ROOT / "data" / "dataIA"
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
EMBED_LOCATION = os.environ.get("EMBEDDING_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
CHAT_LOCATION = os.environ.get("GENERATION_LOCATION", "global")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
DIMENSIONES = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))
TOP_K = int(os.environ.get("TOP_K", "6"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "2048"))

# Pequeños ajustes para que lo oficial gane desempates sin ocultar las opiniones.
BONO_TIPO = {"oficial": 0.03, "mixto": 0.0, "apoyo": -0.02}
ROMANOS = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5"}
MAX_HISTORIAL = 8
ALIAS_TEMAS = {
    "PRECALCULO": ("MATEMAT",),
    "CALCULO": ("MATEMAT",),
    "MATEMATICA": ("MATEMAT",),
    "ALGEBRA": ("MATEMAT",),
    "PROGRAMACION": ("PROGRAMACION", "ALGORITMIA"),
    "ROBOTICA": ("ROBOTICA",),
}
PALABRAS_ANIO = {
    "DIECIOCHO": "2018",
    "VEINTIDIECIOCHO": "2018",
    "VEINTIOCHO": "2018",
    "VEINTICINCO": "2025",
}

SISTEMA = """Hablas como un compañero de Ingeniería Robótica del CUCEI que ya revisó la información de la carrera. Suena natural, directo y amable. Nada de tono de informe, tribunal ni robot.

Reglas:
1. Usa solo lo que está en el CONTEXTO o en la CONVERSACIÓN RECIENTE. No inventes datos de la carrera.
2. NUNCA escribas citas numeradas: ni [1], ni (2), ni "fuente 1", ni "según el contexto". La trazabilidad va aparte en el JSON; el usuario no la ve.
3. Escribe como hablaría un estudiante: frases cortas, español natural, máximo 6 líneas. La app muestra texto plano, así que NO uses markdown: nada de asteriscos para resaltar (**), ni títulos con #. Si ayudan, usa listas cortas con guiones o números.
4. Si algo es opinión o viene de MisProfesores/Open Library, dilo de forma casual: "ojo, eso es lo que dicen los alumnos, no algo oficial". No escribas "tipo mixto" ni "tipo apoyo" salvo que te lo pregunten.
5. Si no hay información suficiente de lo administrativo de la carrera como tal, di: "No tengo un dato seguro de eso por aquí; mejor lo vemos con la Coordinación de Ingeniería Robótica." No suenes a disco rayado: no repitas esa frase dos veces seguidas.
5.1 Si no hay información suficiente de algo fuera de lo administrativo de la carrera, di: "No tengo un dato seguro de eso por aquí; deberias de investigarlo con algun profesor." No suenes a disco rayado: no repitas esa frase dos veces seguidas.
6. Si preguntan por malla y NO hay un dato confirmado de año o plan en la conversación, pregúntalo una sola vez. Si el dato ya aparece en "DATO CONFIRMADO", responde directo y no lo vuelvas a preguntar.
7. Si solo preguntan quién da una materia, da los nombres y ya. Si preguntan a quién recomendar sin decir materia ni criterio, pregunta cuál materia y qué buscan, sin soltar una lista genérica.
8. No repitas el contexto palabra por palabra: reformúlalo como lo diría un compañero.
9. Si la pregunta actual es corta o dice "ese/esa/lo anterior", usa la CONVERSACIÓN RECIENTE para saber de qué habla. Si menciona algo nuevo, cambia de tema.
10. Si preguntan por recomendaciones de un profesor y en el contexto NO hay ninguna opinión o calificación real, di: "De ese profe no tengo opiniones aquí; mejor pregúntale a la banda de CUCEI Social en Facebook." No inventes recomendaciones.
11. Si preguntan cuál profesor es "mejor" o "el mejor", compara las calificaciones reales del contexto y suelta el nombre. Avisa de forma casual que son opiniones de alumnos, no algo oficial.
12. Si el usuario dice "eso no es cierto" o te corrige, NO digas que perdiste el contexto: pide que señale qué parte no le cuadra y sigue en el mismo tema.
"""


@lru_cache(maxsize=1)
def cargar_indice() -> tuple[list[dict], np.ndarray]:
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(INDEX_DIR / "embeddings.npy")
    return chunks, vectors


@lru_cache(maxsize=1)
def _datos_profesores() -> dict:
    """Fuente estructurada local: nombre -> materias y calificaciones."""
    ruta = DATA_DIR / "profesores_inro.json"
    if not ruta.exists():
        return {"profesores": []}
    return json.loads(ruta.read_text(encoding="utf-8"))


def _cliente(location: str) -> genai.Client:
    if not PROJECT:
        raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT en .env")
    return genai.Client(vertexai=True, project=PROJECT, location=location)


def normalizar(texto: str) -> str:
    """Quita acentos y deja solo mayúsculas, números y espacios."""
    sin_acentos = unicodedata.normalize("NFD", str(texto or ""))
    sin_acentos = "".join(c for c in sin_acentos if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Z0-9]+", " ", sin_acentos.upper()).strip()


def _plan_de_anio(anio: int) -> str:
    """El plan nuevo de 2025 arranca para ingresos en 2026."""
    return "2018" if anio <= 2025 else "2025"


def _plan_en_texto(texto: str) -> str | None:
    """Saca el plan del año de ingreso o de una mención explícita."""
    plano = normalizar(texto)
    if not plano:
        return None
    mencion = re.search(
        r"\b(?:PLAN|MALLA|PROGRAMA)\s+(?:DE\s+)?(20\d{2}|[A-Z]+)\b", plano
    )
    if mencion:
        valor = mencion.group(1)
        if valor.isdigit():
            return valor
        if valor in PALABRAS_ANIO:
            return PALABRAS_ANIO[valor]
    ingreso = re.search(
        r"\b(?:INGRES|ENTR|INGRESO|A[ÑN]O|ANO)\w*\s+(?:EN\s+)?(20\d{2})\b", plano
    )
    if ingreso:
        return _plan_de_anio(int(ingreso.group(1)))
    anio = re.search(r"\b(20\d{2})\b", plano)
    if anio and 2018 <= int(anio.group(1)) <= 2026:
        return _plan_de_anio(int(anio.group(1)))
    return None


def _es_respuesta_anio(pregunta: str, historial: list[dict]) -> bool:
    """'2025' o 'ingrese en 2025' después de que preguntamos el año."""
    if not historial:
        return False
    anterior = normalizar(historial[-1].get("respuesta", ""))
    if not re.search(r"\b(A[ÑN]O|PLAN|INGRES\w*|ENTR\w*|INGRESO)\b", anterior):
        return False
    actual = normalizar(pregunta)
    return bool(
        re.fullmatch(r"(?:EL\s+)?(?:A[ÑN]O\s+)?20\d{2}", actual)
        or re.match(r"^(?:INGRES\w*|ENTR\w*|INGRESO|A[ÑN]O)\s+(?:EN\s+)?20\d{2}\b", actual)
    )


def _respuesta_anio(pregunta: str) -> str:
    anio = re.search(r"20\d{2}", normalizar(pregunta))
    ingreso = int(anio.group()) if anio else 2025
    plan = _plan_de_anio(ingreso)
    aviso = (
        "El plan nuevo de 2025 recién aplica para quienes entran desde 2026."
        if plan == "2018"
        else "Ya quedas en la malla nueva."
    )
    return (
        f"Si entraste en {ingreso}, te toca el Plan {plan}. {aviso} "
        "Ya lo tengo en cuenta, no te lo vuelvo a preguntar."
    )


def _plan_en_historial(historial: list[dict]) -> str | None:
    for turno in reversed(historial):
        plan = _plan_en_texto(turno.get("pregunta", ""))
        if plan:
            return plan
    for turno in reversed(historial):
        respuesta = turno.get("respuesta", "")
        plano = normalizar(respuesta)
        mencion = re.search(
            r"\b(?:TE TOCA|TU PLAN|TU MALLA|ESTAS EN|CURSAS)\b.*?\b(20\d{2})\b", plano
        )
        if mencion:
            return _plan_de_anio(int(mencion.group(1)))
        plan = _plan_en_texto(respuesta)
        if plan:
            return plan
    return None


def _pide_recomendacion(pregunta: str) -> bool:
    """'te recomiendo' es una opinión del estudiante, no un pedido."""
    if re.search(r"\bte recomiend", pregunta, re.I):
        return False
    return bool(
        re.search(
            r"recomiend|recomendar|mejor\s+(?:profe|profesor|prof|maestro)",
            pregunta,
            re.I,
        )
    )


def _es_desacuerdo(pregunta: str) -> bool:
    return bool(
        re.search(
            r"eso no es cierto|no es cierto|no es verdad|est[aá]s mal|estas mal|mentira|no coincide",
            pregunta,
            re.I,
        )
    )


def _respuesta_desacuerdo(historial: list[dict]) -> str:
    tema = historial[-1].get("pregunta", "").strip().rstrip("?") if historial else ""
    if tema:
        return (
            f"¿Cuál parte no te cuadra? Si te refieres a lo de «{tema}», dime y lo vuelvo a revisar. "
            "Si es una opinión de profesores, mejor la contrastamos en CUCEI Social."
        )
    return (
        "¿Cuál parte no te cuadra? Dime y lo vuelvo a revisar; si es opinión de profesores, "
        "mejor la contrastamos en CUCEI Social."
    )


def _evaluar_numerico(expresion: str) -> float | None:
    """Calculo mínimo y seguro: solo números, + - * / y paréntesis."""
    try:
        nodo = ast.parse(expresion, mode="eval").body
    except (SyntaxError, ValueError, TypeError):
        return None

    def valor(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = valor(n.operand)
            return None if v is None else (v if isinstance(n.op, ast.UAdd) else -v)
        if isinstance(n, ast.BinOp):
            a, b = valor(n.left), valor(n.right)
            if a is None or b is None:
                return None
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div) and b:
                return a / b
            if isinstance(n.op, ast.Pow):
                return a ** b
        return None

    return valor(nodo)


def _respuesta_operacion(pregunta: str) -> str | None:
    coincidencia = re.fullmatch(
        r"\s*x\s*=\s*(-?\d+(?:\.\d+)?)\s*[,;]\s*([0-9xX\s.+*/()\-]+?)\s*=\s*\?\s*",
        pregunta,
    )
    if not coincidencia:
        return None
    valor_x, expresion = coincidencia.groups()
    calculo = re.sub(r"(?<![\w.])[xX](?![\w.])", f"({valor_x})", expresion)
    resultado = _evaluar_numerico(calculo)
    if resultado is None:
        return None
    return f"{expresion.strip()} = {resultado:g}."


def _respuesta_conversacional(pregunta: str) -> str | None:
    if not re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]", pregunta):
        return "¿Eh? Escríbeme una pregunta y te echo la mano."
    texto = pregunta.lower().strip()
    if re.search(r"c[oó]mo te sientes|c[oó]mo est[aá]s|qu[eé] tal est[aá]s", texto):
        return "Todo bien, aquí ando al cien checando lo de la carrera. ¿En qué te ayudo?"
    if re.search(r"qu[eé] modelo|gemini eres|qu[eé] ia eres|versi[oó]n de gemini", texto):
        return (
            f"Soy {CHAT_MODEL} de Google, corriendo en Vertex AI. "
            "Pero anda, aquí ando de compañero de Ingeniería Robótica."
        )
    if re.fullmatch(r"(?:hola|buenas|qu[eé] onda|que onda|hey)[.!? ]*", texto):
        return "¡Qué onda! ¿En qué te echo la mano?"
    if re.fullmatch(r"(?:gracias|mil gracias)[.!? ]*", texto):
        return "¡Va! Si se te ofrece otra cosa, aquí ando."
    if re.search(r"acoso|hostigamiento|bullying", texto):
        return (
            "No tengo datos de eso en lo que revisé. Si es algo que te está pasando, "
            "mejor pide apoyo a la Coordinación de Ingeniería Robótica o a las instancias "
            "del CUCEI; yo no puedo darte un dato seguro."
        )
    if re.search(r"lugar geom[eé]trico de las ra[ií]ces|root locus", texto):
        return (
            "Es el root locus: por dónde se mueven las raíces de la ecuación característica "
            "al subir la ganancia K. Sirve para ver si el sistema se queda estable. "
            "Eso sí, es concepto general de control, no viene del corpus de la carrera."
        )
    return None


def _nombres_en_texto(texto: str, profesores: list[dict]) -> list[dict]:
    plano = normalizar(texto)
    palabras = plano.split()
    encontrados = []
    for prof in profesores:
        candidatos = {
            normalizar(prof.get("nombre", "")),
            normalizar(prof.get("nombre_normalizado", "")),
            normalizar((prof.get("misprofes") or {}).get("nombre", "")),
            normalizar((prof.get("misprofes") or {}).get("nombre_normalizado", "")),
        }
        candidatos.discard("")
        if not any(all(t in palabras for t in c.split()) for c in candidatos):
            continue
        posicion = min(plano.find(c.split()[0]) for c in candidatos)
        encontrados.append((posicion, prof))
    return [prof for _, prof in sorted(encontrados, key=lambda par: par[0])]


def _titulo_materia(nombre: str) -> str:
    texto = str(nombre or "").title()
    for pequeña in ("De", "Del", "La", "Los", "Y", "Para", "Con", "Por"):
        texto = re.sub(rf"\b{pequeña}\b", pequeña.lower(), texto)
    for romano, arreglo in (("Ii", "II"), ("Iii", "III"), ("Iv", "IV")):
        texto = re.sub(rf"\b{romano}\b", arreglo, texto)
    return texto


def _nombre_amable(nombre: str) -> str:
    texto = str(nombre or "").strip()
    if not texto:
        return texto
    partes = texto.split()
    # MisProfesores trae algunos como "Parra Escamilla GELIZTLE ALEJANDRA".
    if not texto.isupper():
        for i, parte in enumerate(partes[1:], 1):
            if len(parte) > 2 and parte == parte.upper():
                texto = " ".join(partes[i:] + partes[:i])
                break
    if texto == texto.upper():
        texto = texto.title()
        for pequeña in ("De", "La", "Los", "Y", "Del"):
            texto = re.sub(rf"\b{pequeña}\b", pequeña.lower(), texto)
    return texto


def _temas_de_la_pregunta(pregunta: str, chunks: list[dict]) -> list[str]:
    """Temas dichos en la pregunta: 'de los de control', 'de métodos', etc."""
    temas: list[str] = []
    for clave in detectar_claves(pregunta, chunks):
        for chunk in chunks:
            if chunk.get("estructurado") and chunk.get("clave") == clave:
                temas.extend(normalizar(chunk.get("nombre_materia", "")).split())
    plano = normalizar(pregunta)
    if not temas:
        for palabra in (
            "CONTROL",
            "MATEMAT",
            "SISTEMAS",
            "ROBOTICA",
            "ALGORITMIA",
            "PROGRAMACION",
            "REDES",
            "SENSORES",
        ):
            if palabra in plano:
                temas.append(palabra)
    return [tema for tema in temas if len(tema) > 2]


def _respuesta_materias_de_ellos(
    pregunta: str, historial: list[dict], chunks: list[dict]
) -> dict | None:
    if not historial:
        return None
    if not re.search(
        r"cada uno de ellos|cada uno|de esos|de ellos|los anteriores|qu[eé] materias dan|cu[aá]les dan",
        pregunta,
        re.I,
    ):
        return None

    profesores = _datos_profesores().get("profesores", [])
    temas = _temas_de_la_pregunta(pregunta, chunks)

    # Si la pregunta ya nombra profesores, esos son los referentes.
    elegidos = _nombres_en_texto(pregunta, profesores)
    if not elegidos:
        # Buscar en TODA la conversación, no solo en la última respuesta.
        # Primero el turno más reciente que hable del mismo tema; si no,
        # el más reciente con una lista de profesores.
        for turno in reversed(historial):
            texto = f"{turno.get('pregunta', '')} {turno.get('respuesta', '')}"
            encontrados = _nombres_en_texto(texto, profesores)
            if temas:
                filtrados = [
                    prof
                    for prof in encontrados
                    if any(
                        tema in normalizar(materia["materia"])
                        for materia in prof.get("materias", [])
                        for tema in temas
                    )
                ]
                if filtrados:
                    elegidos = filtrados
                    break
            elif len(encontrados) >= 2:
                elegidos = encontrados
                break
            elif encontrados and not elegidos:
                elegidos = encontrados
    if not elegidos:
        return None

    elegidos = elegidos[:10]
    lineas = []
    for prof in elegidos:
        nombre = _nombre_amable(
            (prof.get("misprofes") or {}).get("nombre") or prof.get("nombre", "")
        )
        materias = [
            f"{_titulo_materia(m['materia'])} ({m['clave']})"
            for m in prof.get("materias", [])
        ]
        detalle = ", ".join(materias) if materias else "no tengo materias registradas"
        lineas.append(f"- {nombre}: {detalle}.")

    fuentes = []
    for chunk in chunks:
        texto = normalizar(chunk.get("texto", ""))
        if any(
            all(t in texto.split() for t in normalizar(prof["nombre"]).split())
            for prof in elegidos
        ):
            copia = dict(chunk)
            copia["score"] = 0.0
            fuentes.append(copia)
        if len(fuentes) >= 6:
            break

    encabezado = (
        "De los que estábamos viendo:"
        if temas
        else "Te los paso por nombre, con lo que sale en SIIAU:"
    )
    return {
        "respuesta": f"{encabezado}\n" + "\n".join(lineas),
        "fuentes": _fuentes(fuentes),
    }


def detectar_claves(
    pregunta: str, chunks: list[dict], historial: list[dict] | None = None
) -> set[str]:
    """Reconoce 'metodos matematicos 3', 'III' o 'I9885'.

    Si la pregunta actual no nombra materia, hereda la del historial reciente.
    """
    alias: dict[str, set[str]] = {}
    for chunk in chunks:
        if not chunk.get("estructurado") or not chunk.get("clave"):
            continue
        clave = chunk["clave"].upper()
        nombre = normalizar(chunk.get("nombre_materia", ""))
        if not nombre:
            continue
        variantes = {nombre}
        for romano, arabe in ROMANOS.items():
            variantes.add(re.sub(rf"\b{romano}\b", arabe, nombre))
        alias.setdefault(clave, set()).update(variantes)

    def buscar(texto: str) -> set[str]:
        consulta = normalizar(texto)
        encontradas: set[str] = set()
        for clave, nombres in alias.items():
            if re.search(rf"(?<!\w){re.escape(clave)}(?!\w)", consulta):
                encontradas.add(clave)
                continue
            for nombre in sorted(nombres, key=len, reverse=True):
                if re.search(rf"(?<!\w){re.escape(nombre)}(?!\w)", consulta):
                    encontradas.add(clave)
                    break
        return encontradas

    encontradas = buscar(pregunta)
    if (
        not encontradas
        and historial
        and _consulta_con_contexto(pregunta, historial) != pregunta
    ):
        for turno in reversed(historial[-MAX_HISTORIAL:]):
            encontradas = buscar(
                f"{turno.get('pregunta', '')} {turno.get('respuesta', '')}"
            )
            if encontradas:
                break
    return encontradas


def es_pregunta_de_profesores(pregunta: str) -> bool:
    return bool(
        re.search(
            r"profesor|profe\b|profes\b|maestro|docente|imparte|imparten|calificaci|reseña|opini|qui[eé]n|\bda\b|\bdan\b|\bclase\b",
            pregunta,
            re.I,
        )
    )


def es_solo_lista_de_profesores(pregunta: str) -> bool:
    """El usuario solo quiere nombres: sin opiniones ni calificaciones."""
    pregunta_baja = pregunta.lower()
    pide_nombres = re.search(
        r"profesor|profe\b|maestro|docente|imparte|imparten|\bda\b|\bdan\b|\bclase\b|qui[eé]n",
        pregunta_baja,
    )
    pide_opiniones = re.search(
        r"opini|calificaci|reseña|recomiend|recomendar|recomendaci|promedio|suerte|mejor|c[óo]mo es",
        pregunta_baja,
    )
    return bool(pide_nombres) and not bool(pide_opiniones)


def _limpiar_respuesta(texto: str) -> str:
    """Quita citas y espacios raros aunque el modelo los vuelva a escribir."""
    texto = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", texto)
    texto = re.sub(r"\(\s*\d{1,2}\s*\)", "", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def _historial_reciente(historial: list[dict] | None) -> list[dict]:
    return list(historial or [])[-MAX_HISTORIAL:]


def _consulta_con_contexto(pregunta: str, historial: list[dict] | None) -> str:
    """Para '¿y quién la da?', busca con la materia de la pregunta anterior."""
    historial = _historial_reciente(historial)
    if not historial:
        return pregunta

    corta_o_deictica = (
        len(pregunta.split()) <= 5
        or re.search(
            r"\b(ese|esa|estos|estas|el mismo|la misma|lo anterior|además|ademas|también|tambien|entonces|sigue|igual|los dos|las dos|de esos|de esas|de ellos|de ellas|de los anteriores|de las anteriores|entre ellos|cu[aá]l de)\b",
            pregunta,
            re.I,
        )
    )
    # Un pedido de recomendación también mira lo que estábamos hablando,
    # aunque la pregunta no sea corta ("¿cuál de los dos me recomiendas?").
    if not corta_o_deictica and not re.search(
        r"recomiend|recomendar|mejor|profe|profesor|qui[eé]n|cu[aá]l de",
        pregunta,
        re.I,
    ):
        return pregunta

    anterior = historial[-1].get("pregunta", "")
    return f"{anterior} {pregunta}".strip() if anterior else pregunta


def _historial_contexto(historial: list[dict] | None) -> str:
    historial = _historial_reciente(historial)
    if not historial:
        return ""
    lineas = []
    for turno in historial:
        lineas.append(
            f"Estudiante: {turno.get('pregunta', '')}\n"
            f"Asistente: {turno.get('respuesta', '')}"
        )
    return "CONVERSACIÓN RECIENTE:\n" + "\n".join(lineas)


def _sugerir_materias(pregunta: str, chunks: list[dict]) -> list[tuple[str, str]]:
    """Análogos reales del catálogo para nombres que no existen literalmente."""
    consulta = normalizar(pregunta)
    temas: set[str] = set()
    for alias, palabras in ALIAS_TEMAS.items():
        if alias in consulta:
            temas.update(palabras)
    if not temas:
        return []

    sugeridas: list[tuple[str, str]] = []
    vistas: set[str] = set()
    for chunk in chunks:
        if not chunk.get("estructurado"):
            continue
        clave, nombre = chunk.get("clave", ""), chunk.get("nombre_materia", "")
        if not clave or clave in vistas:
            continue
        normalizado = normalizar(nombre)
        if "SEMINARIO" in normalizado:
            continue
        if any(tema in normalizado for tema in temas):
            sugeridas.append((clave, nombre))
            vistas.add(clave)
    return sugeridas[:3]


def _respuesta_sin_opiniones(fragmentos: list[dict]) -> str:
    chunk = fragmentos[0]
    materia = chunk.get("nombre_materia", chunk.get("seccion", "esa materia"))
    clave = chunk.get("clave", "")
    nombre = f"{materia} ({clave})" if clave else materia
    return (
        f"De {nombre} no tengo opiniones de profesores por aquí. "
        "Mejor pregúntale a la banda de CUCEI Social en Facebook; ahí sí sueltan experiencias reales."
    )


def recuperar(
    pregunta: str,
    k: int = TOP_K,
    historial: list[dict] | None = None,
    claves: set[str] | None = None,
) -> list[dict]:
    """Devuelve los fragmentos más relevantes, con oficiales ganando desempates."""
    chunks, vectors = cargar_indice()
    consulta = _consulta_con_contexto(pregunta, historial)

    cliente = _cliente(EMBED_LOCATION)
    respuesta = cliente.models.embed_content(
        model=EMBED_MODEL,
        contents=[consulta],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIMENSIONES,
        ),
    )
    q = np.asarray(respuesta.embeddings[0].values, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)

    similitudes = vectors @ q

    # El peso de cada tipo depende de lo que pregunte el estudiante.
    pesos = dict(BONO_TIPO)
    if re.search(
        r"profesor|profe|maestro|docente|imparte|quién da|quien da|calificaci|reseña|recomiend|opini|mejor",
        pregunta,
        re.I,
    ):
        pesos["mixto"] = 0.06
        pesos["oficial"] = 0.01
    if re.search(r"libro|bibliograf|leer|texto de apoyo", pregunta, re.I):
        pesos["apoyo"] = 0.04

    bonos = np.array([pesos.get(c.get("tipo", "oficial"), 0.0) for c in chunks], dtype=np.float32)
    ranking = list(np.argsort(-(similitudes + bonos)))

    if claves is None:
        claves = detectar_claves(pregunta, chunks, historial)
    if claves and es_pregunta_de_profesores(pregunta):
        # El bloque estructurado de esa materia va primero, aunque la tabla tenga
        # mejor similitud superficial: es la fuente limpia de SIIAU.
        mejores: dict[str, int] = {}
        for pos, chunk in enumerate(chunks):
            if chunk.get("estructurado") and chunk.get("clave") in claves:
                clave = chunk["clave"]
                if clave not in mejores or similitudes[pos] > similitudes[mejores[clave]]:
                    mejores[clave] = pos
        prioritarios = sorted(mejores.values(), key=lambda pos: -similitudes[pos])
        if prioritarios and es_solo_lista_de_profesores(pregunta):
            # Basta el bloque de esa materia: ni tablas, ni opiniones, ni extra.
            orden = prioritarios[:1]
        else:
            vistos = set(prioritarios)
            # No mezclar bloques limpios de otras materias en la misma respuesta.
            relevantes = [
                pos
                for pos in ranking
                if pos not in vistos
                and not (
                    chunks[pos].get("estructurado")
                    and chunks[pos].get("clave") not in claves
                )
            ]
            orden = (prioritarios + relevantes)[:k]
    else:
        orden = ranking[:k]

    resultados = []
    for pos in orden:
        chunk = dict(chunks[int(pos)])
        chunk["score"] = round(float(similitudes[int(pos)]), 4)
        resultados.append(chunk)
    return resultados


def _contexto(fragmentos: list[dict]) -> str:
    bloques = []
    for i, c in enumerate(fragmentos, 1):
        cabecera = (
            f"[{i}] {c['titulo']} | tipo: {c['tipo']} | "
            f"fuente: {c['fuente']} | fecha: {c['fecha_captura'] or 'no disponible'}"
        )
        if c.get("advertencia"):
            cabecera += f" | advertencia: {c['advertencia']}"
        bloques.append(f"{cabecera}\n{c['texto']}")
    return "\n\n---\n\n".join(bloques)


def _texto_visible(respuesta) -> str:
    """Descarta partes de razonamiento interno que Gemini puede emitir."""
    partes = [
        parte.text
        for parte in respuesta.candidates[0].content.parts
        if parte.text and not getattr(parte, "thought", False)
    ]
    return "".join(partes).strip()


def _fuentes(fragmentos: list[dict]) -> list[dict]:
    return [
        {
            "numero": i,
            "titulo": c["titulo"],
            "tipo": c["tipo"],
            "fuente": c["fuente"],
            "url": c["url"],
            "fecha_captura": c["fecha_captura"],
            "extracto": c["texto"][:220],
            "score": c["score"],
        }
        for i, c in enumerate(fragmentos, 1)
    ]


def _respuesta_directa(fragmentos: list[dict]) -> str | None:
    """Para 'quién da X', copiar los nombres sin pasar por el modelo."""
    chunk = fragmentos[0]
    if not chunk.get("estructurado"):
        return None

    nombres = [
        linea[2:].strip()
        for linea in chunk["texto"].splitlines()
        if linea.startswith("- ")
    ]
    if not nombres:
        return None

    materia = chunk.get("nombre_materia", chunk.get("seccion", ""))
    clave = chunk.get("clave", "")
    encabezado = f"{materia} ({clave})" if clave else materia
    lista = "\n".join(f"- {nombre}" for nombre in nombres)
    return f"Los profesores registrados para dar {encabezado} son:\n{lista}"


def _opiniones_de_claves(claves: set[str], chunks: list[dict], limite: int = 8) -> str:
    """Reseñas completas del JSON local para comparar profesores de una materia."""
    nombres: dict[str, str] = {}
    for chunk in chunks:
        if chunk.get("estructurado") and chunk.get("clave") in claves:
            nombre = normalizar(chunk.get("nombre_materia", ""))
            if nombre:
                nombres[nombre] = chunk["clave"]
    if not nombres:
        return ""

    lineas = []
    for prof in _datos_profesores().get("profesores", []):
        if not {m["clave"] for m in prof.get("materias", [])} & claves:
            continue
        nombre = _nombre_amable(
            (prof.get("misprofes") or {}).get("nombre") or prof.get("nombre", "")
        )
        for reseña in prof.get("recomendaciones", []):
            materia = normalizar(reseña.get("materia", ""))
            if not any(
                re.search(rf"(?<!\w){re.escape(n)}(?!\w)", materia) for n in nombres
            ):
                continue
            texto = re.sub(r"\s+", " ", reseña.get("resena", "")).strip()
            if texto:
                lineas.append(f"- {nombre} ({reseña.get('materia')}): {texto[:450]}")
            if len(lineas) >= limite:
                return "\n".join(lineas)
    return "\n".join(lineas)


def _es_robotica_generica(pregunta: str, chunks: list[dict]) -> bool:
    """'robótica' a secas no es Robótica Móvil: hablamos de toda la carrera."""
    plano = normalizar(pregunta).split()
    if "ROBOTICA" not in plano or "MOVIL" in plano:
        return False
    if detectar_claves(pregunta, chunks):
        return False
    return bool(re.search(r"profe|profesor|mejor|recomiend|dan|imparte", pregunta, re.I))


def _respuesta_robotica_general() -> str:
    candidatos = []
    for prof in _datos_profesores().get("profesores", []):
        datos = prof.get("misprofes") or {}
        promedio, opiniones = datos.get("promedio"), datos.get("calificaciones", 0) or 0
        if not promedio or opiniones < 5:
            continue
        nombre = _nombre_amable(datos.get("nombre") or prof.get("nombre", ""))
        materias = prof.get("materias", [])
        primera = (
            _titulo_materia(materias[0]["materia"])
            if materias
            else "sin materia registrada"
        )
        candidatos.append((float(promedio), int(opiniones), nombre, primera))
    candidatos.sort(key=lambda item: (item[0], item[1]), reverse=True)
    lineas = [
        f"- {nombre}: {promedio:g} ({opiniones} opiniones) — {materia}"
        for promedio, opiniones, nombre, materia in candidatos[:3]
    ]
    return (
        "Si dices robótica a secas, hablo de la carrera completa: Control, Robótica Móvil, "
        "Sistemas Inteligentes, Sistemas Embebidos y Visión Robótica, entre otras.\n"
        "De los profes con suficientes opiniones, los que van arriba son:\n"
        + "\n".join(lineas)
        + "\nDime una materia y te digo a quién conviene. "
        "Ojo, son opiniones de alumnos, no algo oficial."
    )


def responder(pregunta: str, historial: list[dict] | None = None) -> dict:
    chunks, _ = cargar_indice()
    historial = _historial_reciente(historial)

    respuesta = _respuesta_conversacional(pregunta)
    if respuesta:
        return {"respuesta": respuesta, "fuentes": []}

    respuesta = _respuesta_operacion(pregunta)
    if respuesta:
        return {"respuesta": respuesta, "fuentes": []}

    if _es_desacuerdo(pregunta):
        return {"respuesta": _respuesta_desacuerdo(historial), "fuentes": []}

    respuesta = _respuesta_materias_de_ellos(pregunta, historial, chunks)
    if respuesta:
        return respuesta

    if _es_respuesta_anio(pregunta, historial):
        return {"respuesta": _respuesta_anio(pregunta), "fuentes": []}

    claves = detectar_claves(pregunta, chunks, historial)
    pide_recomendacion = _pide_recomendacion(pregunta)

    if _es_robotica_generica(pregunta, chunks):
        return {"respuesta": _respuesta_robotica_general(), "fuentes": []}

    if pide_recomendacion and not claves:
        opciones = _sugerir_materias(pregunta, chunks)
        if len(opciones) == 1:
            # "mejor profe de robótica": solo existe Robótica Móvil, no hace falta preguntar.
            claves = {opciones[0][0]}
        elif opciones:
            lista = "\n".join(f"- {nombre} ({clave})" for clave, nombre in opciones)
            return {
                "respuesta": (
                    "No tengo una materia que se llame exactamente así, pero sí tengo "
                    f"estas parecidas:\n{lista}\nDime cuál y te digo a quién conviene."
                ),
                "fuentes": [],
            }
        else:
            return {
                "respuesta": (
                    "¿Para qué materia? Dime cuál y qué buscas (que explique bien, que sea "
                    "flexible, que no cargue mucho) y te digo sin rodeos."
                ),
                "fuentes": [],
            }

    plan = _plan_en_texto(pregunta) or _plan_en_historial(historial)
    pregunta_malla = bool(plan) and bool(
        re.search(r"malla|plan|semestre|cr[eé]ditos|materias", pregunta, re.I)
    )
    consulta = f"{pregunta} Plan {plan}" if pregunta_malla else pregunta
    fragmentos = recuperar(consulta, historial=historial, claves=claves)
    reseñas = _opiniones_de_claves(claves, chunks) if pide_recomendacion and claves else ""

    if fragmentos and es_solo_lista_de_profesores(pregunta):
        directa = _respuesta_directa(fragmentos)
        if directa:
            return {"respuesta": directa, "fuentes": _fuentes(fragmentos)}

    if (
        pide_recomendacion
        and claves
        and not reseñas
        and not any(c["tipo"] in ("mixto", "apoyo") for c in fragmentos)
    ):
        return {
            "respuesta": _respuesta_sin_opiniones(fragmentos),
            "fuentes": _fuentes(fragmentos),
        }

    partes = []
    if pregunta_malla:
        partes.append(
            f"DATO CONFIRMADO: el estudiante cursa el Plan {plan}. "
            "NO vuelvas a preguntar el año de ingreso ni el plan; responde directo."
        )
    historial_texto = _historial_contexto(historial)
    if historial_texto:
        partes.append(historial_texto)
    partes.append(f"CONTEXTO:\n{_contexto(fragmentos)}")
    if reseñas:
        partes.append("RESEÑAS DE ALUMNOS (texto completo del corpus local):\n" + reseñas)
    partes.append(f"PREGUNTA ACTUAL: {pregunta}")

    cliente = _cliente(CHAT_LOCATION)
    contenido = cliente.models.generate_content(
        model=CHAT_MODEL,
        contents="\n\n".join(partes),
        config=types.GenerateContentConfig(
            system_instruction=SISTEMA,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    texto = _limpiar_respuesta(_texto_visible(contenido))
    return {"respuesta": texto, "fuentes": _fuentes(fragmentos)}
