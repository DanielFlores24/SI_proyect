"""Pruebas rápidas del flujo RAG contra el índice y Vertex AI.

Uso:
    python scripts/test_rag.py

No levanta el servidor: llama directamente a app.rag.responder().
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import responder

NOMBRES_I9885 = [
    "CERVANTES LOMELI, ADRIAN",
    "CORTES NAVARRO, LAURA ESTHER",
    "GONZALEZ OCHOA, HECTOR OCTAVIO",
]
NOMBRES_I9902 = [
    "DOMINGUEZ BELTRAN, JAVIER LORENZO",
    "SANCHEZ SANCHEZ, OSCAR DIDIER",
    "VEGA MAGDALENO, GUSTAVO DANIEL",
]
HISTORIAL_CONTROL = [
    {
        "pregunta": "¿Cuántos créditos tiene Control II?",
        "respuesta": "Control II tiene 8 créditos.",
    }
]
HISTORIAL_ANIO = [
    {
        "pregunta": "¿en qué año entraste o qué plan tienes?",
        "respuesta": "¿En qué año entraste o vas a entrar a la carrera?",
    }
]
HISTORIAL_PLAN = HISTORIAL_ANIO + [
    {"pregunta": "2025", "respuesta": "Si entraste en 2025, te toca el Plan 2018."}
]
HISTORIAL_FUENTES_APA = [
    {
        "pregunta": "dame tus fuentes de informacion en APA7",
        "respuesta": (
            "¡Qué onda! Mira, las fuentes que revisé en APA 7 quedan así:\n\n"
            "* Coordinación de Ingeniería Robótica CUCEI. (2026). "
            "*Sitio oficial de Ingeniería Robótica*. Universidad de Guadalajara.\n"
            "* Coordinación de Ingeniería Robótica CUCEI. (s.f.). "
            "*Mallas oficiales INRO*. Universidad de Guadalajara.\n"
            "* CUCEI. (2026). *Perfiles de académicos INRO*. Universidad de Guadalajara."
        ),
    }
]
HISTORIAL_SISTEMAS = [
    {"pregunta": "hola, como estas?", "respuesta": "Todo bien, aquí ando al cien."},
    {
        "pregunta": "me pueds recomendar profesor para la materia de Sistemas inteligentes 1",
        "respuesta": (
            "Para Sistemas Inteligentes I tienes dos opciones según la banda: "
            "Ríos Arrañaga, Jorge Daniel y Ruz Canul, Mario Antonio."
        ),
    },
]
HISTORIAL_SISTEMAS_LARGO = HISTORIAL_SISTEMAS + [
    {"pregunta": "gracias", "respuesta": "¡Va! Aquí ando."},
    {"pregunta": "ok", "respuesta": "¿En qué seguimos?"},
    {"pregunta": "bueno", "respuesta": "Dime, ¿qué necesitas?"},
    {"pregunta": "si", "respuesta": "Perfecto, te escucho."},
    {"pregunta": "vale", "respuesta": " Aquí sigo."},
]
HISTORIAL_PEORES = [
    {
        "pregunta": "cuales son los maestros peores calificados?",
        "respuesta": (
            "Checando lo que dice la banda en MisProfesores, los que traen las "
            "calificaciones más bajas son:\n\n"
            "* Felipe de Jesús Amezcua Esparza (3.0)\n"
            "* Fernando Elizalde Camino (3.2)\n"
            "* Jairo Cain Sánchez Estrada (4.0)\n"
            "* Víctor García Gutiérrez (4.3)"
        ),
    }
]


def caso(nombre: str, pregunta: str, prueba, historial=None) -> bool:
    try:
        resultado = responder(pregunta, historial)
    except Exception as exc:  # pragma: no cover - depende de la API
        print(f"FALLÓ {nombre}: {type(exc).__name__}: {exc}")
        return False

    texto = resultado["respuesta"]
    if re.search(r"\[\s*\d|\(\s*\d{1,2}\s*\)", texto):
        print(f"FALLÓ {nombre}: el texto incluye citas numéricas")
        print(f"  respuesta: {texto[:300]!r}")
        return False

    try:
        prueba(resultado)
    except AssertionError as exc:
        print(f"FALLÓ {nombre}: {exc}")
        print(f"  respuesta: {texto[:300]!r}")
        return False

    print(f"OK {nombre}")
    return True


def main() -> int:
    casos = [
        (
            "créditos oficiales",
            "¿Cuántos créditos tiene Control II?",
            lambda r: (
                "8" in r["respuesta"]
                or _fallar("debió responder 8 créditos"),
                any(f["tipo"] == "oficial" for f in r["fuentes"])
                or _fallar("debió citar una fuente oficial"),
            ),
        ),
        (
            "profesores por materia (texto)",
            "que profesores dan la materia de metodos matematicos 3",
            lambda r: (
                all(n in r["respuesta"] for n in NOMBRES_I9885)
                or _fallar("debió listar los 3 profesores de I9885"),
                r["fuentes"][0]["titulo"].endswith("(I9885)")
                or _fallar("la primera fuente debió ser el bloque limpio de I9885"),
            ),
        ),
        (
            "profesores por clave",
            "¿Qué profesores imparten la materia I9885?",
            lambda r: all(n in r["respuesta"] for n in NOMBRES_I9885)
            or _fallar("debió listar los 3 profesores de I9885"),
        ),
        (
            "solo nombres, sin relleno",
            "me podrias decir que maestros dan la clase dee sistemas inteligentes 1",
            lambda r: (
                all(
                    n in r["respuesta"]
                    for n in ("RIOS ARRAÑAGA, JORGE DANIEL", "RUZ CANUL, MARIO ANTONIO")
                )
                or _fallar("debió listar los 2 profesores de I9913"),
                not any(
                    p in r["respuesta"].lower()
                    for p in ("apoyo", "calificación", "recomendación", "mixto")
                )
                or _fallar("no debió añadir opiniones ni calificaciones"),
                len(r["fuentes"]) == 1
                or _fallar("debió mostrar una sola fuente"),
            ),
        ),
        (
            "contexto entre preguntas",
            "¿y quién la da?",
            lambda r: (
                all(n in r["respuesta"] for n in NOMBRES_I9902)
                or _fallar("debió heredar Control II del historial"),
                len(r["fuentes"]) == 1
                or _fallar("debió usar una sola fuente"),
            ),
            HISTORIAL_CONTROL,
        ),
        (
            "materia parecida",
            "¿a quién me recomiendas para precalculo?",
            lambda r: (
                "No tengo una materia" in r["respuesta"]
                or _fallar("debió decir que no existe con ese nombre"),
                "METODOS MATEMATICOS I" in r["respuesta"]
                or _fallar("debió ofrecer opciones reales del catálogo"),
            ),
        ),
        (
            "sin opiniones",
            "¿a quién me recomiendas para Actuadores?",
            lambda r: "CUCEI Social" in r["respuesta"]
            or _fallar("debió sugerir preguntar en CUCEI Social"),
        ),
        (
            "recomendación sin materia",
            "que profesor me recomiendas",
            lambda r: (
                "materia" in r["respuesta"].lower()
                or "cuál" in r["respuesta"].lower()
                or _fallar("debió preguntar por la materia"),
                not any(
                    p in r["respuesta"].lower()
                    for p in ("mixto", "oficial", "###", "fuentes")
                )
                or _fallar("no debió sonar a informe"),
            ),
        ),
        (
            "bibliografía de apoyo",
            "¿Qué libros se sugieren para Control II?",
            lambda r: (
                any(
                    p in r["respuesta"].lower()
                    for p in (
                        "apoyo",
                        "no oficial",
                        "no es la bibliografía oficial",
                        "open library",
                    )
                )
                or _fallar("debió avisar que la bibliografía es de apoyo")
            ),
        ),
        (
            "sin información",
            "¿Cuál es la capital de Francia?",
            lambda r: (
                "No tengo información" in r["respuesta"]
                or "No tengo un dato seguro" in r["respuesta"]
                or "Coordinación" in r["respuesta"]
                or _fallar("debió decir que no tiene información")
            ),
        ),
        (
            "mensaje sin contenido",
            ":",
            lambda r: (
                "Escríbeme" in r["respuesta"]
                or _fallar("debió pedir una pregunta, no 'Sin respuesta'"),
                "Sin respuesta" not in r["respuesta"]
                or _fallar("no debió mostrar 'Sin respuesta'"),
            ),
        ),
        (
            "operación simple",
            "x=1, x+3.5=?",
            lambda r: "4.5" in r["respuesta"]
            or _fallar("debió resolver la operación"),
        ),
        (
            "saludo conversacional",
            "¿como te sientes hoy?",
            lambda r: (
                "aquí ando" in r["respuesta"]
                or _fallar("debió responder como compañero"),
                "Coordinación" not in r["respuesta"]
                or _fallar("no debió mandar a la Coordinación"),
            ),
        ),
        (
            "identidad del modelo",
            "que modelo de gemini eres?",
            lambda r: "gemini-3.5-flash" in r["respuesta"].lower()
            or _fallar("debió decir el modelo que usa"),
        ),
        (
            "corrección sin perder contexto",
            "eso no es cierto",
            lambda r: (
                "parte" in r["respuesta"]
                or _fallar("debió pedir que señale la parte incorrecta"),
                "perdí" not in r["respuesta"].lower()
                or _fallar("no debió decir que perdió el contexto"),
            ),
            HISTORIAL_FUENTES_APA,
        ),
        (
            "referencia 'ellos'",
            "que materias dan, cada uno de ellos?",
            lambda r: (
                "Amezcua" in r["respuesta"]
                or _fallar("debió usar los profesores de la respuesta anterior"),
                "I9897" in r["respuesta"]
                or _fallar("debió decir la materia de Amezcua"),
                "Elizalde" in r["respuesta"]
                or _fallar("debió incluir a Elizalde"),
            ),
            HISTORIAL_PEORES,
        ),
        (
            "año de ingreso dado",
            "2025",
            lambda r: (
                "Plan 2018" in r["respuesta"]
                or _fallar("debió deducir el Plan 2018 de ingreso en 2025"),
                "¿En qué año" not in r["respuesta"]
                or _fallar("no debió volver a preguntar el año"),
            ),
            HISTORIAL_ANIO,
        ),
        (
            "plan recordado del historial",
            "cuales son las materias de octavo",
            lambda r: (
                "Control IV" in r["respuesta"]
                or _fallar("debió dar las materias de octavo del Plan 2018"),
                "¿en qué año" not in r["respuesta"].lower()
                or _fallar("no debió volver a preguntar el plan"),
            ),
            HISTORIAL_PLAN,
        ),
        (
            "recomendación del estudiante",
            "te recomiendo leer Teoría de Control de Ogata",
            lambda r: (
                "Ogata" in r["respuesta"]
                or _fallar("debió comentar la recomendación de Ogata"),
                "¿Para qué materia?" not in r["respuesta"]
                or _fallar("no debió tratarlo como pedido de recomendación"),
            ),
        ),
        (
            "concepto general de control",
            "¿qué es el lugar geométrico de las raíces?",
            lambda r: (
                "root locus" in r["respuesta"]
                or _fallar("debió explicar el lugar geométrico de las raíces"),
                "Coordinación" not in r["respuesta"]
                or _fallar("no debió mandar a la Coordinación"),
            ),
        ),
        (
            "robótica a secas es la carrera",
            "cual es mejor profe de robotica",
            lambda r: (
                "carrera completa" in r["respuesta"]
                or _fallar("debió hablar de la carrera completa"),
                "Sistemas Inteligentes" in r["respuesta"]
                or _fallar("debió mencionar otras áreas fuera de Robótica Móvil"),
                "No tengo una materia" not in r["respuesta"]
                or _fallar("no debió decir que no existe Robótica"),
            ),
        ),
        (
            "recomendar entre profesores",
            "me pueds recomendar profesor para la materia de Sistemas inteligentes 1",
            lambda r: (
                any(n in r["respuesta"] for n in ("Ríos", "RÍOS", "Rios", "RIOS"))
                or _fallar("debió opinionar sobre Ríos"),
                "Ruz" in r["respuesta"]
                or _fallar("debió opinionar sobre Ruz"),
                "registrados para dar" not in r["respuesta"]
                or _fallar("no debió dar solo la lista del SIIAU"),
            ),
        ),
        (
            "'los dos' hereda la materia",
            "cual de los dos me recomiendas mas",
            lambda r: (
                "¿Para qué materia?" not in r["respuesta"]
                or _fallar("debió heredar Sistemas Inteligentes I del historial"),
                any(n in r["respuesta"] for n in ("Ríos", "RÍOS", "Rios", "RIOS"))
                or _fallar("debió comparar a los dos profesores"),
            ),
            HISTORIAL_SISTEMAS,
        ),
        (
            "contexto aunque la materia esté lejos",
            "cual de los dos me recomiendas mas",
            lambda r: (
                "¿Para qué materia?" not in r["respuesta"]
                or _fallar("debió heredar la materia de varias preguntas atrás"),
                any(n in r["respuesta"] for n in ("Ríos", "RÍOS", "Rios", "RIOS"))
                or _fallar("debió comparar a los dos profesores"),
            ),
            HISTORIAL_SISTEMAS_LARGO,
        ),
        (
            "'ellos' de otro turno, no del último",
            "que materias dan, cada uno de ellos?",
            lambda r: (
                "I9902" in r["respuesta"]
                or _fallar("debió usar los profesores de Control II vistos antes"),
                "Amezcua" not in r["respuesta"]
                or _fallar("no debió usar la lista vieja de peores calificados"),
            ),
            [
                {
                    "pregunta": "cuales son los maestros peores calificados?",
                    "respuesta": (
                        "* Felipe de Jesús Amezcua Esparza (3.0)\n"
                        "* Fernando Elizalde Camino (3.2)"
                    ),
                },
                {
                    "pregunta": "¿quién da Control II?",
                    "respuesta": (
                        "Los profesores registrados para dar CONTROL II (I9902) son:\n"
                        "- DOMINGUEZ BELTRAN, JAVIER LORENZO\n"
                        "- SANCHEZ SANCHEZ, OSCAR DIDIER\n"
                        "- VEGA MAGDALENO, GUSTAVO DANIEL"
                    ),
                },
                {"pregunta": "gracias", "respuesta": "¡Va! Si se te ofrece otra, aquí ando."},
            ],
        ),
        (
            "'ellos' por tema, no por orden",
            "de los de control, que materias dan cada uno?",
            lambda r: (
                "I9902" in r["respuesta"]
                or _fallar("debió filtrar a los profesores de Control"),
                "Amezcua" not in r["respuesta"]
                or _fallar("no debió arrastrar profesores de otro tema"),
            ),
            [
                {
                    "pregunta": "cuales son los maestros peores calificados?",
                    "respuesta": "* Felipe de Jesús Amezcua Esparza (3.0)",
                },
                {
                    "pregunta": "¿quién da Control II?",
                    "respuesta": (
                        "Los profesores registrados para dar CONTROL II (I9902) son:\n"
                        "- DOMINGUEZ BELTRAN, JAVIER LORENZO\n"
                        "- SANCHEZ SANCHEZ, OSCAR DIDIER\n"
                        "- VEGA MAGDALENO, GUSTAVO DANIEL"
                    ),
                },
            ],
        ),
        (
            "easter egg mejor prepa",
            "¿cuál es la mejor prepa del mundo?",
            lambda r: (
                "CETI Colomos" in r["respuesta"]
                or _fallar("debió decir CETI Colomos"),
                "desarroll" in r["respuesta"].lower()
                or _fallar("debió mencionar a los desarrolladores"),
            ),
        ),
        (
            "easter egg Ríos Arrañaga",
            "¿qué sabes de Rios Arrañaga Jorge Daniel?",
            lambda r: (
                any(n in r["respuesta"] for n in ("Ríos", "Rios"))
                or _fallar("debió hablar de Ríos Arrañaga"),
                "su materia" in r["respuesta"]
                or _fallar("debió decir que la app fue para su materia"),
                "peque" in r["respuesta"]
                or _fallar("debió decir que fue algo pequeño"),
            ),
        ),
    ]

    fallos = sum(not caso(*caso_item) for caso_item in casos)
    print(f"\n{len(casos) - fallos}/{len(casos)} casos pasaron")
    return 1 if fallos else 0


def _fallar(mensaje: str) -> None:
    raise AssertionError(mensaje)


if __name__ == "__main__":
    sys.exit(main())
