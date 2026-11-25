# Chat de IA — Ingeniería en Robótica (CUCEI, UdeG)

Proyecto escolar: un chat con IA que responde preguntas sobre la carrera, profesores, malla curricular y trámites, usando un corpus local con trazabilidad de fuentes.

## Estructura

```
app/                    Aplicación (main.py, rag.py, templates, static)
data/
  dataIA/
    mallas.md                    Mallas 2018 y 2025 (resumen curricular)
    profesores_inro.json         Profesores consolidados (SIIAU + PDF + MisProfes)
    profesores_inro.md           Reporte legible de profesores
    contexto_extra.json          Todas las fuentes adicionales consolidadas
    fuentes/
      INDICE.md                  Resumen de fuentes, fechas y cobertura
      sitio_oficial/             ~46 páginas del sitio oficial de la carrera
      catalogo_materias.md       Créditos, horas y prerrequisitos (SIIAU)
      perfiles_academicos.md     Perfiles SNII/PRODEP del CUCEI
      resenas_misprofes.md       Reseñas y etiquetas de estudiantes
      bibliografia_sugerida.md   Libros por materia (NO oficial)
  docs/                   PDFs fuente (mallas, recomendaciones de maestros)
scripts/
  scrape_profesores.py    Profesores: SIIAU público + PDF + MisProfesores
  scrape_fuentes_extra.py Fuentes adicionales (5 fuentes)
```

## Regenerar los datos

```powershell
pip install -r requirements.txt
python scripts/scrape_profesores.py
python scripts/scrape_fuentes_extra.py
```

`scrape_fuentes_extra.py` necesita que `profesores_inro.json` ya exista.

## Convenciones de los documentos

Todo archivo en `data/dataIA/fuentes/` lleva un encabezado YAML con:

- `titulo`, `fuente`, `url`
- `fecha_captura`: fecha de obtención (para detectar información caducada)
- `tipo`: `oficial` (UdeG/SIIAU) o `apoyo` (MisProfesores, Open Library)

Esto permite que el RAG priorice fuentes oficiales y advierta cuando una respuesta provenga de opiniones o sugerencias.

## Fuentes y precauciones

| Fuente | Tipo | Uso |
| --- | --- | --- |
| Sitio oficial de la carrera y SIIAU | oficial | Mallas, créditos, prerrequisitos, trámites, laboratorios |
| Académicos CUCEI (SNII/PRODEP) | oficial | Líneas de investigación y perfiles docentes |
| PDF "Acción U 2026A: INRO" | apoyo | Reseñas de maestros por materia |
| MisProfesores.com | apoyo | Calificaciones y comentarios de estudiantes |
| Open Library | apoyo (NO oficial) | Bibliografía **sugerida**; no sustituye el programa de la asignatura |

Los datos de MisProfesores se emparejaron por nombre, por lo que pueden existir coincidencias erróneas; cada entrada incluye el URL original para verificarla.

## Pendiente

- Integrar `data/dataIA/` en `app/rag.py` para indexar los documentos con sus metadatos.
- Definir comportamiento del chat ante preguntas sin fuente oficial (responder "no tengo información oficial" en lugar de inventar).
