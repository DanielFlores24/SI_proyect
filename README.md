# Chat de IA — Ingeniería en Robótica (CUCEI, UdeG)

Chat con IA sobre la carrera, profesores, malla, trámites y bibliografía sugerida. Usa Gemini vía Vertex AI (GCP) y un índice semántico local con trazabilidad de fuentes.

## Estructura

```
app/
  main.py                 FastAPI: /health, /api/ask, /
  rag.py                  recuperación semántica + prompt con citas
  templates/index.html    interfaz de chat
  static/                 CSS y JS del front
data/
  dataIA/                 corpus crudo: mallas, profesores y 50 documentos de fuentes
  index/                  chunks.json + embeddings.npy + meta.json
  docs/                   PDFs fuente
scripts/
  scrape_profesores.py    SIIAU + PDF + MisProfesores
  scrape_fuentes_extra.py sitio oficial, catálogo, académicos, reseñas, libros
  build_index.py          convierte el corpus en el índice de búsqueda
  check_gcp.py            comprueba credenciales y modelos de Vertex AI
```

## Puesta en marcha local

```powershell
pip install -r requirements.txt
python scripts/check_gcp.py
python scripts/build_index.py --dry-run   # opcional: ver estadísticas
python scripts/build_index.py             # regenera data/index/
python scripts/test_rag.py                # 26 preguntas de prueba (usa la API)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abre http://127.0.0.1:8000. El puerto 8080 suele estar bloqueado en Windows.

## Autenticación

Local se usan credenciales ADC con impersonación de la cuenta de servicio (la organización prohíbe llaves JSON):

```powershell
gcloud auth application-default login --impersonate-service-account=inro-chat-si@project-75994596-89bd-4024-a24.iam.gserviceaccount.com
```

En Cloud Run **no** se usa llave: se adjunta la cuenta de servicio al despliegue.

### Acceso al chat y freno al gasto

`/` y `/api/ask` están protegidos por una contraseña configurada en `CHAT_PASSWORD` (en `.env` local y en Secret Manager en Cloud Run). Al entrar se guarda una cookie firmada con `CHAT_SECRET` que incluye cuántas preguntas se han usado: no hay sesión en servidor ni base de datos, así que funciona aunque Cloud Run cambie de instancia.

- `SESSION_HOURS=8`: duración de la sesión.
- `MAX_PREGUNTAS=100`: al llegar al límite, `/api/ask` responde `429` y pide volver a entrar.
- `COOKIE_SECURE=0` en local; `1` en Cloud Run (HTTPS).
- Si `CHAT_PASSWORD` o `CHAT_SECRET` están vacíos, el chat queda abierto solo para desarrollo. En Cloud Run **deben** estar definidos.

`/health` y `/static/*` siguen públicos; el primero solo reporta estado y si la protección está activa.

El proyecto usa dos regiones porque los modelos no están en las mismas:

- embeddings (`gemini-embedding-001`, 768 dims): `us-central1`
- generación (`gemini-3.5-flash`): `global`

## Regenerar el corpus

```powershell
python scripts/scrape_profesores.py
python scripts/scrape_fuentes_extra.py
python scripts/build_index.py
```

## Convenciones de datos

Cada documento lleva `fuente`, `url`, `fecha_captura` y `tipo`:

- `oficial`: UdeG / SIIAU / sitio de la carrera
- `apoyo`: MisProfesores, Open Library (bibliografía **no oficial**)
- `mixto`: SIIAU + opiniones de estudiantes

Además, `build_index.py` crea **un fragmento limpio por materia** desde `profesores_inro.json`. Así, preguntas como "¿quién da Métodos Matemáticos III?" no dependen de una tabla gigante: `rag.py` detecta la clave (`I9885`) o el nombre con número romano/árabe y pone ese fragmento primero.

El prompt ya **no pide citas en el texto**: las respuestas suenan como las diría un compañero y no muestran `[1]` ni `(1)`. La trazabilidad sigue en el campo `fuentes` de `/api/ask`. Si una respuesta depende de opiniones, lo dice de forma natural; si no hay info, sugiere preguntar a la coordinación.

Cuando la pregunta es solo "¿quién da X?", `rag.py` responde directamente desde el bloque estructurado: **solo los nombres**, sin calificaciones ni materias extra. La interfaz no muestra las fuentes; el campo `fuentes` sigue disponible en la respuesta JSON de `/api/ask` para trazabilidad.

### Contexto y recomendaciones

El front envía hasta 16 mensajes (8 turnos) en `historial`. El backend los usa para entender respuestas cortas como "¿y quién la da?", heredar la materia aunque se haya dicho varias preguntas atrás y resolver "los dos" o "ellos" por el tema de la conversación. Es memoria de conversación enviada por el cliente, no sesión en el servidor, así que funciona igual en Cloud Run.

Si piden una recomendación con un nombre que no existe literalmente (por ejemplo "precalculo"), se ofrecen los análogos reales del catálogo: Métodos Matemáticos I/II/III. Si no hay ninguna opinión de un profesor, se sugiere preguntar en CUCEI Social (Facebook). "Te recomiendo leer..." se trata como opinión del estudiante, no como pedido. Si dicen "robótica" a secas, se habla de la carrera completa (Control, Robótica Móvil, Sistemas Inteligentes, Embebidos, Visión Robótica), no solo de Robótica Móvil.

El año de ingreso y el plan se guardan en el `historial`: decir "2025" o "ingresé en 2025" fija el Plan 2018 y no se vuelve a preguntar. Las referencias ("cada uno de ellos", "de los de control", "eso no es cierto") se resuelven con toda la conversación reciente y con el tema de la pregunta, no con una lista fija de cuatro profesores. Saludos, identidad del modelo, conceptos generales de control y operaciones simples se responden sin llamar al modelo.

Las respuestas van en texto plano: la interfaz no interpreta markdown, así que el prompt prohíbe negritas con `**` y títulos con `#`. Para enumerar, se usan guiones o números.

## Cloud Run

Los valores no secretos salen de `.env`. Genera `.env.yaml` (ignorado por Git) con:

```powershell
python scripts/make_env_yaml.py
gcloud services enable cloudbuild.googleapis.com secretmanager.googleapis.com
printf "w9zX..." | gcloud secrets create inro-chat-password --data-file=-
printf "..." | gcloud secrets create inro-chat-secret --data-file=-

gcloud run deploy chat-inro `
  --source . `
  --region us-central1 `
  --service-account inro-chat-si@project-75994596-89bd-4024-a24.iam.gserviceaccount.com `
  --env-vars-file .env.yaml `
  --set-secrets "CHAT_PASSWORD=inro-chat-password:latest,CHAT_SECRET=inro-chat-secret:latest" `
  --allow-unauthenticated `
  --port 8080 `
  --memory 512Mi `
  --timeout 60
```

`--allow-unauthenticated` es necesario para que se vea la página de login; el acceso al chat lo controla la contraseña de la app, no IAM de Cloud Run.

Servicio desplegado: https://chat-inro-63003863631.us-central1.run.app
