const chat = document.getElementById('chat');
const form = document.getElementById('formulario');
const input = document.getElementById('pregunta');
const boton = form.querySelector('button');
let historial = [];
// Guardamos suficiente conversación para resolver "ellos", "los dos", etc.,
// aunque la materia se haya dicho varias preguntas atrás.
const MAX_MENSAJES = 16;

function mensaje(texto, clase) {
  const nodo = document.createElement('div');
  nodo.className = `mensaje ${clase}`;
  nodo.textContent = texto;
  chat.appendChild(nodo);
  return nodo;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const texto = input.value.trim();
  if (!texto) return;

  mensaje(texto, 'usuario');
  input.value = '';
  boton.disabled = true;
  const espera = mensaje('Pensando…', 'bot');

  try {
    const r = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pregunta: texto, historial: historial.slice(-MAX_MENSAJES) })
    });
    if (!r.ok) {
      if (r.status === 401) {
        location.href = '/login';
        return;
      }
      let detalle = `Error ${r.status}`;
      try {
        const cuerpo = await r.json();
        detalle = cuerpo.detail || detalle;
      } catch (_) {
        detalle = (await r.text()) || detalle;
      }
      espera.remove();
      mensaje(detalle, 'bot');
      return;
    }

    const datos = await r.json();
    const respuesta = (datos.respuesta || '').trim();
    espera.remove();
    if (!respuesta) {
      mensaje('No me salió una respuesta, inténtalo otra vez.', 'bot');
      return;
    }
    mensaje(respuesta, 'bot');
    historial.push({ pregunta: texto, respuesta });
    historial = historial.slice(-MAX_MENSAJES);
  } catch (err) {
    espera.remove();
    mensaje(`Error: ${err.message}`, 'bot');
  } finally {
    boton.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }
});
