const stream = document.querySelector('#stream');
const loading = document.querySelector('#loading');
const error = document.querySelector('#error');
const play = document.querySelector('#play');
const reset = document.querySelector('#reset');
const camera = document.querySelector('#camera');
const menu = document.querySelector('#camera-menu');
const connectionDot = document.querySelector('#connection-dot');
const connectionLabel = document.querySelector('#connection-label');
let paused = true;

async function request(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function control(action, value) {
  const state = await request('/api/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, value }),
  });
  applyState(state);
}

function applyState(state) {
  paused = state.paused;
  play.dataset.playing = String(!paused);
  play.setAttribute('aria-label', paused ? 'Play simulation' : 'Pause simulation');
  document.querySelector('#model').textContent = state.model.replace(' / unitree_rl_mjlab', '');
  document.querySelector('#sim-time').textContent = `${Number(state.sim_time).toFixed(3)} s`;
  document.querySelector('#fps').textContent = `${Number(state.fps).toFixed(1)} fps`;
  connectionDot.dataset.connected = 'true';
  connectionLabel.textContent = paused ? 'PAUSED' : 'RUNNING';
  error.hidden = true;
}

async function refresh() {
  try {
    applyState(await request('/api/state'));
  } catch (_) {
    connectionDot.dataset.connected = 'false';
    connectionLabel.textContent = 'OFFLINE';
    error.hidden = false;
  }
}

stream.addEventListener('load', () => {
  loading.hidden = true;
  stream.dataset.ready = 'true';
});
stream.addEventListener('error', () => {
  loading.hidden = true;
  error.hidden = false;
});

play.addEventListener('click', () => control('pause', !paused));
reset.addEventListener('click', () => control('reset'));

function closeMenu() {
  camera.setAttribute('aria-expanded', 'false');
  menu.hidden = true;
}

camera.addEventListener('click', () => {
  const open = camera.getAttribute('aria-expanded') === 'true';
  camera.setAttribute('aria-expanded', String(!open));
  menu.hidden = open;
});

menu.addEventListener('click', async (event) => {
  const item = event.target.closest('[data-camera]');
  if (!item) return;
  menu.querySelectorAll('[aria-current]').forEach((node) => node.removeAttribute('aria-current'));
  item.setAttribute('aria-current', 'true');
  camera.querySelector('span').textContent = item.textContent;
  closeMenu();
  await control('camera', item.dataset.camera);
});

document.addEventListener('click', (event) => {
  if (!event.target.closest('.camera-control')) closeMenu();
});

document.addEventListener('keydown', (event) => {
  if (event.code === 'Space' && event.target === document.body) {
    event.preventDefault();
    control('pause', !paused);
  }
  if (event.code === 'Escape') closeMenu();
});

refresh();
setInterval(refresh, 1000);
