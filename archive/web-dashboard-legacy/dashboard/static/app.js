import * as THREE from 'three';
import { OrbitControls } from '/vendor/three/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from '/vendor/three/examples/jsm/loaders/STLLoader.js';
import { TerrainRenderer } from '/terrain-renderer.js';
import { SnowDebugControls } from '/debug-controls.js';
import { WeatherEffects } from '/weather-effects.js';

const viewport = document.querySelector('#viewport');
const loading = document.querySelector('#loading');
const error = document.querySelector('#error');
const play = document.querySelector('#play');
const reset = document.querySelector('#reset');
const homeCamera = document.querySelector('#home-camera');
const stand = document.querySelector('#stand');
const connectionDot = document.querySelector('#connection-dot');
const connectionLabel = document.querySelector('#connection-label');
const terrainMode = document.querySelector('#terrain-mode');
const terrainDetail = document.querySelector('#terrain-detail');
const debugPanel = document.querySelector('#debug-panel');
const debugOpen = document.querySelector('#debug-open');
const weatherReadout = document.querySelector('#weather');
const weatherDetail = document.querySelector('#weather-detail');
const snowPhysicsNote = document.querySelector('#snow-physics-note');

// MuJoCo, Newton telemetry, Unitree STL assets, and the local path preview
// are all expressed in metres. Camera/weather tuning must never be fixed
// by silently scaling either the robot or terrain.
const METRES_PER_SCENE_UNIT = 1;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x121417);
scene.fog = new THREE.Fog(0x121417, 24, 90);
scene.userData.metresPerSceneUnit = METRES_PER_SCENE_UNIT;

const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 140);
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.18;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 1.25;
controls.maxDistance = 52;
controls.maxPolarAngle = Math.PI * 0.49;
controls.target.set(0, 0.9, 0);

function resetCamera() {
  camera.position.set(5.6, 3.5, 8.4);
  controls.target.set(0, 0.95, -3.8);
  controls.update();
}
resetCamera();

const robotRoot = new THREE.Group();
robotRoot.rotation.x = -Math.PI / 2;
robotRoot.scale.setScalar(METRES_PER_SCENE_UNIT);
robotRoot.userData.units = 'metres';
scene.add(robotRoot);

const hemi = new THREE.HemisphereLight(0xdde6ff, 0x252a31, 2.05);
scene.add(hemi);
const key = new THREE.DirectionalLight(0xf3f6ff, 3.1);
key.position.set(2.8, 4.5, 3.2);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 0.1;
key.shadow.camera.far = 35;
key.shadow.camera.left = -12;
key.shadow.camera.right = 12;
key.shadow.camera.top = 16;
key.shadow.camera.bottom = -8;
scene.add(key);
const rim = new THREE.DirectionalLight(0x728cff, 1.7);
rim.position.set(-3, 2.4, -2.5);
scene.add(rim);

const weatherEffects = new WeatherEffects(scene);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(7, 96),
  new THREE.MeshStandardMaterial({ color: 0x1a1d22, roughness: 0.93, metalness: 0.03 }),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.045;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(12, 48, 0x3f4652, 0x252a31);
grid.position.y = -0.035;
grid.material.opacity = 0.18;
grid.material.transparent = true;
scene.add(grid);

const terrainRenderer = new TerrainRenderer(scene, {
  onStatus(status) {
    const live = status.mode === 'live';
    const surface = status.surface === 'ice' ? 'ICE PACK' : 'SNOW PACK';
    terrainMode.textContent = live ? `LIVE · ${surface}` : (status.mode === 'path' ? 'PATH DEBUG' : surface);
    terrainMode.dataset.live = String(live);
    const [width, depth] = status.resolution || [0, 0];
    const particles = status.particleCount ? ` · ${status.particleCount.toLocaleString()} particles` : '';
    const packDepth = Number(status.surfaceDepth || 0);
    const thickness = packDepth ? ` · ${Math.round(packDepth * 100)} cm` : '';
    const source = status.mode === 'path'
      ? `${status.path?.width_m || 8}×${status.path?.length_m || 28} m path`
      : (status.mode === 'everest' ? 'Everest map' : `${width}×${depth}`);
    terrainDetail.textContent = `${source}${thickness}${particles}`;
  },
});
terrainRenderer.root.scale.setScalar(METRES_PER_SCENE_UNIT);
terrainRenderer.root.userData.units = 'metres';
const debugControls = new SnowDebugControls(debugPanel, {
  onChange(parameters) {
    if (everestTile) terrainRenderer.loadPathPreview(everestTile, { surface: 'snow', depth: parameters.layers[0].thickness_m }, parameters);
    weatherEffects.setEnvironment(parameters);
    scheduleSnowPhysics(parameters);
  },
});
window.__everestViewer = {
  scene,
  camera,
  renderer,
  controls,
  terrainRenderer,
  debugControls,
  weatherEffects,
  scale: { metresPerSceneUnit: METRES_PER_SCENE_UNIT },
};

const bodyGroups = new Map();
let previousFrame = null;
let currentFrame = null;
let currentReceivedAt = 0;
let paused = true;
let pollTimer = null;
let connected = false;
let rateStartedAt = performance.now();
let rateStartedSequence = 0;
let measuredHz = 0;
let terrainTimer = null;
let terrainEndpointAvailable = true;
let terrainTelemetryUrl = null;
let everestTile = null;
let snowPhysicsTimer = null;
let snowPhysicsRevision = 0;

async function request(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  if (!response.ok) {
    const detail = (await response.text()).trim();
    throw new Error(`${response.status} ${detail || response.statusText}`);
  }
  return response.json();
}

function setConnection(isConnected) {
  connected = isConnected;
  connectionDot.dataset.connected = String(isConnected);
  connectionLabel.textContent = isConnected ? (paused ? 'PAUSED' : 'LIVE') : 'OFFLINE';
  error.hidden = isConnected;
}

function applyState(state) {
  if (state.telemetry_error) throw new Error(state.telemetry_error);
  paused = state.paused;
  play.dataset.playing = String(!paused);
  play.setAttribute('aria-label', paused ? 'Play simulation' : 'Pause simulation');
  document.querySelector('#engine-name').textContent = String(state.engine || 'mujoco').toUpperCase();
  document.querySelector('#policy').textContent = state.policy?.enabled
    ? `G1 velocity · ${Math.abs(Number(state.policy.command?.[0] || 0)) >= 0.1 ? 'walk' : 'stand'}`
    : 'disabled';
  if (state.simulation_fault) {
    error.textContent = `Simulation paused · ${state.simulation_fault}`;
    error.hidden = false;
  }
  document.querySelectorAll('[data-surface]').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.surface === state.surface));
  });
  const movementAllowed = state.weather_parameters?.movement_allowed !== false;
  play.disabled = !movementAllowed;
  play.title = movementAllowed ? 'Play / pause (Space)' : 'Movement disabled by weather risk gate';
  connectionLabel.textContent = paused ? 'PAUSED' : 'LIVE';
  applyWeather(state.weather, state.weather_parameters);
  const mpm = state.snow?.mpm;
  const active = state.snow?.mpm_active === true && mpm?.active === true;
  snowPhysicsNote.dataset.active = String(active);
  snowPhysicsNote.dataset.error = String(Boolean(mpm?.error));
  if (active) {
    const sink = Number(mpm.max_sinkage_m || 0) * 100;
    const mode = mpm.accumulation_enabled === false ? 'STATIC' : 'LIVE';
    snowPhysicsNote.querySelector('b').textContent = `${mode} · Newton MPM ${String(mpm.device || '').toUpperCase()} · sink ${sink.toFixed(1)} cm`;
  } else if (mpm?.error) {
    snowPhysicsNote.querySelector('b').textContent = `MPM unavailable · ${mpm.error}`;
  } else {
    snowPhysicsNote.querySelector('b').textContent = 'Newton MPM waiting for snow parameters';
  }
}

function scheduleSnowPhysics(parameters) {
  const revision = ++snowPhysicsRevision;
  const payload = JSON.parse(JSON.stringify(parameters));
  window.clearTimeout(snowPhysicsTimer);
  snowPhysicsTimer = window.setTimeout(async () => {
    snowPhysicsNote.dataset.active = 'false';
    snowPhysicsNote.dataset.error = 'false';
    snowPhysicsNote.querySelector('b').textContent = 'Rebuilding Newton MPM material…';
    try {
      const state = await request('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'snow_parameters', value: payload }),
      });
      if (revision === snowPhysicsRevision) applyState(state);
    } catch (reason) {
      if (revision !== snowPhysicsRevision) return;
      snowPhysicsNote.dataset.error = 'true';
      snowPhysicsNote.querySelector('b').textContent = `MPM update failed · ${reason.message}`;
    }
  }, 450);
}

function applyWeather(weather, parameters = {}) {
  if (!weather) {
    weatherReadout.textContent = 'NO FEED';
    weatherDetail.textContent = 'parameters idle';
    scene.fog.near = 24;
    scene.fog.far = 90;
    weatherEffects.setLiveWeather(null);
    return;
  }
  const conditions = weather.conditions || {};
  const risk = weather.risk || {};
  weatherReadout.textContent = String(risk.level || conditions.summary || 'ACTIVE');
  const temp = conditions.temperature_c == null ? '—' : `${Number(conditions.temperature_c).toFixed(1)}°C`;
  const wind = parameters.wind_force_n == null ? '—' : `${Number(parameters.wind_force_n).toFixed(0)}N wind`;
  weatherDetail.textContent = `${temp} · ${wind}`;
  // visibility_scale is normalized against kilometres of atmospheric range.
  // Mapping it directly to local scene metres turned a 280 m report into a
  // ~5.6 m fog wall, completely hiding a camera 10-16 m from the G1.
  const fallbackScale = Math.max(0.001, Number(parameters.visibility_scale ?? weather.simulation?.visibility_scale ?? 1));
  const visibilityMetres = Math.max(1, Number(conditions.visibility_m || fallbackScale * 10000));
  scene.fog.far = THREE.MathUtils.clamp(visibilityMetres * 0.28, 32, 110);
  scene.fog.near = THREE.MathUtils.clamp(scene.fog.far * 0.38, 12, 44);
  weatherEffects.setLiveWeather(weather);
}

async function control(action, value) {
  applyState(await request('/api/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, value }),
  }));
}

function quaternionFromWxyz(value) {
  return new THREE.Quaternion(value[1], value[2], value[3], value[0]);
}

async function loadRobot() {
  const manifest = await request('/api/scene');
  const tile = await request(manifest.terrain_tile?.url || '/everest-terrain.json');
  everestTile = tile;
  terrainRenderer.loadEverestTile(everestTile, manifest.terrain);
  terrainRenderer.loadPathPreview(everestTile, manifest.terrain, debugControls.parameters);
  floor.visible = false;
  grid.visible = false;
  terrainTelemetryUrl = manifest.terrain_stream?.url || manifest.terrain?.telemetry_url || null;
  const loader = new STLLoader();
  const geometryPromises = new Map();
  const geometryFor = (url) => {
    if (!geometryPromises.has(url)) {
      geometryPromises.set(url, loader.loadAsync(url).then((geometry) => {
        geometry.computeVertexNormals();
        return geometry;
      }));
    }
    return geometryPromises.get(url);
  };

  await Promise.all(manifest.visuals.map(async (visual) => {
    let group = bodyGroups.get(visual.body);
    if (!group) {
      group = new THREE.Group();
      group.name = visual.body;
      bodyGroups.set(visual.body, group);
      robotRoot.add(group);
    }
    const rgba = visual.rgba;
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      opacity: rgba[3],
      transparent: rgba[3] < 1,
      roughness: rgba[0] < 0.4 ? 0.48 : 0.34,
      metalness: rgba[0] < 0.4 ? 0.28 : 0.58,
    });
    const mesh = new THREE.Mesh((await geometryFor(visual.url)).clone(), material);
    mesh.name = visual.mesh;
    mesh.position.fromArray(visual.position);
    mesh.quaternion.copy(quaternionFromWxyz(visual.quaternion));
    mesh.scale.fromArray(visual.scale);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
  }));
}

function ingestFrame(frame) {
  if (frame.schema !== 'everest-viewer/v1') throw new Error(`Unsupported frame schema: ${frame.schema}`);
  if (currentFrame && frame.sequence <= currentFrame.sequence) return;
  const now = performance.now();
  if (!rateStartedSequence) {
    rateStartedSequence = frame.sequence;
    rateStartedAt = now;
  } else if (now - rateStartedAt >= 750) {
    measuredHz = (frame.sequence - rateStartedSequence) / ((now - rateStartedAt) / 1000);
    rateStartedSequence = frame.sequence;
    rateStartedAt = now;
  }
  previousFrame = frame.reset_frame ? frame : (currentFrame || frame);
  currentFrame = frame;
  currentReceivedAt = now;
  paused = frame.paused;
  play.dataset.playing = String(!paused);
  play.setAttribute('aria-label', paused ? 'Play simulation' : 'Pause simulation');
  document.querySelector('#sim-time').textContent = `${Number(frame.sim_time).toFixed(3)} s`;
  document.querySelector('#fps').textContent = `${measuredHz ? measuredHz.toFixed(1) : '—'} Hz`;
  setConnection(true);
  if (frame.terrain) terrainRenderer.ingest(frame.terrain);
  if (frame.simulation_fault) {
    error.textContent = `Simulation paused · ${frame.simulation_fault}`;
    error.hidden = false;
  }
  applyWeather(frame.weather, frame.weather_parameters);
}

async function pollFrame() {
  try {
    ingestFrame(await request('/api/frame'));
  } catch (_) {
    setConnection(false);
  } finally {
    pollTimer = window.setTimeout(pollFrame, 40);
  }
}

async function pollTerrain() {
  try {
    if (!terrainTelemetryUrl) return;
    const response = await fetch(terrainTelemetryUrl, { cache: 'no-store' });
    if (response.status === 404) {
      terrainEndpointAvailable = false;
      terrainTimer = window.setTimeout(() => {
        terrainTimer = null;
        pollTerrain();
      }, 1500);
      return;
    }
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    terrainEndpointAvailable = true;
    terrainRenderer.ingest(await response.json());
  } catch (_) {
    // Robot telemetry remains authoritative even when the optional Newton
    // terrain feed is starting, rebuilding, or unavailable.
    terrainEndpointAvailable = false;
  } finally {
    if (!terrainTimer) terrainTimer = window.setTimeout(() => {
      terrainTimer = null;
      pollTerrain();
    }, terrainEndpointAvailable ? 66 : 1500);
  }
}

function applyPose(now) {
  if (!currentFrame) return;
  const older = previousFrame || currentFrame;
  const interval = Math.max(16, (currentFrame.timestamp - older.timestamp) * 1000);
  const alpha = currentFrame.reset_frame || currentFrame.paused
    ? 1
    : THREE.MathUtils.clamp((now - currentReceivedAt) / interval, 0, 1);
  const oldIndex = new Map(older.body_names.map((name, index) => [name, index]));
  currentFrame.body_names.forEach((name, index) => {
    const group = bodyGroups.get(name);
    if (!group) return;
    const prior = oldIndex.get(name);
    if (prior === undefined) {
      group.position.fromArray(currentFrame.body_pos_w[index]);
      group.quaternion.copy(quaternionFromWxyz(currentFrame.body_quat_w[index]));
      return;
    }
    const fromPosition = new THREE.Vector3().fromArray(older.body_pos_w[prior]);
    const toPosition = new THREE.Vector3().fromArray(currentFrame.body_pos_w[index]);
    group.position.lerpVectors(fromPosition, toPosition, alpha);
    const fromQuaternion = quaternionFromWxyz(older.body_quat_w[prior]);
    const toQuaternion = quaternionFromWxyz(currentFrame.body_quat_w[index]);
    group.quaternion.copy(fromQuaternion).slerp(toQuaternion, alpha);
  });
}

function resize() {
  const { clientWidth, clientHeight } = viewport;
  renderer.setSize(clientWidth, clientHeight, false);
  camera.aspect = clientWidth / Math.max(clientHeight, 1);
  camera.updateProjectionMatrix();
}

function render(now) {
  applyPose(now);
  terrainRenderer.tick(now);
  weatherEffects.tick(now);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}

play.addEventListener('click', async () => {
  // A zero velocity command is an intentional stand controller. The old UI
  // never sent a walking command, so Play looked broken even as inference ran.
  if (paused) await control('play');
  else await control('pause', true);
});
reset.addEventListener('click', () => control('reset'));
homeCamera.addEventListener('click', resetCamera);
stand.addEventListener('click', async () => {
  await control('command', [0.0, 0.0, 0.0]);
  if (!paused) await control('pause', true);
});
document.querySelectorAll('[data-surface]').forEach((button) => {
  button.addEventListener('click', async () => {
    const state = await request('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'surface', value: button.dataset.surface }),
    });
    applyState(state);
    if (everestTile) terrainRenderer.loadPathPreview(everestTile, state.snow, debugControls.parameters);
    else terrainRenderer.loadPreview(state.snow);
  });
});
debugPanel.querySelector('#debug-close').addEventListener('click', () => {
  debugPanel.dataset.open = 'false';
  debugOpen.hidden = false;
  debugOpen.setAttribute('aria-expanded', 'false');
});
debugOpen.addEventListener('click', () => {
  debugPanel.dataset.open = 'true';
  debugOpen.hidden = true;
  debugOpen.setAttribute('aria-expanded', 'true');
});
document.addEventListener('keydown', (event) => {
  if (event.code === 'Space' && event.target === document.body) {
    event.preventDefault();
    play.click();
  }
});
window.addEventListener('resize', resize);

async function start() {
  try {
    const [state] = await Promise.all([request('/api/state'), loadRobot()]);
    applyState(state);
    ingestFrame(await request('/api/frame'));
    loading.hidden = true;
    resize();
    requestAnimationFrame(render);
    pollFrame();
    if (terrainTelemetryUrl) pollTerrain();
  } catch (reason) {
    loading.hidden = true;
    error.textContent = `Viewer unavailable · ${reason.message}`;
    error.hidden = false;
    setConnection(false);
  }
}

start();
