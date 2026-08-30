const activityButtons = [...document.querySelectorAll('[data-workspace]')];
const workspacePanels = [...document.querySelectorAll('[data-panel]')];
const inspectorTabs = [...document.querySelectorAll('[data-inspector-tab]')];
const inspectorGroups = [...document.querySelectorAll('[data-inspector-group]')];
const editorTabs = [...document.querySelectorAll('.editor-tab[data-editor-tab]')];
const treeItems = [...document.querySelectorAll('.tree-item[data-editor-tab]')];
const terrainInputs = [...document.querySelectorAll('[data-terrain-param]')];
const terrainStatus = document.querySelector('#terrain-edit-status');
const terrainApply = document.querySelector('#terrain-apply');
const terrainReset = document.querySelector('#terrain-reset');

function setWorkspace(name) {
  activityButtons.forEach((button) => button.classList.toggle('active', button.dataset.workspace === name));
  workspacePanels.forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === name));
}

function setInspector(name) {
  inspectorTabs.forEach((button) => button.classList.toggle('active', button.dataset.inspectorTab === name));
  inspectorGroups.forEach((group) => group.classList.toggle('active', group.dataset.inspectorGroup === name));
}

function setEditorTab(name) {
  editorTabs.forEach((button) => button.classList.toggle('active', button.dataset.editorTab === name));
  treeItems.forEach((button) => button.classList.toggle('active', button.dataset.editorTab === name));
  if (name === 'terrain') setInspector('terrain');
  if (name === 'snow') setInspector('snow');
  if (name === 'robot') setInspector('simulation');
}

activityButtons.forEach((button) => button.addEventListener('click', () => {
  setWorkspace(button.dataset.workspace);
  if (button.dataset.workspace !== 'explorer') setInspector(button.dataset.workspace === 'simulation' ? 'simulation' : button.dataset.workspace);
}));
inspectorTabs.forEach((button) => button.addEventListener('click', () => setInspector(button.dataset.inspectorTab)));
[...editorTabs, ...treeItems].forEach((button) => button.addEventListener('click', () => setEditorTab(button.dataset.editorTab)));
document.querySelectorAll('[data-focus-inspector]').forEach((button) => button.addEventListener('click', () => setInspector(button.dataset.focusInspector)));
document.querySelectorAll('[data-canvas-tool]').forEach((button) => button.addEventListener('click', () => {
  document.querySelectorAll('[data-canvas-tool]').forEach((item) => item.classList.toggle('active', item === button));
  if (button.dataset.canvasTool === 'crop') {
    setWorkspace('terrain');
    setInspector('terrain');
  }
}));

function terrainValue(name) {
  return Number(document.querySelector(`[data-terrain-param="${name}"]`)?.value || 0);
}

function renderTerrainOutputs() {
  const formats = {
    scale_xy: `${terrainValue('scale_xy').toFixed(2)}×`,
    scale_z: `${terrainValue('scale_z').toFixed(2)}×`,
    crop_left: `${Math.round(terrainValue('crop_left'))}%`,
    crop_right: `${Math.round(terrainValue('crop_right'))}%`,
    crop_bottom: `${Math.round(terrainValue('crop_bottom'))}%`,
    crop_top: `${Math.round(terrainValue('crop_top'))}%`,
    display_z: `${terrainValue('display_z').toFixed(2)}×`,
  };
  Object.entries(formats).forEach(([name, value]) => {
    const output = document.querySelector(`[data-terrain-output="${name}"]`);
    if (output) output.textContent = value;
  });
}

function terrainPayload() {
  return {
    scale_xy: terrainValue('scale_xy'),
    scale_z: terrainValue('scale_z'),
    crop: [
      terrainValue('crop_left') / 100,
      terrainValue('crop_right') / 100,
      terrainValue('crop_bottom') / 100,
      terrainValue('crop_top') / 100,
    ],
  };
}

function applyDisplayExaggeration() {
  const factor = terrainValue('display_z');
  const viewer = window.__everestViewer;
  if (viewer?.terrainRenderer?.root) viewer.terrainRenderer.root.scale.z = factor;
}

terrainInputs.forEach((input) => input.addEventListener('input', () => {
  renderTerrainOutputs();
  if (input.dataset.terrainParam === 'display_z') applyDisplayExaggeration();
}));

async function postControl(action, value) {
  const response = await fetch('/api/control', {
    method: 'POST',
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, value }),
  });
  if (!response.ok) throw new Error(await response.text() || `${response.status} ${response.statusText}`);
  return response.json();
}

async function reloadEditedTerrain(state) {
  const viewer = window.__everestViewer;
  if (!viewer?.terrainRenderer) return;
  const response = await fetch('/everest-terrain.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`terrain ${response.status}`);
  const tile = await response.json();
  viewer.terrainRenderer.loadEverestTile(tile, state?.snow || {});
  applyDisplayExaggeration();
}

terrainApply.addEventListener('click', async () => {
  terrainApply.disabled = true;
  terrainStatus.textContent = 'Applying shared physical terrain…';
  try {
    const payload = terrainPayload();
    if (payload.crop[1] - payload.crop[0] < 0.08 || payload.crop[3] - payload.crop[2] < 0.08) {
      throw new Error('Crop must retain at least 8% of each axis');
    }
    const state = await postControl('terrain_edit', payload);
    await reloadEditedTerrain(state);
    terrainStatus.textContent = `${state.terrain_edit.scale_xy.toFixed(2)}× XY · ${state.terrain_edit.scale_z.toFixed(2)}× Z · collision rebuilt`;
    setEditorTab('terrain');
  } catch (error) {
    terrainStatus.textContent = `Edit rejected · ${error.message}`;
  } finally {
    terrainApply.disabled = false;
  }
});

terrainReset.addEventListener('click', () => {
  const defaults = { scale_xy: 1, scale_z: 1, crop_left: 0, crop_right: 100, crop_bottom: 0, crop_top: 100, display_z: 1 };
  Object.entries(defaults).forEach(([name, value]) => {
    const input = document.querySelector(`[data-terrain-param="${name}"]`);
    if (input) input.value = String(value);
  });
  renderTerrainOutputs();
  applyDisplayExaggeration();
  terrainStatus.textContent = 'Reset staged · Apply physical edit to commit';
});

async function hydrateTerrainEditor() {
  try {
    const response = await fetch('/api/state', { cache: 'no-store' });
    if (!response.ok) return;
    const state = await response.json();
    const edit = state.terrain_edit;
    if (!edit) return;
    const values = {
      scale_xy: edit.scale_xy,
      scale_z: edit.scale_z,
      crop_left: edit.crop[0] * 100,
      crop_right: edit.crop[1] * 100,
      crop_bottom: edit.crop[2] * 100,
      crop_top: edit.crop[3] * 100,
    };
    Object.entries(values).forEach(([name, value]) => {
      const input = document.querySelector(`[data-terrain-param="${name}"]`);
      if (input) input.value = String(value);
    });
    renderTerrainOutputs();
  } catch (_) {
    // The normal viewer connection indicator owns offline reporting.
  }
}

renderTerrainOutputs();
hydrateTerrainEditor();
window.setTimeout(applyDisplayExaggeration, 800);
