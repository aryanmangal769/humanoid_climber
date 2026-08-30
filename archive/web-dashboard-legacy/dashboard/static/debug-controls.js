const DEFAULTS = {
  wind_speed_m_s: 18,
  wind_direction_deg: 245,
  snowfall_mm_h: 2.5,
  temperature_c: -14,
  slope_deg: 21,
  path_width_m: 8,
  path_length_m: 28,
  surface_friction: 0.35,
  layers: [
    {
      type: 'POWDER',
      label: 'Fresh snow',
      color: [0.94, 0.97, 1.0],
      thickness_m: 0.03,
      density_kg_m3: 120,
      stiffness_pa: 60000,
      compressive_strength_pa: 4000,
      shear_strength_pa: 1200,
      compaction_hardening: 4,
      bond_strength_below_pa: 800,
    },
    {
      type: 'CRUST',
      label: 'Wind crust',
      color: [0.72, 0.82, 0.9],
      thickness_m: 0.06,
      density_kg_m3: 380,
      stiffness_pa: 600000,
      compressive_strength_pa: 70000,
      shear_strength_pa: 25000,
      compaction_hardening: 18,
      bond_strength_below_pa: 6000,
    },
    {
      type: 'POWDER',
      label: 'Weak layer',
      color: [0.82, 0.89, 0.95],
      thickness_m: 0.25,
      density_kg_m3: 210,
      stiffness_pa: 140000,
      compressive_strength_pa: 12000,
      shear_strength_pa: 3500,
      compaction_hardening: 7,
      bond_strength_below_pa: 1200,
    },
    {
      type: 'DENSE_SNOW',
      label: 'Dense old snow',
      color: [0.57, 0.69, 0.78],
      thickness_m: 0.4,
      density_kg_m3: 520,
      stiffness_pa: 1500000,
      compressive_strength_pa: 180000,
      shear_strength_pa: 75000,
      compaction_hardening: 28,
      bond_strength_below_pa: 12000,
    },
  ],
};

function cloneDefaults() {
  return JSON.parse(JSON.stringify(DEFAULTS));
}

function formatGlobal(name, value) {
  const formats = {
    wind_speed_m_s: `${Math.round(value)} m/s`,
    wind_direction_deg: `${Math.round(value)}°`,
    snowfall_mm_h: `${Number(value).toFixed(1)} mm/h`,
    temperature_c: `${value < 0 ? '−' : ''}${Math.abs(Math.round(value))} °C`,
    slope_deg: `${Math.round(value)}°`,
    surface_friction: `μ ${Number(value).toFixed(2)}`,
  };
  return formats[name] || String(value);
}

function formatLayer(name, value) {
  if (name === 'thickness_m') return `${Math.round(value * 100)} cm`;
  if (name === 'density_kg_m3') return `${Math.round(value)} kg/m³`;
  if (name === 'compaction_hardening') return Number(value).toFixed(0);
  const kilopascals = value / 1000;
  return kilopascals >= 1000 ? `${(kilopascals / 1000).toFixed(2)} MPa` : `${kilopascals.toFixed(kilopascals < 10 ? 1 : 0)} kPa`;
}

export class SnowDebugControls {
  constructor(root, { onChange = () => {} } = {}) {
    this.root = root;
    this.onChange = onChange;
    this.parameters = cloneDefaults();
    this.activeLayer = 0;
    this.scheduled = false;
    this.select = root.querySelector('#active-layer');
    this.stack = root.querySelector('#layer-stack');
    this.globalInputs = [...root.querySelectorAll('[data-param]')];
    this.layerInputs = [...root.querySelectorAll('[data-layer-param]')];
    this.bind();
    this.renderAll();
    this.emit();
  }

  bind() {
    this.globalInputs.forEach((input) => {
      input.addEventListener('input', () => {
        this.parameters[input.dataset.param] = Number(input.value);
        this.renderGlobal(input.dataset.param);
        this.emit();
      });
    });
    this.layerInputs.forEach((input) => {
      input.addEventListener('input', () => {
        const name = input.dataset.layerParam;
        const multiplier = name.endsWith('_pa') ? 1000 : 1;
        this.parameters.layers[this.activeLayer][name] = Number(input.value) * multiplier;
        this.renderLayerOutput(name);
        this.renderStack();
        this.emit();
      });
    });
    this.select.addEventListener('change', () => this.selectLayer(Number(this.select.value)));
    this.stack.addEventListener('click', (event) => {
      const button = event.target.closest('[data-layer-index]');
      if (button) this.selectLayer(Number(button.dataset.layerIndex));
    });
    this.root.querySelector('#debug-reset').addEventListener('click', () => {
      this.parameters = cloneDefaults();
      this.activeLayer = 0;
      this.renderAll();
      this.emit();
    });
  }

  selectLayer(index) {
    this.activeLayer = Math.max(0, Math.min(this.parameters.layers.length - 1, index));
    this.select.value = String(this.activeLayer);
    this.renderLayerInputs();
    this.renderStack();
  }

  renderAll() {
    this.globalInputs.forEach((input) => {
      input.value = this.parameters[input.dataset.param];
      this.renderGlobal(input.dataset.param);
    });
    this.select.replaceChildren(...this.parameters.layers.map((layer, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = `${index + 1} · ${layer.label}`;
      return option;
    }));
    this.select.value = String(this.activeLayer);
    this.renderLayerInputs();
    this.renderStack();
  }

  renderGlobal(name) {
    const output = this.root.querySelector(`[data-output="${name}"]`);
    if (output) output.textContent = formatGlobal(name, this.parameters[name]);
  }

  renderLayerInputs() {
    const layer = this.parameters.layers[this.activeLayer];
    this.layerInputs.forEach((input) => {
      const name = input.dataset.layerParam;
      input.value = name.endsWith('_pa') ? layer[name] / 1000 : layer[name];
      this.renderLayerOutput(name);
    });
  }

  renderLayerOutput(name) {
    const output = this.root.querySelector(`[data-layer-output="${name}"]`);
    if (output) output.textContent = formatLayer(name, this.parameters.layers[this.activeLayer][name]);
  }

  renderStack() {
    const total = this.parameters.layers.reduce((sum, layer) => sum + layer.thickness_m, 0);
    this.stack.replaceChildren(...this.parameters.layers.map((layer, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.layerIndex = String(index);
      button.dataset.active = String(index === this.activeLayer);
      button.style.flexGrow = String(Math.max(0.75, layer.thickness_m / Math.max(total, 0.01) * 5));
      button.style.setProperty('--layer-color', `rgb(${layer.color.map((value) => Math.round(value * 255)).join(' ')})`);
      button.innerHTML = `<span>${index + 1}</span><small>${Math.round(layer.thickness_m * 100)} cm</small>`;
      button.title = `${layer.label}: ${Math.round(layer.density_kg_m3)} kg/m³`;
      return button;
    }));
  }

  emit() {
    if (this.scheduled) return;
    this.scheduled = true;
    requestAnimationFrame(() => {
      this.scheduled = false;
      this.onChange(this.parameters);
    });
  }
}
