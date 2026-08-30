import * as THREE from 'three';

const DEFAULT_LAYERS = [
  { id: 0, name: 'loose snow', color: [0.94, 0.97, 1.0], roughness: 0.96 },
  { id: 1, name: 'packed snow', color: [0.88, 0.93, 0.98], roughness: 0.78 },
  { id: 2, name: 'ice crust', color: [0.42, 0.68, 0.85], roughness: 0.16 },
];

function flatten(values) {
  if (!Array.isArray(values)) return values;
  return Array.isArray(values[0]) ? values.flat() : values;
}

function decodeBase64(value, Type) {
  const encoded = value.includes(',') ? value.slice(value.indexOf(',') + 1) : value;
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new Type(bytes.buffer);
}

function typedValues(value, Type = Float32Array) {
  if (value == null) return null;
  if (ArrayBuffer.isView(value)) return value;
  if (typeof value === 'string') return decodeBase64(value, Type);
  return new Type(flatten(value));
}

function makeSnowTexture() {
  const size = 128;
  const data = new Uint8Array(size * size * 4);
  let seed = 0x8f3d29a1;
  const random = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 4294967295;
  };
  for (let index = 0; index < size * size; index += 1) {
    const grain = Math.round(154 + random() * 101);
    data[index * 4] = grain;
    data[index * 4 + 1] = grain;
    data[index * 4 + 2] = grain;
    data[index * 4 + 3] = 255;
  }
  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(18, 18);
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;
  return texture;
}

function layerColor(layer, compaction = 0) {
  const base = new THREE.Color(...(layer?.color || DEFAULT_LAYERS[0].color));
  const packed = new THREE.Color(0.75, 0.84, 0.92);
  return base.lerp(packed, THREE.MathUtils.clamp(compaction, 0, 1) * 0.2);
}

/** Browser-side renderer for Newton snow/ice telemetry.
 *
 * Accepted frame shape (arrays may also be little-endian base64):
 * {
 *   schema: "everest-terrain/v1", sequence, timestamp, sim_time,
 *   origin: [x_min, y_min, z_reference], size: [width, depth],
 *   resolution: [nx, ny], heights: float[nx*ny],
 *   material_ids: uint8[nx*ny], compaction: float[nx*ny],
 *   layers: [{id, name, color, roughness, depth}],
 *   particles: {positions: float[n*3], radii: float[n], material_ids: uint8[n]}
 * }
 */
export class TerrainRenderer {
  constructor(scene, { onStatus = () => {} } = {}) {
    this.onStatus = onStatus;
    this.root = new THREE.Group();
    this.root.name = 'newton-terrain';
    this.root.rotation.x = -Math.PI / 2;
    scene.add(this.root);

    this.sequence = -1;
    this.mode = 'preview';
    this.layers = DEFAULT_LAYERS;
    this.resolution = [0, 0];
    this.heightMesh = null;
    this.snowCap = null;
    this.iceBase = null;
    this.fromHeights = null;
    this.targetHeights = null;
    this.fromCapHeights = null;
    this.targetCapHeights = null;
    this.surface = 'snow';
    this.surfaceDepth = 0.12;
    this.transitionStarted = 0;
    this.transitionMs = 110;
    this.normalTick = 0;
    this.normalsSettled = false;
    this.snowTexture = makeSnowTexture();
    this.material = new THREE.MeshStandardMaterial({
      color: 0x6f8fa3,
      roughness: 0.9,
      metalness: 0,
      wireframe: true,
      transparent: true,
      opacity: 0.32,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.particleGeometry = new THREE.IcosahedronGeometry(1, 1);
    this.particleMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.9,
      metalness: 0,
    });
    this.particles = null;
    this.particleCapacity = 0;
    this.routeLine = null;
    this.windArrow = null;
    this.matrix = new THREE.Matrix4();
    this.position = new THREE.Vector3();
    this.scale = new THREE.Vector3();
    this.quaternion = new THREE.Quaternion();
  }

  loadPreview(config = {}) {
    const resolution = [97, 97];
    const origin = [-4, -4, 0];
    const size = [8, 8];
    const count = resolution[0] * resolution[1];
    const heights = new Float32Array(count);
    const materials = new Uint8Array(count);
    const compaction = new Float32Array(count);
    for (let row = 0; row < resolution[1]; row += 1) {
      for (let column = 0; column < resolution[0]; column += 1) {
        const index = row * resolution[0] + column;
        const x = origin[0] + (column / (resolution[0] - 1)) * size[0];
        const y = origin[1] + (row / (resolution[1] - 1)) * size[1];
        const drift = 0.052 * Math.exp(-((x + 1.35) ** 2) / 5.2 - ((y + 0.4) ** 2) / 1.9);
        const wind = 0.012 * Math.sin(x * 1.6 + y * 0.34) + 0.006 * Math.sin(y * 3.1 - x * 0.7);
        const ice = config.surface === 'ice';
        heights[index] = Math.max(0.012, 0.055 + drift + wind - (ice ? 0.032 : 0));
        materials[index] = ice ? 2 : (drift > 0.025 ? 0 : 1);
        compaction[index] = ice ? 1 : THREE.MathUtils.clamp(0.3 - drift * 3, 0, 0.7);
      }
    }
    this.ingest({
      schema: 'everest-terrain/v1',
      surface_kind: config.surface || 'snow',
      surface_depth: Number(config.depth || 0.12),
      sequence: -1,
      timestamp: Date.now() / 1000,
      origin,
      size,
      resolution,
      heights,
      material_ids: materials,
      compaction,
      layers: DEFAULT_LAYERS.map((layer, index) => ({
        ...layer,
        depth: index === 0 ? Number(config.depth || 0.12) : (index === 1 ? 0.08 : 0.035),
      })),
      preview: true,
    });
  }

  loadEverestTile(tile, config = {}) {
    if (tile?.schema !== 'everest-terrain/v1') {
      throw new Error(`Unsupported Everest terrain schema: ${tile?.schema || 'missing'}`);
    }
    const width = Number(tile.grid_width);
    const depth = Number(tile.grid_height);
    const sourceHeights = typedValues(tile.heights, Float32Array);
    const center = tile.terrain_center;
    if (width < 2 || depth < 2 || sourceHeights?.length !== width * depth) {
      throw new Error('Everest terrain has invalid grid dimensions');
    }
    if (!Array.isArray(center) || center.length !== 3) {
      throw new Error('Everest terrain is missing terrain_center');
    }

    // GeoTIFF rows are north-to-south; the renderer grid grows south-to-north.
    const heights = new Float32Array(sourceHeights.length);
    for (let row = 0; row < depth; row += 1) {
      const sourceRow = depth - 1 - row;
      heights.set(sourceHeights.subarray(sourceRow * width, (sourceRow + 1) * width), row * width);
    }
    const worldWidth = Number(tile.world_width_m);
    const worldDepth = Number(tile.world_depth_m);
    if (!(worldWidth > 0) || !(worldDepth > 0)) {
      throw new Error('Everest terrain has invalid world dimensions');
    }
    let minimumHeight = Infinity;
    for (const height of sourceHeights) minimumHeight = Math.min(minimumHeight, height);
    const origin = [
      Number(center[0]) - worldWidth / 2,
      Number(center[1]) - worldDepth / 2,
      Number(center[2]) + minimumHeight,
    ];
    const materialId = config.surface === 'ice' ? 2 : 1;
    const materials = new Uint8Array(sourceHeights.length).fill(materialId);
    const compaction = new Float32Array(sourceHeights.length).fill(materialId === 2 ? 1 : 0.45);
    this.ingest({
      schema: 'everest-terrain/v1',
      mode: 'everest',
      surface_kind: config.surface || 'snow',
      surface_depth: Number(config.depth || 0.12),
      preview: true,
      sequence: -1,
      timestamp: Date.now() / 1000,
      origin,
      size: [worldWidth, worldDepth],
      resolution: [width, depth],
      heights,
      material_ids: materials,
      compaction,
      layers: DEFAULT_LAYERS.map((layer, index) => ({
        ...layer,
        depth: index === 0 ? Number(config.depth || 0.12) : (index === 1 ? 0.08 : 0.035),
      })),
    });
    this.updateRoute(tile.route);
  }

  loadPathPreview(tile, config = {}, parameters = {}) {
    const widthMetres = Number(parameters.path_width_m || 8);
    const lengthMetres = Number(parameters.path_length_m || 28);
    const columns = 73;
    const rows = 225;
    const origin = [-widthMetres / 2, -2.5, 0];
    const size = [widthMetres, lengthMetres];
    const heights = new Float32Array(columns * rows);
    const materials = new Uint8Array(columns * rows);
    const compaction = new Float32Array(columns * rows);
    const layers = Array.isArray(parameters.layers) && parameters.layers.length
      ? parameters.layers
      : DEFAULT_LAYERS;
    const totalDepth = layers.reduce((sum, layer) => sum + Number(layer.thickness_m || layer.depth || 0), 0);
    const windSpeed = Number(parameters.wind_speed_m_s || 0);
    const windDirection = THREE.MathUtils.degToRad(Number(parameters.wind_direction_deg || 0));
    const snowfall = Number(parameters.snowfall_mm_h || 0);
    const temperature = Number(parameters.temperature_c ?? -14);
    const slopeDeg = THREE.MathUtils.clamp(Number(parameters.slope_deg || 21), 8, 32);
    const slope = Math.tan(THREE.MathUtils.degToRad(slopeDeg));
    const windX = Math.sin(windDirection);
    const windY = Math.cos(windDirection);
    const surfaceLayer = layers[0] || {};
    const stiffness = Math.max(1000, Number(surfaceLayer.stiffness_pa || 60000));
    const compression = Math.max(500, Number(surfaceLayer.compressive_strength_pa || 4000));
    const shear = Math.max(200, Number(surfaceLayer.shear_strength_pa || 1200));
    const density = Math.max(50, Number(surfaceLayer.density_kg_m3 || 120));
    const softness = THREE.MathUtils.clamp(
      1.15 - Math.log10(stiffness + compression) / 6.2,
      0.12,
      0.95,
    );
    const erosion = THREE.MathUtils.clamp(1.2 - Math.log10(shear) / 4.6, 0.08, 0.9);
    const semantic = String(surfaceLayer.type || surfaceLayer.name || 'POWDER').toUpperCase();
    const surfaceMaterial = semantic.includes('ICE') || temperature > -1
      ? 2
      : (semantic.includes('PACK') || semantic.includes('CRUST') ? 1 : 0);

    for (let row = 0; row < rows; row += 1) {
      const v = row / (rows - 1);
      const y = origin[1] + v * lengthMetres;
      const centerline = 0.55 * Math.sin(v * Math.PI * 1.35) + 0.16 * Math.sin(v * Math.PI * 4.8);
      for (let column = 0; column < columns; column += 1) {
        const index = row * columns + column;
        const u = column / (columns - 1);
        const x = origin[0] + u * widthMetres;
        const crossSlope = 0.025 * x + 0.035 * Math.sin(x * 1.3 + y * 0.22);
        const rock = 0.018 * Math.sin(x * 2.7 - y * 0.8) * Math.sin(y * 0.55);
        const windCoordinate = x * windX + y * windY;
        const windDrift = (0.003 + windSpeed * 0.0007 * erosion + snowfall * 0.00035)
          * Math.sin(windCoordinate * 1.35 + Math.sin(y * 0.3));
        const pathCompression = Math.exp(-((x - centerline) ** 2) / 0.42);
        const grade = Math.max(0, y) * slope;
        heights[index] = grade + crossSlope + rock + windDrift - pathCompression * (0.006 + softness * 0.055);
        materials[index] = surfaceMaterial;
        compaction[index] = THREE.MathUtils.clamp(
          Number(surfaceLayer.compaction_hardening || 0) / 30
            + density / 1800
            + pathCompression * (0.18 + softness * 0.25)
            + windSpeed / 160,
          0,
          1,
        );
      }
    }
    this.ingest({
      schema: 'everest-terrain/v1',
      mode: 'path',
      preview: true,
      sequence: -1,
      timestamp: Date.now() / 1000,
      origin,
      size,
      resolution: [columns, rows],
      heights,
      material_ids: materials,
      compaction,
      surface_kind: semantic.includes('ICE') ? 'ice' : 'snow',
      surface_depth: Math.max(0.03, totalDepth),
      surface_friction: Number(parameters.surface_friction ?? config.friction ?? 0.35),
      layers,
      wireframe_base: true,
      path: {
        width_m: widthMetres,
        length_m: lengthMetres,
        slope_deg: slopeDeg,
      },
    });
    this.updatePathLine(origin, size, slope);
    this.updateWindArrow(Number(parameters.wind_direction_deg || 0), windSpeed);
  }

  updatePathLine(origin, size, slope) {
    const points = [];
    for (let index = 0; index < 80; index += 1) {
      const v = index / 79;
      const y = origin[1] + v * size[1];
      const x = 0.55 * Math.sin(v * Math.PI * 1.35) + 0.16 * Math.sin(v * Math.PI * 4.8);
      points.push(new THREE.Vector3(x, y, Math.max(0, y) * slope + 0.035));
    }
    this.replaceRoute(points, 0x8ba9bb);
  }

  updateWindArrow(directionDeg, speed) {
    if (this.windArrow) {
      this.root.remove(this.windArrow);
      this.windArrow.dispose();
      this.windArrow = null;
    }
    const direction = THREE.MathUtils.degToRad(directionDeg);
    const vector = new THREE.Vector3(-Math.sin(direction), -Math.cos(direction), 0).normalize();
    this.windArrow = new THREE.ArrowHelper(
      vector,
      new THREE.Vector3(-2.4, -0.8, 0.55),
      THREE.MathUtils.clamp(0.8 + speed * 0.035, 0.8, 2.2),
      0x8ea8ff,
      0.2,
      0.1,
    );
    this.windArrow.name = 'debug-wind-vector';
    this.root.add(this.windArrow);
  }

  updateRoute(route) {
    if (this.routeLine) {
      this.root.remove(this.routeLine);
      this.routeLine.geometry.dispose();
      this.routeLine.material.dispose();
      this.routeLine = null;
    }
    if (!Array.isArray(route) || route.length < 2) return;
    const points = route.map(([x, y, z]) => new THREE.Vector3(Number(x), Number(y), Number(z) + 0.008));
    this.replaceRoute(points, 0xff8a3d);
  }

  replaceRoute(points, color) {
    if (this.routeLine) {
      this.root.remove(this.routeLine);
      this.routeLine.geometry.dispose();
      this.routeLine.material.dispose();
      this.routeLine = null;
    }
    if (!Array.isArray(points) || points.length < 2) return;
    this.routeLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.82, toneMapped: false }),
    );
    this.routeLine.name = 'everest-south-col-route';
    this.routeLine.renderOrder = 2;
    this.root.add(this.routeLine);
  }

  ingest(payload) {
    if (!payload) return false;
    const frame = payload.surface && typeof payload.surface === 'object'
      ? payload.surface
      : (payload.heightfield || payload);
    const resolution = frame.resolution || payload.resolution;
    const heights = typedValues(frame.heights ?? frame.height, Float32Array);
    if (!resolution || resolution.length !== 2 || !heights) return false;
    const [width, depth] = resolution.map(Number);
    if (width < 2 || depth < 2 || heights.length !== width * depth) return false;
    const sequence = Number(payload.sequence ?? frame.sequence ?? 0);
    if (!payload.preview && sequence <= this.sequence) return false;

    this.mode = payload.mode || (payload.preview ? 'preview' : 'live');
    this.surface = payload.surface_kind
      || (typeof payload.surface === 'string' ? payload.surface : null)
      || payload.snow?.surface
      || this.surface
      || 'snow';
    this.surfaceDepth = Math.max(
      0.002,
      Number(payload.surface_depth ?? payload.snow?.depth ?? (this.surface === 'ice' ? 0.035 : 0.12)),
    );
    if (!payload.preview) this.sequence = sequence;
    this.layers = payload.layers || frame.layers || this.layers || DEFAULT_LAYERS;
    const origin = frame.origin || payload.origin || [-2, -2, 0];
    const size = frame.size || payload.size || [4, 4];
    const materials = typedValues(frame.material_ids ?? frame.materials, Uint8Array);
    const compaction = typedValues(frame.compaction, Float32Array);

    if (!this.heightMesh || this.resolution[0] !== width || this.resolution[1] !== depth) {
      this.buildSurface(width, depth, origin, size);
      this.fromHeights = heights.map((height) => height - this.surfaceDepth);
      this.buildSnowCap(width, depth, origin, size);
      this.fromCapHeights = new Float32Array(heights);
    } else {
      const positions = this.heightMesh.geometry.attributes.position.array;
      this.fromHeights = new Float32Array(width * depth);
      for (let index = 0; index < this.fromHeights.length; index += 1) this.fromHeights[index] = positions[index * 3 + 2];
      const capPositions = this.snowCap.geometry.attributes.position.array;
      this.fromCapHeights = new Float32Array(width * depth);
      for (let index = 0; index < this.fromCapHeights.length; index += 1) this.fromCapHeights[index] = capPositions[index * 3 + 2];
    }
    this.targetHeights = new Float32Array(heights.length);
    this.targetCapHeights = new Float32Array(heights.length);
    for (let index = 0; index < heights.length; index += 1) {
      this.targetHeights[index] = heights[index] - this.surfaceDepth;
      this.targetCapHeights[index] = heights[index];
    }
    this.transitionStarted = performance.now();
    this.normalsSettled = false;
    this.updateColors(materials, compaction);
    this.updateCapMaterial();
    if (payload.wireframe_base) {
      if (this.iceBase) {
        this.root.remove(this.iceBase);
        this.iceBase.geometry.dispose();
        this.iceBase.material.dispose();
        this.iceBase = null;
      }
    } else {
      this.updateIceBase(origin, size, this.layers);
    }
    this.updateParticles(payload.particles || frame.particles, origin);
    this.onStatus({
      mode: this.mode,
      sequence,
      resolution: [width, depth],
      particleCount: this.particles?.count || 0,
      layers: this.layers,
      simTime: payload.sim_time,
      surface: this.surface,
      surfaceDepth: this.surfaceDepth,
      path: payload.path,
    });
    return true;
  }

  buildSurface(width, depth, origin, size) {
    if (this.heightMesh) {
      this.root.remove(this.heightMesh);
      this.heightMesh.geometry.dispose();
    }
    const count = width * depth;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const uvs = new Float32Array(count * 2);
    for (let row = 0; row < depth; row += 1) {
      for (let column = 0; column < width; column += 1) {
        const index = row * width + column;
        const u = column / (width - 1);
        const v = row / (depth - 1);
        positions[index * 3] = origin[0] + u * size[0];
        positions[index * 3 + 1] = origin[1] + v * size[1];
        positions[index * 3 + 2] = origin[2] || 0;
        colors.set(DEFAULT_LAYERS[0].color, index * 3);
        uvs[index * 2] = u;
        uvs[index * 2 + 1] = v;
      }
    }
    const triangles = new Uint32Array((width - 1) * (depth - 1) * 6);
    let offset = 0;
    for (let row = 0; row < depth - 1; row += 1) {
      for (let column = 0; column < width - 1; column += 1) {
        const a = row * width + column;
        const b = a + 1;
        const c = a + width;
        const d = c + 1;
        triangles.set([a, b, c, b, d, c], offset);
        offset += 6;
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geometry.setIndex(new THREE.BufferAttribute(triangles, 1));
    geometry.computeVertexNormals();
    this.heightMesh = new THREE.Mesh(geometry, this.material);
    this.heightMesh.name = 'newton-heightfield';
    this.heightMesh.renderOrder = 2;
    this.heightMesh.receiveShadow = true;
    this.heightMesh.castShadow = true;
    this.root.add(this.heightMesh);
    this.resolution = [width, depth];
  }

  buildSnowCap(width, depth, origin, size) {
    if (this.snowCap) {
      this.root.remove(this.snowCap);
      this.snowCap.geometry.dispose();
      this.snowCap.material.dispose();
    }
    const geometry = this.heightMesh.geometry.clone();
    this.snowCap = new THREE.Mesh(geometry, new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.84,
      metalness: 0.01,
      clearcoat: 0.2,
      clearcoatRoughness: 0.5,
      bumpMap: this.snowTexture,
      bumpScale: 0.006,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.84,
      depthWrite: false,
    }));
    this.snowCap.name = 'snow-ice-pack-cap';
    this.snowCap.receiveShadow = true;
    this.snowCap.castShadow = true;
    this.snowCap.renderOrder = 1;
    this.root.add(this.snowCap);
  }

  updateCapMaterial() {
    if (!this.snowCap) return;
    const material = this.snowCap.material;
    if (this.surface === 'ice') {
      material.color.setRGB(1, 1, 1);
      material.roughness = 0.12;
      material.metalness = 0.04;
      material.transmission = 0.28;
      material.opacity = 0.72;
      material.clearcoat = 1;
      material.clearcoatRoughness = 0.08;
    } else {
      material.color.setRGB(1, 1, 1);
      material.roughness = 0.84;
      material.metalness = 0.01;
      material.transmission = 0.02;
      material.opacity = 0.84;
      material.clearcoat = 0.2;
      material.clearcoatRoughness = 0.5;
    }
    material.needsUpdate = true;
  }

  updateColors(materials, compaction) {
    const colors = this.snowCap.geometry.attributes.color;
    const color = new THREE.Color();
    for (let index = 0; index < colors.count; index += 1) {
      const materialId = materials?.[index] ?? 0;
      const layer = this.layers.find((item) => Number(item.id) === Number(materialId)) || DEFAULT_LAYERS[0];
      color.copy(layerColor(layer, compaction?.[index] || 0));
      colors.setXYZ(index, color.r, color.g, color.b);
    }
    colors.needsUpdate = true;
  }

  updateIceBase(origin, size, layers) {
    const ice = layers.find((layer) => String(layer.name).toLowerCase().includes('ice')) || DEFAULT_LAYERS[2];
    const thickness = Number(ice.depth || 0.035);
    if (this.iceBase) {
      this.root.remove(this.iceBase);
      this.iceBase.geometry.dispose();
      this.iceBase.material.dispose();
    }
    this.iceBase = new THREE.Mesh(
      new THREE.BoxGeometry(size[0], size[1], thickness),
      new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(...(ice.color || DEFAULT_LAYERS[2].color)),
        roughness: Number(ice.roughness ?? 0.16),
        metalness: 0.06,
        transmission: 0.16,
        transparent: true,
        opacity: 0.68,
        clearcoat: 1,
        clearcoatRoughness: 0.1,
      }),
    );
    this.iceBase.name = 'ice-stratum';
    this.iceBase.position.set(origin[0] + size[0] / 2, origin[1] + size[1] / 2, (origin[2] || 0) - thickness / 2);
    this.iceBase.receiveShadow = true;
    this.root.add(this.iceBase);
  }

  updateParticles(particles) {
    const positions = typedValues(particles?.positions, Float32Array);
    const count = Math.min(positions ? Math.floor(positions.length / 3) : 0, 20000);
    if (count > this.particleCapacity) {
      if (this.particles) this.root.remove(this.particles);
      this.particleCapacity = Math.min(20000, Math.max(1024, 2 ** Math.ceil(Math.log2(count))));
      this.particles = new THREE.InstancedMesh(this.particleGeometry, this.particleMaterial, this.particleCapacity);
      this.particles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.particles.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(this.particleCapacity * 3), 3);
      this.particles.frustumCulled = false;
      this.particles.castShadow = true;
      this.root.add(this.particles);
    }
    if (!this.particles) return;
    const radii = typedValues(particles?.radii, Float32Array);
    const materials = typedValues(particles?.material_ids, Uint8Array);
    for (let index = 0; index < count; index += 1) {
      this.position.fromArray(positions, index * 3);
      const radius = Math.max(0.004, Math.min(0.045, radii?.[index] || 0.012));
      this.scale.setScalar(radius);
      this.matrix.compose(this.position, this.quaternion, this.scale);
      this.particles.setMatrixAt(index, this.matrix);
      const layer = this.layers.find((item) => Number(item.id) === Number(materials?.[index] ?? 0));
      this.particles.setColorAt(index, layerColor(layer));
    }
    this.particles.count = count;
    this.particles.instanceMatrix.needsUpdate = true;
    if (this.particles.instanceColor) this.particles.instanceColor.needsUpdate = true;
  }

  tick(now) {
    if (!this.heightMesh || !this.snowCap || !this.targetHeights || !this.fromHeights || !this.targetCapHeights || !this.fromCapHeights) return;
    const alpha = THREE.MathUtils.clamp((now - this.transitionStarted) / this.transitionMs, 0, 1);
    const eased = 1 - (1 - alpha) ** 3;
    const positions = this.heightMesh.geometry.attributes.position;
    for (let index = 0; index < this.targetHeights.length; index += 1) {
      positions.array[index * 3 + 2] = THREE.MathUtils.lerp(this.fromHeights[index], this.targetHeights[index], eased);
    }
    positions.needsUpdate = true;
    const capPositions = this.snowCap.geometry.attributes.position;
    for (let index = 0; index < this.targetCapHeights.length; index += 1) {
      capPositions.array[index * 3 + 2] = THREE.MathUtils.lerp(this.fromCapHeights[index], this.targetCapHeights[index], eased);
    }
    capPositions.needsUpdate = true;
    if (alpha < 1 || !this.normalsSettled) {
      this.heightMesh.geometry.computeVertexNormals();
      this.heightMesh.geometry.attributes.normal.needsUpdate = true;
      this.snowCap.geometry.computeVertexNormals();
      this.snowCap.geometry.attributes.normal.needsUpdate = true;
      if (alpha >= 1) this.normalsSettled = true;
    }
    this.normalTick += 1;
  }

  setVisible(visible) {
    this.root.visible = visible;
  }
}
