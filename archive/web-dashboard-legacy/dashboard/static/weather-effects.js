import * as THREE from 'three';

function makeSnowflakeTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  const gradient = context.createRadialGradient(32, 32, 1, 32, 32, 30);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.28, 'rgba(247,251,255,.96)');
  gradient.addColorStop(0.68, 'rgba(224,236,255,.28)');
  gradient.addColorStop(1, 'rgba(224,236,255,0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, 64, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export class WeatherEffects {
  constructor(targetScene) {
    this.capacity = 3200;
    this.spindriftCapacity = 600;
    this.snowfallMmH = 2.5;
    this.windSpeed = 18;
    this.windDirectionDeg = 245;
    this.liveSnowfallMmH = 0;
    this.lastTick = performance.now();
    this.snowflakeTexture = makeSnowflakeTexture();

    this.positions = new Float32Array(this.capacity * 3);
    this.phase = new Float32Array(this.capacity);
    this.fallSpeed = new Float32Array(this.capacity);
    this.snowGeometry = new THREE.BufferGeometry();
    this.snowGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage),
    );
    this.snowMaterial = new THREE.PointsMaterial({
      color: 0xf4f8ff,
      map: this.snowflakeTexture,
      alphaMap: this.snowflakeTexture,
      size: 0.085,
      transparent: true,
      opacity: 0.82,
      alphaTest: 0.025,
      depthWrite: false,
      sizeAttenuation: true,
      fog: true,
    });
    this.snow = new THREE.Points(this.snowGeometry, this.snowMaterial);
    this.snow.name = 'weather-snowfall';
    this.snow.frustumCulled = false;
    this.snow.renderOrder = 4;
    targetScene.add(this.snow);

    this.spindriftPositions = new Float32Array(this.spindriftCapacity * 3);
    this.spindriftGeometry = new THREE.BufferGeometry();
    this.spindriftGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(this.spindriftPositions, 3).setUsage(THREE.DynamicDrawUsage),
    );
    this.spindriftMaterial = new THREE.PointsMaterial({
      color: 0xdcecff,
      map: this.snowflakeTexture,
      alphaMap: this.snowflakeTexture,
      size: 0.2,
      transparent: true,
      opacity: 0.13,
      alphaTest: 0.01,
      depthWrite: false,
      sizeAttenuation: true,
      fog: true,
    });
    this.spindrift = new THREE.Points(this.spindriftGeometry, this.spindriftMaterial);
    this.spindrift.name = 'weather-spindrift';
    this.spindrift.frustumCulled = false;
    this.spindrift.renderOrder = 3;
    targetScene.add(this.spindrift);

    for (let index = 0; index < this.capacity; index += 1) this.respawnSnow(index, true);
    for (let index = 0; index < this.spindriftCapacity; index += 1) this.respawnSpindrift(index, true);
  }

  respawnSnow(index, initial = false) {
    const offset = index * 3;
    this.positions[offset] = (Math.random() - 0.5) * 16;
    this.positions[offset + 1] = initial ? 0.2 + Math.random() * 8.5 : 7 + Math.random() * 2.5;
    this.positions[offset + 2] = 4 - Math.random() * 25;
    this.phase[index] = Math.random() * Math.PI * 2;
    this.fallSpeed[index] = 0.65 + Math.random() * 1.45;
  }

  respawnSpindrift(index, initial = false) {
    const offset = index * 3;
    this.spindriftPositions[offset] = (Math.random() - 0.5) * 13;
    this.spindriftPositions[offset + 1] = 0.05 + Math.random() * 0.8;
    this.spindriftPositions[offset + 2] = initial ? 3 - Math.random() * 24 : 3.5;
  }

  setEnvironment(parameters = {}) {
    this.snowfallMmH = Math.max(0, Number(parameters.snowfall_mm_h ?? this.snowfallMmH));
    this.windSpeed = Math.max(0, Number(parameters.wind_speed_m_s ?? this.windSpeed));
    this.windDirectionDeg = Number(parameters.wind_direction_deg ?? this.windDirectionDeg);
  }

  setLiveWeather(weather) {
    const conditions = weather?.conditions || {};
    this.liveSnowfallMmH = Math.max(0, Number(conditions.snowfall_cm || 0) * 10);
  }

  tick(now) {
    const dt = Math.min(0.04, Math.max(0.001, (now - this.lastTick) / 1000));
    this.lastTick = now;
    // The visible flakes mirror the same dashboard snowfall-depth rate that is
    // sent to Newton MPM. Live weather remains available as context, but must
    // not silently make the visual storm stronger than the physical mass flux.
    const snowfall = this.snowfallMmH;
    const intensity = THREE.MathUtils.clamp(
      0.06 + snowfall / 14 + this.windSpeed / 180,
      0.06,
      1,
    );
    const count = Math.max(180, Math.round(this.capacity * intensity));
    const driftCount = Math.max(
      90,
      Math.round(
        this.spindriftCapacity
        * THREE.MathUtils.clamp(this.windSpeed / 34, 0.15, 1),
      ),
    );
    this.snowGeometry.setDrawRange(0, count);
    this.spindriftGeometry.setDrawRange(0, driftCount);

    const direction = THREE.MathUtils.degToRad(this.windDirectionDeg);
    const windX = Math.sin(direction) * this.windSpeed * 0.055;
    const windZ = Math.cos(direction) * this.windSpeed * 0.055;
    for (let index = 0; index < count; index += 1) {
      const offset = index * 3;
      const flutter = Math.sin(now * 0.0018 + this.phase[index]) * 0.15;
      this.positions[offset] += (windX + flutter) * dt;
      this.positions[offset + 1] -= this.fallSpeed[index] * dt;
      this.positions[offset + 2] += (windZ + flutter * 0.35) * dt;
      if (
        this.positions[offset + 1] < 0.02
        || Math.abs(this.positions[offset]) > 10
        || this.positions[offset + 2] < -23
        || this.positions[offset + 2] > 6
      ) {
        this.respawnSnow(index);
      }
    }
    this.snowGeometry.attributes.position.needsUpdate = true;

    const groundWindX = Math.sin(direction) * (1.2 + this.windSpeed * 0.12);
    const groundWindZ = Math.cos(direction) * (1.2 + this.windSpeed * 0.12);
    for (let index = 0; index < driftCount; index += 1) {
      const offset = index * 3;
      this.spindriftPositions[offset] += groundWindX * dt;
      this.spindriftPositions[offset + 1] += Math.sin(now * 0.002 + index) * 0.055 * dt;
      this.spindriftPositions[offset + 2] += groundWindZ * dt;
      if (
        Math.abs(this.spindriftPositions[offset]) > 9
        || this.spindriftPositions[offset + 2] < -24
        || this.spindriftPositions[offset + 2] > 6
      ) {
        this.respawnSpindrift(index);
      }
    }
    this.spindriftGeometry.attributes.position.needsUpdate = true;
  }
}
