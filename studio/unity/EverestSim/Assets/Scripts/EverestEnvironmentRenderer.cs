using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace EverestSim
{
    public sealed class EverestEnvironmentRenderer : MonoBehaviour
    {
        private Light _sun;
        private ParticleSystem _snowfall;
        private Camera _camera;
        private EverestRobotRenderer _robot;
        private Material _skyMaterial;
        private GameObject _cloudVolume;
        private Material _cloudMaterial;

        public float TemperatureC { get; private set; }
        public float WindSpeedMS { get; private set; }
        public float WindDirectionDeg { get; private set; }
        public float SnowfallMMH { get; private set; }
        public float VisibilityScale { get; private set; } = 1f;
        public float CloudDensity { get; private set; } = 0.28f;
        public float CloudCoverage { get; private set; } = 0.42f;
        public float CloudRadiusM { get; private set; } = 120f;
        public float CloudAltitudeM { get; private set; } = 42f;
        public float CloudThicknessM { get; private set; } = 30f;
        public float CloudSpeed { get; private set; } = 0.35f;
        public float CloudQuality { get; private set; } = 0.55f;
        public bool MovementAllowed { get; private set; } = true;
        public bool CloudsReady => _cloudMaterial != null && _cloudMaterial.shader != null && _cloudMaterial.shader.isSupported;
        public bool SnowfallReady => _snowfall != null && _snowfall.GetComponent<ParticleSystemRenderer>()?.sharedMaterial != null;

        public void Initialize(EverestRobotRenderer robot)
        {
            _robot = robot;
        }

        private void Awake()
        {
            _camera = Camera.main;
            CreateLighting();
            CreateSky();
            CreateVolumetricClouds();
            CreateSnowfall();
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogColor = new Color(0.68f, 0.73f, 0.78f);
            ApplyAtmosphere();
        }

        private void CreateLighting()
        {
            var lightObject = new GameObject("Everest Sun");
            lightObject.transform.SetParent(transform, false);
            lightObject.transform.rotation = Quaternion.Euler(38f, -28f, 0f);
            _sun = lightObject.AddComponent<Light>();
            _sun.type = LightType.Directional;
            _sun.intensity = 1.05f;
            _sun.color = new Color(0.88f, 0.92f, 1f);
            _sun.shadows = LightShadows.Soft;
        }

        private void CreateSky()
        {
            var shader = Shader.Find("Skybox/Procedural");
            if (shader == null) return;
            _skyMaterial = new Material(shader) { name = "Everest Procedural Sky" };
            _skyMaterial.SetFloat("_SunDisk", 2f);
            _skyMaterial.SetFloat("_SunSize", 0.025f);
            _skyMaterial.SetFloat("_SunSizeConvergence", 5f);
            _skyMaterial.SetFloat("_AtmosphereThickness", 0.72f);
            _skyMaterial.SetColor("_SkyTint", new Color(0.34f, 0.54f, 0.78f));
            _skyMaterial.SetColor("_GroundColor", new Color(0.09f, 0.15f, 0.22f));
            _skyMaterial.SetFloat("_Exposure", 1.12f);
            RenderSettings.skybox = _skyMaterial;
            RenderSettings.sun = _sun;
            DynamicGI.UpdateEnvironment();
        }

        private void CreateVolumetricClouds()
        {
            _cloudMaterial = EverestRuntimeMaterials.Load("EverestVolumetricClouds", "Everest/VolumetricClouds");
            if (_cloudMaterial == null) return;

            _cloudVolume = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _cloudVolume.name = "Active Volumetric Cloud Volume";
            _cloudVolume.transform.SetParent(transform, false);
            var collider = _cloudVolume.GetComponent<Collider>();
            if (collider != null) Destroy(collider);

            var renderer = _cloudVolume.GetComponent<MeshRenderer>();
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            renderer.sharedMaterial = _cloudMaterial;
        }

        private void CreateSnowfall()
        {
            var go = new GameObject("Atmospheric Snowfall (visual only)");
            go.transform.SetParent(transform, false);
            _snowfall = go.AddComponent<ParticleSystem>();

            var main = _snowfall.main;
            main.loop = true;
            main.startLifetime = 5.5f;
            main.startSpeed = 0f;
            main.startSize = new ParticleSystem.MinMaxCurve(0.018f, 0.055f);
            main.startColor = new Color(0.96f, 0.98f, 1f, 0.9f);
            main.maxParticles = 7000;
            main.simulationSpace = ParticleSystemSimulationSpace.World;

            var shape = _snowfall.shape;
            shape.shapeType = ParticleSystemShapeType.Box;

            var emission = _snowfall.emission;
            emission.rateOverTime = 0f;

            var velocity = _snowfall.velocityOverLifetime;
            velocity.enabled = true;
            velocity.space = ParticleSystemSimulationSpace.World;
            velocity.y = -3.5f;

            var noise = _snowfall.noise;
            noise.enabled = true;
            noise.strength = 0.55f;
            noise.frequency = 0.25f;
            noise.scrollSpeed = 0.4f;

            var renderer = go.GetComponent<ParticleSystemRenderer>();
            var material = EverestRuntimeMaterials.Load("EverestSnowfall", "Everest/Snowfall");
            if (material != null)
            {
                renderer.sharedMaterial = material;
                renderer.renderMode = ParticleSystemRenderMode.Stretch;
                renderer.velocityScale = 0.015f;
                renderer.lengthScale = 0.12f;
                renderer.cameraVelocityScale = 0f;
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
            }
            _snowfall.Play();
        }

        public void OnEnvironment(JObject environment)
        {
            TemperatureC = environment.Value<float?>("temperature_c") ?? TemperatureC;
            WindSpeedMS = environment.Value<float?>("wind_speed_m_s") ?? WindSpeedMS;
            WindDirectionDeg = environment.Value<float?>("wind_direction_deg") ?? WindDirectionDeg;
            SnowfallMMH = environment.Value<float?>("snowfall_mm_h") ?? SnowfallMMH;
            VisibilityScale = Mathf.Clamp01(environment.Value<float?>("visibility_scale") ?? VisibilityScale);
            CloudDensity = Mathf.Clamp01(environment.Value<float?>("cloud_density") ?? CloudDensity);
            CloudCoverage = Mathf.Clamp01(environment.Value<float?>("cloud_coverage") ?? CloudCoverage);
            CloudRadiusM = Mathf.Clamp(environment.Value<float?>("cloud_radius_m") ?? CloudRadiusM, 15f, 600f);
            CloudAltitudeM = Mathf.Clamp(environment.Value<float?>("cloud_altitude_m") ?? CloudAltitudeM, 5f, 300f);
            CloudThicknessM = Mathf.Clamp(environment.Value<float?>("cloud_thickness_m") ?? CloudThicknessM, 5f, 180f);
            CloudSpeed = Mathf.Clamp(environment.Value<float?>("cloud_speed") ?? CloudSpeed, 0f, 2f);
            CloudQuality = Mathf.Clamp01(environment.Value<float?>("cloud_quality") ?? CloudQuality);
            MovementAllowed = environment.Value<bool?>("movement_allowed") ?? MovementAllowed;
            ApplyAtmosphere();
        }

        private void ApplyAtmosphere()
        {
            RenderSettings.fogDensity = Mathf.Lerp(0.0025f, 0.00008f, VisibilityScale);
            RenderSettings.fogColor = Color.Lerp(
                new Color(0.30f, 0.39f, 0.50f),
                new Color(0.63f, 0.76f, 0.88f),
                VisibilityScale);
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = Color.Lerp(
                new Color(0.20f, 0.29f, 0.42f),
                new Color(0.52f, 0.67f, 0.82f),
                VisibilityScale);
            RenderSettings.ambientEquatorColor = Color.Lerp(
                new Color(0.14f, 0.20f, 0.28f),
                new Color(0.36f, 0.46f, 0.56f),
                VisibilityScale);
            RenderSettings.ambientGroundColor = new Color(0.07f, 0.10f, 0.14f);

            if (_skyMaterial != null)
            {
                _skyMaterial.SetFloat("_Exposure", Mathf.Lerp(0.72f, 1.14f, VisibilityScale));
                _skyMaterial.SetFloat("_AtmosphereThickness", Mathf.Lerp(1.05f, 0.68f, VisibilityScale));
            }
            if (_sun != null) _sun.intensity = Mathf.Lerp(0.42f, 1.08f, VisibilityScale);

            var radians = WindDirectionDeg * Mathf.Deg2Rad;
            var visualDirection = new Vector4(Mathf.Sin(radians), 0f, Mathf.Cos(radians), 0f);
            Shader.SetGlobalVector("_EverestWindDir", visualDirection);
            Shader.SetGlobalFloat("_EverestWindStrength", Mathf.Clamp01(WindSpeedMS / 25f));

            if (_cloudMaterial != null)
            {
                _cloudMaterial.SetFloat("_CloudRadius", CloudRadiusM);
                _cloudMaterial.SetFloat("_CloudThickness", CloudThicknessM);
                _cloudMaterial.SetFloat("_CloudDensity", CloudDensity);
                _cloudMaterial.SetFloat("_CloudCoverage", CloudCoverage);
                _cloudMaterial.SetFloat("_CloudSpeed", CloudSpeed);
                _cloudMaterial.SetFloat("_CloudQuality", CloudQuality);
                _cloudMaterial.SetVector("_CloudWind", visualDirection);
            }
            if (_cloudVolume != null)
                _cloudVolume.transform.localScale = Vector3.one * (CloudRadiusM * 2f);

            if (_snowfall != null)
            {
                var emission = _snowfall.emission;
                emission.rateOverTime = Mathf.Clamp(SnowfallMMH * 38f, 0f, 1800f);
                var visualWind = Mathf.Min(12f, WindSpeedMS * 0.45f);
                var velocity = _snowfall.velocityOverLifetime;
                velocity.x = Mathf.Sin(radians) * visualWind;
                velocity.z = Mathf.Cos(radians) * visualWind;
                velocity.y = -Mathf.Lerp(2.2f, 5.0f, Mathf.Clamp01(WindSpeedMS / 25f));
                var shape = _snowfall.shape;
                var snowRadius = Mathf.Clamp(CloudRadiusM * 0.28f, 18f, 70f);
                shape.scale = new Vector3(snowRadius * 2f, 10f, snowRadius * 2f);
            }
        }

        private void LateUpdate()
        {
            if (_camera == null) _camera = Camera.main;
            Vector3 anchor;
            if (_robot != null && _robot.TryGetBodyPosition("pelvis", out var pelvis)) anchor = pelvis;
            else if (_camera != null) anchor = _camera.transform.position;
            else anchor = Vector3.zero;

            if (_cloudVolume != null)
            {
                var center = anchor + Vector3.up * CloudAltitudeM;
                _cloudVolume.transform.position = center;
                if (_cloudMaterial != null) _cloudMaterial.SetVector("_CloudCenter", new Vector4(center.x, center.y, center.z, 1f));
            }
            if (_camera != null && _snowfall != null)
            {
                var position = _camera.transform.position;
                position.y += 9f;
                _snowfall.transform.position = position;
            }
        }

        private void OnDestroy()
        {
            if (_skyMaterial != null) Destroy(_skyMaterial);
            if (_cloudMaterial != null) Destroy(_cloudMaterial);
        }
    }
}
