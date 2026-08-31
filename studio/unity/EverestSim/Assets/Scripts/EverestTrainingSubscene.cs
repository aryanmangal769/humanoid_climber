using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EverestSim
{
    /// <summary>
    /// Isolated Unity rendering of the captured failure subset. The backend
    /// supplies the Newton terrain slice, snow/weather state, and replay robot
    /// poses; this component never drives the main simulation.
    /// </summary>
    public sealed class EverestTrainingSubscene : MonoBehaviour
    {
        private const int TrainingLayer = 30;
        private GameObject _root;
        private EverestRobotRenderer _robot;
        private EverestSnowRenderer _snow;
        private Camera _camera;
        private Light _light;
        private RenderTexture _target;
        private JObject _scene;

        public RenderTexture Texture => _target;
        public bool Ready { get; private set; }
        public string Caption { get; private set; } = "Waiting for captured failure subset";

        private void Awake()
        {
            _root = new GameObject("RL Training Unity Subscene");
            _root.transform.SetParent(transform, false);

            var robotRoot = new GameObject("Training Robot");
            robotRoot.transform.SetParent(_root.transform, false);
            _robot = robotRoot.AddComponent<EverestRobotRenderer>();

            var snowRoot = new GameObject("Captured Newton Terrain Slice");
            snowRoot.transform.SetParent(_root.transform, false);
            _snow = snowRoot.AddComponent<EverestSnowRenderer>();

            var cameraObject = new GameObject("RL Training Camera");
            cameraObject.transform.SetParent(_root.transform, false);
            _camera = cameraObject.AddComponent<Camera>();
            _camera.clearFlags = CameraClearFlags.SolidColor;
            _camera.backgroundColor = new Color(0.055f, 0.085f, 0.12f);
            _camera.cullingMask = 1 << TrainingLayer;
            _camera.nearClipPlane = 0.03f;
            _camera.farClipPlane = 120f;
            _camera.fieldOfView = 52f;
            _target = new RenderTexture(960, 540, 24, RenderTextureFormat.ARGB32)
            {
                name = "Everest RL Training Subscene"
            };
            _target.Create();
            _camera.targetTexture = _target;

            var lightObject = new GameObject("Training Sun");
            lightObject.transform.SetParent(_root.transform, false);
            _light = lightObject.AddComponent<Light>();
            _light.type = LightType.Directional;
            _light.intensity = 1.15f;
            _light.color = new Color(0.78f, 0.88f, 1.0f);
            _light.cullingMask = 1 << TrainingLayer;
            lightObject.transform.rotation = Quaternion.Euler(48f, -32f, 0f);

            if (Camera.main != null)
                Camera.main.cullingMask &= ~(1 << TrainingLayer);
            SetLayerRecursively(_root, TrainingLayer);
        }

        public void OnScene(JObject scene)
        {
            _scene = scene;
            _robot.OnScene(scene);
            SetLayerRecursively(_root, TrainingLayer);
        }

        public void OnSubsetView(JObject subset)
        {
            var terrain = subset?["terrain"] as JObject;
            var frame = subset?["robot_frame"] as JObject;
            if (terrain == null || frame == null) return;
            if (_scene != null && _robot.Sequence < 0) _robot.OnScene(_scene);
            _snow.OnSnow(terrain);
            _robot.OnFrame(frame);
            SetLayerRecursively(_root, TrainingLayer);

            var origin = terrain["origin"] as JArray;
            var size = terrain["size"] as JArray;
            var center = Vector3.zero;
            if (origin != null && size != null && origin.Count >= 2 && size.Count >= 2)
            {
                var x = origin[0].Value<float>() + size[0].Value<float>() * 0.5f;
                var z = origin[1].Value<float>() + size[1].Value<float>() * 0.5f;
                var heights = terrain["heights"] as JArray;
                var y = heights != null && heights.Count > 0 ? heights[heights.Count / 2].Value<float>() : 0f;
                center = new Vector3(x, y + 0.55f, z);
            }
            if (_robot.TryGetBodyPosition("pelvis", out var pelvis)) center = pelvis;
            _camera.transform.position = center + new Vector3(3.2f, 2.0f, 3.2f);
            _camera.transform.rotation = Quaternion.LookRotation(center - _camera.transform.position, Vector3.up);

            var weather = subset["weather"] as JObject;
            var visibility = weather?.Value<float?>("visibility_scale") ?? 0.8f;
            var snowfall = weather?.Value<float?>("snowfall_mm_h") ?? 0f;
            _camera.backgroundColor = Color.Lerp(
                new Color(0.035f, 0.055f, 0.08f),
                new Color(0.30f, 0.42f, 0.55f),
                Mathf.Clamp01(visibility));
            _light.intensity = Mathf.Lerp(0.55f, 1.25f, Mathf.Clamp01(visibility));
            var payloadCaption = subset.Value<string>("caption");
            Caption = !string.IsNullOrWhiteSpace(payloadCaption)
                ? payloadCaption
                : $"Live Newton + MuJoCo subset · snowfall {snowfall:0.0} mm/h";
            Ready = true;
        }

        private static void SetLayerRecursively(GameObject item, int layer)
        {
            if (item == null) return;
            item.layer = layer;
            foreach (Transform child in item.transform)
                SetLayerRecursively(child.gameObject, layer);
        }

        private void OnDestroy()
        {
            if (_target == null) return;
            _target.Release();
            Destroy(_target);
        }
    }
}
