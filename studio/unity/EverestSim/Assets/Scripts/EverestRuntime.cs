using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EverestSim
{
    [DefaultExecutionOrder(-1000)]
    public sealed class EverestRuntime : MonoBehaviour
    {
        private EverestBackendClient _backend;
        private EverestRobotRenderer _robot;
        private EverestTerrainRenderer _terrain;
        private EverestVisualTerrainRenderer _visualTerrain;
        private EverestSnowRenderer _snow;
        private EverestEnvironmentRenderer _environment;
        private EverestDemoRenderer _demo;
        private EverestTrainingSubscene _trainingSubscene;
        private EverestCameraController _camera;
        private EverestEditorHud _hud;

        private Vector3 _lastCommand = new Vector3(float.NaN, float.NaN, float.NaN);
        private float _nextCommandTime;
        private bool _uiInputBlockedLastFrame;
        private bool _paused = true;
        private string _dataMode = "sim";
        private bool _demoActive;
        private Vector3 _pointerDownPosition;
        private bool _pointerCandidate;
        private GameObject _navigationMarker;

        public bool ManualControlEnabled { get; private set; }
        public bool CheatModeEnabled { get; private set; }
        public bool Paused => _paused;
        public bool LiveReadOnly => _dataMode == "live";
        public EverestCameraController CameraController => _camera;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Bootstrap()
        {
            if (FindObjectOfType<EverestRuntime>() != null) return;
            var go = new GameObject("Everest Unity Renderer");
            DontDestroyOnLoad(go);
            go.AddComponent<EverestRuntime>();
        }

        private void Awake()
        {
            _robot = gameObject.AddComponent<EverestRobotRenderer>();
            _terrain = gameObject.AddComponent<EverestTerrainRenderer>();
            _visualTerrain = gameObject.AddComponent<EverestVisualTerrainRenderer>();
            _snow = gameObject.AddComponent<EverestSnowRenderer>();
            _camera = gameObject.AddComponent<EverestCameraController>();
            _camera.Initialize(_robot);
            _environment = gameObject.AddComponent<EverestEnvironmentRenderer>();
            _environment.Initialize(_robot);
            _demo = gameObject.AddComponent<EverestDemoRenderer>();
            _trainingSubscene = gameObject.AddComponent<EverestTrainingSubscene>();
            _backend = gameObject.AddComponent<EverestBackendClient>();
            _hud = gameObject.AddComponent<EverestEditorHud>();
            _hud.Initialize(_backend, _robot, _snow, _terrain, _visualTerrain, _environment, _camera, _trainingSubscene, this);
            _camera.SetEditorHud(_hud);

            _backend.SceneReceived += _robot.OnScene;
            _backend.SceneReceived += _trainingSubscene.OnScene;
            _backend.FrameReceived += _robot.OnFrame;
            _backend.FrameReceived += OnFrame;
            _backend.TerrainReceived += _terrain.OnLocalTerrain;
            _backend.TerrainReceived += _visualTerrain.OnLocalTerrain;
            _backend.MacroTerrainReceived += _terrain.OnMacroTerrain;
            _backend.MacroTerrainReceived += _visualTerrain.OnMacroTerrain;
            _backend.SnowReceived += _snow.OnSnow;
            _backend.SnowHistoryReceived += _snow.OnSnowHistory;
            _backend.SnowReceived += _visualTerrain.OnSnow;
            _backend.EnvironmentReceived += _environment.OnEnvironment;
            _backend.EnvironmentReceived += _visualTerrain.OnEnvironment;
            _backend.StateReceived += OnState;
            _backend.StateReceived += _demo.OnState;
            _backend.SubsetViewReceived += _trainingSubscene.OnSubsetView;
            _backend.FaultReceived += OnFault;
            CreateNavigationMarker();
        }

        public void SetManualControl(bool enabled)
        {
            if (LiveReadOnly) return;
            ManualControlEnabled = enabled;
            _backend?.SendManualForceMode(enabled);
            if (enabled) _backend?.SendPause(false);
            _lastCommand = new Vector3(float.NaN, float.NaN, float.NaN);
            if (!enabled && _backend != null)
                _backend.SendCommand(0f, 0f, 0f);
        }

        public void SetCheatMode(bool enabled)
        {
            if (LiveReadOnly) return;
            CheatModeEnabled = enabled;
            if (enabled) _backend?.SendManualForceMode(false);
            if (enabled) ManualControlEnabled = true;
            _lastCommand = new Vector3(float.NaN, float.NaN, float.NaN);
            _backend?.SendCheatMode(enabled);
            if (enabled) _backend?.SendPause(false);
            else _backend?.SendCommand(0f, 0f, 0f);
        }

        private void OnState(JObject state)
        {
            var nextMode = state.Value<string>("data_mode") ?? _dataMode;
            if (nextMode == "live" && _dataMode != "live")
            {
                ManualControlEnabled = false;
                CheatModeEnabled = false;
                _lastCommand = Vector3.zero;
            }
            _dataMode = nextMode;
            var source = state["source"] as JObject;
            if (source != null)
            {
                var epoch = source.Value<long?>("epoch") ?? -1;
                _robot?.SetSourceEpoch(epoch);
                _snow?.SetSourceEpoch(epoch);
            }
            _paused = state.Value<bool?>("paused") ?? _paused;
            _demoActive = state["demo"]?.Value<bool?>("active") == true;
            CheatModeEnabled = state.Value<bool?>("cheat_mode") ?? CheatModeEnabled;
            _snow?.SetVisualOnly(state["newton"]?.Value<bool?>("visual_only") == true);
            SyncNavigationMarker(state["navigation"] as JObject);
        }

        private void OnFrame(JObject frame)
        {
            _snow?.OnFeet(frame["feet"] as JObject);
        }

        private static void OnFault(JObject fault)
        {
            Debug.LogError($"Everest backend fault: {fault.Value<string>("message")}");
        }

        private void Update()
        {
            if (_backend == null) return;

            var uiBlocked = _hud != null && _hud.BlocksSceneInput;
            if (uiBlocked)
            {
                if (!_uiInputBlockedLastFrame && (ManualControlEnabled || CheatModeEnabled))
                {
                    _backend.SendCommand(0f, 0f, 0f);
                    _lastCommand = Vector3.zero;
                    _nextCommandTime = Time.unscaledTime + 0.05f;
                }
                _uiInputBlockedLastFrame = true;
                return;
            }
            _uiInputBlockedLastFrame = false;

            if (LiveReadOnly) return;

            HandleTerrainNavigationClick();
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                _backend.SendNavigationCancel();
                if (_navigationMarker != null) _navigationMarker.SetActive(false);
            }

            if (Input.GetKeyDown(KeyCode.P))
                _backend.SendPause(!_paused);
            if (Input.GetKeyDown(KeyCode.R) && !(_camera != null && _camera.FreeMoveModifierHeld))
                _backend.SendReset();

            if (!ManualControlEnabled && !CheatModeEnabled) return;
            if (_camera != null && _camera.FreeMoveModifierHeld) return;

            var forward = 0f;
            var lateral = 0f;
            var yaw = 0f;
            if (CheatModeEnabled)
            {
                // Direct floating-base transport. ASDF is supported explicitly:
                // A/D strafe, S/F back/forward; WASD also remains convenient.
                if (Input.GetKey(KeyCode.F) || Input.GetKey(KeyCode.W)) forward += 1.0f;
                if (Input.GetKey(KeyCode.S)) forward -= 1.0f;
                if (Input.GetKey(KeyCode.A)) lateral += 1.0f;
                if (Input.GetKey(KeyCode.D)) lateral -= 1.0f;
                if (Input.GetKey(KeyCode.Q)) yaw += 1.0f;
                if (Input.GetKey(KeyCode.E)) yaw -= 1.0f;
            }
            else
            {
                // Manual control applies a physical pelvis force/torque while
                // the standing policy holds a neutral command.
                if (Input.GetKey(KeyCode.W)) forward += 1.0f;
                if (Input.GetKey(KeyCode.S)) forward -= 1.0f;
                if (Input.GetKey(KeyCode.A)) yaw += 1.0f;
                if (Input.GetKey(KeyCode.D)) yaw -= 1.0f;
            }
            if (Input.GetKey(KeyCode.Space)) forward = lateral = yaw = 0f;

            if (!CheatModeEnabled && _environment != null && !_environment.MovementAllowed)
                forward = lateral = yaw = 0f;

            var command = new Vector3(forward, lateral, yaw);
            if (Time.unscaledTime >= _nextCommandTime && (command - _lastCommand).sqrMagnitude > 1e-6f)
            {
                _backend.SendCommand(command.x, command.y, command.z);
                _lastCommand = command;
                _nextCommandTime = Time.unscaledTime + 0.05f;
            }
        }

        private void HandleTerrainNavigationClick()
        {
            if (_demoActive) return;
            if (Input.GetMouseButtonDown(0))
            {
                _pointerDownPosition = Input.mousePosition;
                _pointerCandidate = true;
            }
            if (!Input.GetMouseButtonUp(0) || !_pointerCandidate) return;
            _pointerCandidate = false;
            if ((Input.mousePosition - _pointerDownPosition).sqrMagnitude > 36f) return;
            if (_camera == null || Camera.main == null) return;

            var ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            Vector3 destination;
            if (Physics.Raycast(ray, out var hit, 30000f)
                && hit.collider.GetComponent<EverestTerrainClickSurface>() != null)
                destination = hit.point;
            else if (_visualTerrain != null && _visualTerrain.TryPickTerrain(ray, out var sampled))
                destination = sampled;
            else
                return;
            ManualControlEnabled = false;
            CheatModeEnabled = false;
            _lastCommand = Vector3.zero;
            _backend.SendNavigationTarget(destination);
            _backend.SendPause(false);
            if (_navigationMarker != null)
            {
                _navigationMarker.transform.position = destination + Vector3.up * 0.06f;
                _navigationMarker.SetActive(true);
            }
        }

        private void CreateNavigationMarker()
        {
            _navigationMarker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            _navigationMarker.name = "Straight-line navigation destination";
            _navigationMarker.transform.SetParent(transform, false);
            _navigationMarker.transform.localScale = new Vector3(0.24f, 0.018f, 0.24f);
            var collider = _navigationMarker.GetComponent<Collider>();
            if (collider != null) Destroy(collider);
            var renderer = _navigationMarker.GetComponent<MeshRenderer>();
            var shader = Shader.Find("Unlit/Color") ?? Shader.Find("Sprites/Default");
            if (renderer != null && shader != null)
            {
                var material = new Material(shader) { color = new Color(0.05f, 0.82f, 1f, 0.92f) };
                renderer.sharedMaterial = material;
            }
            _navigationMarker.SetActive(false);
        }

        private void SyncNavigationMarker(JObject navigation)
        {
            if (_navigationMarker == null || navigation == null) return;
            var active = navigation.Value<bool?>("active") == true;
            var target = navigation["target"];
            if (active && target != null && target.HasValues)
            {
                _navigationMarker.transform.position = EverestCoordinates.Position(target) + Vector3.up * 0.06f;
                _navigationMarker.SetActive(true);
            }
            else if (!active)
            {
                _navigationMarker.SetActive(false);
            }
        }
    }
}
