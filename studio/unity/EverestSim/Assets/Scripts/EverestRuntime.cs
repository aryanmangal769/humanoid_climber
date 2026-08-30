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
        private EverestCameraController _camera;
        private EverestEditorHud _hud;

        private Vector3 _lastCommand = new Vector3(float.NaN, float.NaN, float.NaN);
        private float _nextCommandTime;
        private bool _uiInputBlockedLastFrame;
        private bool _paused = true;
        private string _dataMode = "sim";

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
            _backend = gameObject.AddComponent<EverestBackendClient>();
            _hud = gameObject.AddComponent<EverestEditorHud>();
            _hud.Initialize(_backend, _robot, _snow, _terrain, _visualTerrain, _environment, _camera, this);
            _camera.SetEditorHud(_hud);

            _backend.SceneReceived += _robot.OnScene;
            _backend.FrameReceived += _robot.OnFrame;
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
            _backend.FaultReceived += OnFault;
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
            CheatModeEnabled = state.Value<bool?>("cheat_mode") ?? CheatModeEnabled;
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
    }
}
