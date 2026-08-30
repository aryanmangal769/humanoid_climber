using System;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EverestSim
{
    /// <summary>
    /// Compact simulation-editor shell. High-frequency global actions stay in
    /// the command bar; every tuning control lives in one persistent dock.
    /// </summary>
    public sealed class EverestEditorHud : MonoBehaviour
    {
        private sealed class LayerDraft
        {
            public string Type = "POWDER";
            public string Label = "Fresh powder";
            public float Thickness = 0.08f;
            public float Density = 120f;
            public float Stiffness = 35000f;
            public float Compressive = 3500f;
            public float Shear = 1800f;
            public float Hardening = 12f;
            public float Bond = 2500f;

            public JObject ToJson()
            {
                JArray color;
                if (Type == "ICE") color = new JArray(0.24, 0.64, 0.86);
                else if (Type == "FIRN") color = new JArray(0.62, 0.77, 0.88);
                else if (Type == "CRUST") color = new JArray(0.79, 0.89, 0.97);
                else if (Type == "WIND_PACK") color = new JArray(0.84, 0.91, 0.97);
                else color = new JArray(0.95, 0.98, 1.0);
                return new JObject
                {
                    ["type"] = Type,
                    ["label"] = Label,
                    ["color"] = color,
                    ["thickness_m"] = Thickness,
                    ["density_kg_m3"] = Density,
                    ["stiffness_pa"] = Stiffness,
                    ["compressive_strength_pa"] = Compressive,
                    ["shear_strength_pa"] = Shear,
                    ["compaction_hardening"] = Hardening,
                    ["bond_strength_below_pa"] = Bond
                };
            }
        }

        private EverestBackendClient _backend;
        private EverestRobotRenderer _robot;
        private EverestSnowRenderer _snow;
        private EverestTerrainRenderer _terrain;
        private EverestVisualTerrainRenderer _visualTerrain;
        private EverestEnvironmentRenderer _environment;
        private EverestCameraController _camera;
        private EverestRuntime _runtime;

        private GUIStyle _chrome;
        private GUIStyle _dock;
        private GUIStyle _section;
        private GUIStyle _title;
        private GUIStyle _sectionTitle;
        private GUIStyle _label;
        private GUIStyle _muted;
        private GUIStyle _tiny;
        private GUIStyle _button;
        private GUIStyle _buttonActive;
        private GUIStyle _buttonDanger;
        private GUIStyle _good;
        private GUIStyle _warn;
        private GUIStyle _metric;
        private Texture2D _chromeTex;
        private Texture2D _dockTex;
        private Texture2D _sectionTex;
        private Texture2D _buttonTex;
        private Texture2D _buttonHoverTex;
        private Texture2D _accentTex;
        private Texture2D _dangerTex;
        private Texture2D _metricTex;
        private Texture2D _splitterTex;

        private Vector2 _scroll;
        private Vector2 _demoScroll;
        private bool _draftInitialized;
        private string _surface = "snow";
        private string _dataMode = "sim";
        private float _temperature = -18f;
        private float _windSpeed = 9f;
        private float _windDirection = 250f;
        private float _snowfall = 1.5f;
        private float _visibility = 0.88f;
        private float _cloudDensity = 0.42f;
        private float _cloudCoverage = 0.58f;
        private float _cloudRadius = 170f;
        private float _cloudAltitude = 38f;
        private float _cloudThickness = 46f;
        private float _cloudSpeed = 0.40f;
        private float _cloudQuality = 0.55f;
        private float _surfaceFriction = 0.36f;
        private float _iceFriction = 0.08f;
        private float _rockFriction = 0.82f;
        private float _physicsRadius = 1.25f;
        private float _mpmVoxelSize = 0.10f;
        private int _physicsDetailCells = 24;
        private float _mpmCouplingHz = 3f;
        private float _mpmContactRefineRadius = 0.55f;
        private int _mpmCoarseStride = 4;
        private float _patchRecenter = 0.70f;
        private bool _snowAccumulationEnabled = true;
        private float _weatherTimeScale = 1f;
        private float _cheatSpeed = 1.6f;
        private float _cheatYawRate = 1.4f;
        private float _uiScale = 1.6f;
        private float _dockWidthRatio = 0.37f;
        private float _currentDockWidth = 340f;
        private bool _dockDragging;
        private bool _compactDock;
        private int _localLod = 1;
        private int _macroLod = 7;
        private int _selectedLayer;
        private int _dockTab;
        private Texture2D _subsetTexture;
        private string _subsetCaption = "Subset preview disabled";
        private string _checkpointReturnPath = "";
        private readonly List<LayerDraft> _layers = new List<LayerDraft>();

        private static readonly string[] LayerTypes = { "POWDER", "WIND_PACK", "CRUST", "DENSE_SNOW", "FIRN", "ICE" };
        private static readonly string[] CompactLayerTypes = { "POW", "WIND", "CRUST", "DENSE", "FIRN", "ICE" };
        private const string UiScalePlayerPref = "EverestSim.EditorUiScaleV4";
        private const string DockWidthPlayerPref = "EverestSim.EditorDockWidthV1";
        private const float TopHeight = 36f;
        private const float BottomHeight = 22f;
        private const float SplitterWidth = 9f;

        public bool BlocksSceneInput
        {
            get
            {
                if (!isActiveAndEnabled) return false;
                var virtualWidth = Screen.width / Mathf.Max(_uiScale, 0.01f);
                var virtualHeight = Screen.height / Mathf.Max(_uiScale, 0.01f);
                var dockWidth = ResolveDockWidth(virtualWidth);
                var point = ScreenToVirtual(Input.mousePosition);
                var command = new Rect(0f, 0f, virtualWidth, TopHeight);
                var dock = new Rect(virtualWidth - dockWidth, TopHeight, dockWidth, Mathf.Max(0f, virtualHeight - TopHeight - BottomHeight));
                var splitter = new Rect(dock.x - SplitterWidth * 0.5f, dock.y, SplitterWidth, dock.height);
                var status = new Rect(0f, Mathf.Max(0f, virtualHeight - BottomHeight), virtualWidth, BottomHeight);
                return _dockDragging || GUIUtility.hotControl != 0 || command.Contains(point) || dock.Contains(point) || splitter.Contains(point) || status.Contains(point);
            }
        }

        public Rect SceneViewportNormalized
        {
            get
            {
                var scale = Mathf.Max(_uiScale, 0.01f);
                var virtualWidth = Screen.width / scale;
                var dockWidth = ResolveDockWidth(virtualWidth);
                var leftWidthPixels = Mathf.Max(1f, Screen.width - dockWidth * scale);
                var bottomPixels = BottomHeight * scale;
                var topPixels = TopHeight * scale;
                var heightPixels = Mathf.Max(1f, Screen.height - topPixels - bottomPixels);
                return new Rect(
                    0f,
                    bottomPixels / Mathf.Max(1f, Screen.height),
                    leftWidthPixels / Mathf.Max(1f, Screen.width),
                    heightPixels / Mathf.Max(1f, Screen.height));
            }
        }

        private void Awake()
        {
            _uiScale = Mathf.Clamp(PlayerPrefs.GetFloat(UiScalePlayerPref, 1.6f), 1.0f, 2.4f);
            _dockWidthRatio = Mathf.Clamp(PlayerPrefs.GetFloat(DockWidthPlayerPref, 0.37f), 0.24f, 0.72f);
        }

        public void Initialize(
            EverestBackendClient backend,
            EverestRobotRenderer robot,
            EverestSnowRenderer snow,
            EverestTerrainRenderer terrain,
            EverestVisualTerrainRenderer visualTerrain,
            EverestEnvironmentRenderer environment,
            EverestCameraController camera,
            EverestRuntime runtime)
        {
            _backend = backend;
            _robot = robot;
            _snow = snow;
            _terrain = terrain;
            _visualTerrain = visualTerrain;
            _environment = environment;
            _camera = camera;
            _runtime = runtime;
            ResetDraftLayers();
            _backend.SubsetViewReceived += OnSubsetView;
        }

        private void OnSubsetView(JObject data)
        {
            var encoded = data.Value<string>("image");
            if (string.IsNullOrWhiteSpace(encoded)) return;
            try
            {
                var bytes = Convert.FromBase64String(encoded);
                if (_subsetTexture == null) _subsetTexture = new Texture2D(2, 2, TextureFormat.RGB24, false);
                ImageConversion.LoadImage(_subsetTexture, bytes, false);
                _subsetCaption = data.Value<string>("caption") ?? "Raw MuJoCo subset";
            }
            catch (Exception exc)
            {
                _subsetCaption = $"Subset preview decode failed: {exc.Message}";
            }
        }

        private void EnsureStyles()
        {
            if (_chrome != null) return;

            // Cold technical palette: one cyan accent, neutral surfaces, no decoration.
            _chromeTex = MakeTexture(Hex("0B1017"));
            _dockTex = MakeTexture(Hex("0E151E"));
            _sectionTex = MakeTexture(Hex("141D28"));
            _buttonTex = MakeTexture(Hex("192431"));
            _buttonHoverTex = MakeTexture(Hex("213142"));
            _accentTex = MakeTexture(Hex("197AA9"));
            _dangerTex = MakeTexture(Hex("8E3D45"));
            _metricTex = MakeTexture(Hex("101923"));
            _splitterTex = MakeTexture(Hex("2A4054"));

            _chrome = new GUIStyle(GUI.skin.box)
            {
                normal = { background = _chromeTex },
                padding = new RectOffset(8, 8, 6, 6),
                margin = new RectOffset(0, 0, 0, 0)
            };
            _dock = new GUIStyle(GUI.skin.box)
            {
                normal = { background = _dockTex },
                padding = new RectOffset(10, 10, 10, 10),
                margin = new RectOffset(0, 0, 0, 0)
            };
            _section = new GUIStyle(GUI.skin.box)
            {
                normal = { background = _sectionTex },
                padding = new RectOffset(10, 10, 8, 10),
                margin = new RectOffset(0, 0, 0, 8)
            };
            _title = new GUIStyle(GUI.skin.label)
            {
                fontSize = 13,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleLeft,
                normal = { textColor = Hex("F0F6FC") }
            };
            _sectionTitle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 10,
                fontStyle = FontStyle.Bold,
                normal = { textColor = Hex("69D0FF") }
            };
            _label = new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                normal = { textColor = Hex("D7E1EB") }
            };
            _muted = new GUIStyle(GUI.skin.label)
            {
                fontSize = 10,
                wordWrap = true,
                normal = { textColor = Hex("8799AC") }
            };
            _tiny = new GUIStyle(_muted) { fontSize = 9, alignment = TextAnchor.MiddleLeft };
            _button = new GUIStyle(GUI.skin.button)
            {
                fontSize = 10,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                fixedHeight = 24f,
                padding = new RectOffset(7, 7, 3, 3),
                normal = { background = _buttonTex, textColor = Hex("C8D4E0") },
                hover = { background = _buttonHoverTex, textColor = Color.white },
                active = { background = _accentTex, textColor = Color.white }
            };
            _buttonActive = new GUIStyle(_button)
            {
                normal = { background = _accentTex, textColor = Color.white },
                hover = { background = _accentTex, textColor = Color.white }
            };
            _buttonDanger = new GUIStyle(_button)
            {
                normal = { background = _dangerTex, textColor = Color.white }
            };
            _good = new GUIStyle(_label) { normal = { textColor = Hex("63D69A") } };
            _warn = new GUIStyle(_label) { normal = { textColor = Hex("FFB86B") } };
            _metric = new GUIStyle(_section)
            {
                normal = { background = _metricTex },
                padding = new RectOffset(8, 8, 6, 6),
                margin = new RectOffset(0, 0, 3, 5)
            };
        }

        private void OnGUI()
        {
            if (_backend == null) return;
            EnsureStyles();
            SyncDraftFromBackendOnce();

            var oldMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.Scale(new Vector3(_uiScale, _uiScale, 1f));
            var virtualWidth = Screen.width / _uiScale;
            var virtualHeight = Screen.height / _uiScale;
            _currentDockWidth = ResolveDockWidth(virtualWidth);
            _compactDock = _currentDockWidth < 300f;

            var dockRect = new Rect(
                virtualWidth - _currentDockWidth,
                TopHeight,
                _currentDockWidth,
                Mathf.Max(0f, virtualHeight - TopHeight - BottomHeight));
            var splitterRect = new Rect(
                dockRect.x - SplitterWidth * 0.5f,
                dockRect.y,
                SplitterWidth,
                dockRect.height);

            DrawCommandBar(new Rect(0f, 0f, virtualWidth, TopHeight));
            HandleDockResize(splitterRect, virtualWidth);
            DrawDock(dockRect);
            DrawStatusBar(new Rect(0f, Mathf.Max(0f, virtualHeight - BottomHeight), virtualWidth, BottomHeight));
            GUI.matrix = oldMatrix;
        }

        private void GetDockWidthBounds(float virtualWidth, out float minDock, out float maxDock)
        {
            var minViewport = Mathf.Clamp(virtualWidth * 0.34f, 72f, 320f);
            maxDock = Mathf.Max(80f, Mathf.Min(480f, virtualWidth - minViewport));
            minDock = Mathf.Min(maxDock, Mathf.Clamp(virtualWidth * 0.32f, 140f, 300f));
        }

        private float ResolveDockWidth(float virtualWidth)
        {
            GetDockWidthBounds(virtualWidth, out var minDock, out var maxDock);
            return Mathf.Clamp(virtualWidth * _dockWidthRatio, minDock, maxDock);
        }

        private Vector2 ScreenToVirtual(Vector3 screenPoint)
        {
            return new Vector2(
                screenPoint.x / Mathf.Max(_uiScale, 0.01f),
                (Screen.height - screenPoint.y) / Mathf.Max(_uiScale, 0.01f));
        }

        private void HandleDockResize(Rect rect, float virtualWidth)
        {
            var current = Event.current;
            if (current.type == EventType.MouseDown && current.button == 0 && rect.Contains(current.mousePosition))
            {
                _dockDragging = true;
                current.Use();
            }
            else if (_dockDragging && current.type == EventType.MouseDrag)
            {
                SetDockWidthFromPixels(virtualWidth - current.mousePosition.x, virtualWidth);
                current.Use();
            }
            else if (_dockDragging && (current.type == EventType.MouseUp || current.rawType == EventType.MouseUp))
            {
                _dockDragging = false;
                PlayerPrefs.SetFloat(DockWidthPlayerPref, _dockWidthRatio);
                PlayerPrefs.Save();
                current.Use();
            }

            if (current.type == EventType.Repaint && _splitterTex != null)
            {
                var line = new Rect(rect.center.x - 0.7f, rect.y, 1.4f, rect.height);
                GUI.DrawTexture(line, _splitterTex);
                if (_dockDragging)
                    GUI.DrawTexture(new Rect(rect.center.x - 1.5f, rect.y, 3f, rect.height), _accentTex);
            }
        }

        private void SetDockWidthFromPixels(float width, float virtualWidth)
        {
            if (virtualWidth <= 1f) return;
            GetDockWidthBounds(virtualWidth, out var minDock, out var maxDock);
            var clamped = Mathf.Clamp(width, minDock, maxDock);
            _dockWidthRatio = Mathf.Clamp(clamped / virtualWidth, 0.24f, 0.72f);
        }

        private void SetDockWidthRatio(float ratio)
        {
            _dockWidthRatio = Mathf.Clamp(ratio, 0.24f, 0.72f);
            PlayerPrefs.SetFloat(DockWidthPlayerPref, _dockWidthRatio);
            PlayerPrefs.Save();
        }

        private void DrawCommandBar(Rect rect)
        {
            var compact = rect.width < 760f;
            var narrow = rect.width < 580f;
            var veryNarrow = rect.width < 360f;
            GUILayout.BeginArea(rect, _chrome);
            GUILayout.BeginHorizontal();
            GUILayout.Label(
                veryNarrow ? "EV" : compact ? "EVEREST" : "EVEREST  /  SIM",
                _title,
                GUILayout.Width(veryNarrow ? 24f : compact ? 72f : 118f));
            GUILayout.Label(
                narrow ? (_backend.Connected ? "●" : "○") : (_backend.Connected ? "● ONLINE" : "○ OFFLINE"),
                _backend.Connected ? _good : _warn,
                GUILayout.Width(narrow ? 18f : 66f));

            if (!narrow)
            {
                if (GUILayout.Button("SIM", _dataMode == "sim" ? _buttonActive : _button, GUILayout.Width(compact ? 38f : 44f))) SetDataMode("sim");
                if (GUILayout.Button("LIVE", _dataMode == "live" ? _buttonActive : _button, GUILayout.Width(compact ? 42f : 48f))) SetDataMode("live");
                GUILayout.Space(4f);
            }

            var paused = _runtime == null || _runtime.Paused;
            var supervisor = _backend.LatestState?["policy"]?["supervisor"] as JObject;
            var waitingCheckpoint = supervisor?.Value<string>("stage") == "waiting_checkpoint";
            var playLabel = waitingCheckpoint
                ? (veryNarrow ? "!" : "SAFETY")
                : veryNarrow ? (paused ? "▶" : "Ⅱ") : paused ? "▶ RUN" : "Ⅱ PAUSE";
            var controlsEnabled = GUI.enabled;
            GUI.enabled = controlsEnabled && _dataMode == "sim" && !waitingCheckpoint;
            if (GUILayout.Button(playLabel, waitingCheckpoint ? _buttonDanger : paused ? _buttonActive : _button, GUILayout.Width(veryNarrow ? 34f : compact ? 62f : 68f)))
            {
                if (paused) _backend.SendPlay();
                else _backend.SendPause(true);
            }
            GUI.enabled = controlsEnabled && _dataMode == "sim";
            if (GUILayout.Button(veryNarrow ? "↺" : "RESET SIM", _button, GUILayout.Width(veryNarrow ? 34f : compact ? 58f : 68f)))
                _backend.SendReset();
            GUI.enabled = controlsEnabled;

            GUILayout.FlexibleSpace();
            if (!narrow)
            {
                var orbit = _camera == null || _camera.Mode == EverestCameraMode.Orbit;
                if (GUILayout.Button("ORBIT", orbit ? _buttonActive : _button, GUILayout.Width(compact ? 48f : 54f))) _camera?.SetMode(EverestCameraMode.Orbit);
                if (GUILayout.Button("FREE", !orbit ? _buttonActive : _button, GUILayout.Width(compact ? 44f : 50f))) _camera?.SetMode(EverestCameraMode.Free);
                if (GUILayout.Button("CAM RESET", _button, GUILayout.Width(compact ? 62f : 72f))) _camera?.ResetCamera();
            }

            if (!compact)
            {
                GUI.enabled = controlsEnabled && _dataMode == "sim" && !waitingCheckpoint;
                var manual = _runtime != null && _runtime.ManualControlEnabled;
                if (GUILayout.Button(manual ? "CONTROL ON" : "CONTROL", manual ? _buttonActive : _button, GUILayout.Width(76f)))
                    _runtime?.SetManualControl(!manual);
                var cheat = _runtime != null && _runtime.CheatModeEnabled;
                if (GUILayout.Button(cheat ? "CHEAT ON" : "CHEAT", cheat ? _buttonDanger : _button, GUILayout.Width(66f)))
                    _runtime?.SetCheatMode(!cheat);
                GUI.enabled = controlsEnabled;
            }
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        private void DrawDock(Rect rect)
        {
            GUILayout.BeginArea(rect, _dock);
            GUILayout.Label("SIMULATION CONTROLS", _sectionTitle);
            GUILayout.Label("Backend-authoritative snow, ice, rock and atmosphere", _tiny);
            GUILayout.Space(5f);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("ENV SETUP", _dockTab == 0 ? _buttonActive : _button)) _dockTab = 0;
            if (GUILayout.Button("DEMO", _dockTab == 1 ? _buttonActive : _button)) _dockTab = 1;
            GUILayout.EndHorizontal();
            if (_dockTab == 0)
            {
                _scroll = GUILayout.BeginScrollView(_scroll, false, true);
                DrawEnvironmentSection();
                DrawPhysicsWindowSection();
                DrawMaterialSection();
                DrawTerrainSection();
                DrawControlSection();
                DrawSystemSection();
                GUILayout.EndScrollView();
            }
            else
            {
                _demoScroll = GUILayout.BeginScrollView(_demoScroll, false, true);
                DrawDemoSection();
                GUILayout.EndScrollView();
            }
            GUILayout.EndArea();
        }

        private void DrawDemoSection()
        {
            BeginSection("POLICY SUPERVISOR");
            var state = _backend.LatestState;
            var policy = state?["policy"] as JObject;
            var supervisor = policy?["supervisor"] as JObject;
            var stage = supervisor?.Value<string>("stage") ?? "monitoring";
            var activeKey = supervisor?.Value<string>("active_policy_key") ?? "flat";
            var selectedKey = policy?.Value<string>("selected_policy_key") ?? activeKey;
            var activeLabel = supervisor?.Value<string>("active_policy_label") ?? "Flat-ground walker";
            var executionSuffix = supervisor?.Value<bool?>("demo_pretrained") == true ? " (flat surrogate)" : "";
            GUILayout.Label($"SELECTED  {selectedKey}  ·  EXECUTING  {activeLabel}{executionSuffix}", _label);
            GUILayout.Label($"stage: {stage} · checkpoint: {supervisor?.Value<string>("executed_checkpoint") ?? "none"}", _tiny);
            if (stage == "waiting_checkpoint")
            {
                var safety = state?["simulation_settings"]?["safety_pose"] as JObject;
                var progress = safety?.Value<float?>("transition_progress") ?? 0f;
                var supports = safety?["support_bodies"] as JArray;
                GUILayout.Label("ACTIVE SAFETY POSTURE · physics remains live while awaiting a compatible checkpoint.", _warn);
                GUILayout.Label($"four-point transition {progress:P0} · support contacts {supports?.Count ?? 0}", _tiny);
                GUILayout.Label($"request {supervisor?.Value<string>("request_id") ?? "pending"}", _tiny);
            }
            GUILayout.Space(4f);
            GUILayout.Label("LOADED / DEMO POLICY", _sectionTitle);
            var registry = policy?["registry"] as JArray;
            if (registry != null)
            {
                foreach (var token in registry)
                {
                    var item = token as JObject;
                    if (item == null) continue;
                    var key = item.Value<string>("key") ?? "";
                    var label = item.Value<string>("label") ?? key;
                    var status = item.Value<string>("status") ?? "unknown";
                    var selected = key == selectedKey;
                    var available = status == "available" || status == "selector" || status == "demo_pretrained";
                    var old = GUI.enabled;
                    GUI.enabled = old && available && key != "recovery";
                    if (GUILayout.Button(selected ? $"● {label}" : label, selected ? _buttonActive : _button))
                        _backend.SendPolicySelect(key);
                    GUI.enabled = old;
                    GUILayout.Label($"{status}{(item.Value<bool?>("surrogate") == true ? " · flat checkpoint surrogate" : "")}", status == "available" ? _good : _muted);
                }
            }
            var currentRoute = supervisor?["route"] as JObject;
            var requestedKey = currentRoute?.Value<string>("requested_key") ?? "ice_incline";
            if (GUILayout.Button("RETURN DEMO-PRETRAINED CHECKPOINT", _buttonActive))
                _backend.SendDemoPretrained(requestedKey);
            GUILayout.Label("REAL COMPATIBLE ONNX RETURN", _sectionTitle);
            _checkpointReturnPath = GUILayout.TextField(_checkpointReturnPath ?? "");
            if (GUILayout.Button("LOAD RETURNED CHECKPOINT", _button) && !string.IsNullOrWhiteSpace(_checkpointReturnPath))
                _backend.SendCheckpointReturn(requestedKey, _checkpointReturnPath.Trim());
            GUILayout.Label("Specialist rows are explicit demo-pretrained returns until a compatible ONNX checkpoint is supplied. The simulator never claims a missing model executed.", _muted);
            EndSection();

            BeginSection("FAILURE / RETRAIN WORKFLOW");
            var detector = supervisor?["detector"] as JObject;
            var risk = detector?["risk"] as JObject;
            GUILayout.Label(
                $"detector  {detector?.Value<string>("kind") ?? "deterministic_imu_contact"} · " +
                $"tilt {risk?.Value<float?>("tilt_degrees") ?? 0f:0.0}° · " +
                $"rate {risk?.Value<float?>("tipping_rate_rad_s") ?? 0f:0.00} rad/s · " +
                $"feet {risk?.Value<int?>("feet_in_contact") ?? 0}", _tiny);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("INJECT DEMO FAILURE", _buttonDanger)) _backend.SendDemoFailure();
            if (GUILayout.Button("REQUEST RETRAIN", _button)) _backend.SendRetrainRequest();
            GUILayout.EndHorizontal();
            var route = supervisor?["route"] as JObject;
            GUILayout.Label($"route  {route?.Value<string>("terrain_type") ?? "terrain pending"} → {route?.Value<string>("requested_label") ?? "selector"}", _label);
            GUILayout.Label("Failure detection mirrors humanoid_climber/safety.py: deterministic tilt/contact thresholds with three-frame confirmation. Retraining captures the current Newton-window subset and waits; no trainer is launched without an endpoint.", _muted);
            EndSection();

            BeginSection("DECISION LOG");
            var entries = supervisor?["decision_log"] as JArray;
            if (entries == null || entries.Count == 0) GUILayout.Label("No routing events yet.", _muted);
            else foreach (var token in entries)
            {
                var entry = token as JObject;
                if (entry == null) continue;
                GUILayout.Label($"[{entry.Value<string>("category")}] {entry.Value<string>("message")}", entry.Value<string>("category") == "FAILURE DETECTED" ? _warn : _label);
            }
            EndSection();

            BeginSection("NEWTON SUBSET ENVIRONMENT");
            var enabled = policy?.Value<bool?>("subset_preview_enabled") == true;
            if (GUILayout.Button(enabled ? "HIDE RAW MUJOCO SUBSET" : "SHOW RAW MUJOCO SUBSET", enabled ? _buttonActive : _button))
                _backend.SendSubsetPreview(!enabled);
            GUILayout.Label(_subsetCaption, _tiny);
            if (_subsetTexture != null && enabled)
                GUILayout.Label(_subsetTexture, GUI.skin.label, GUILayout.Width(Mathf.Min(rectWidth(), 320f)), GUILayout.Height(Mathf.Min(rectWidth() * 0.75f, 240f)));
            GUILayout.Label("This is the native MuJoCo offscreen view of the active Newton-window RL environment. It is intentionally diagnostic rather than styled.", _muted);
            EndSection();
        }

        private float rectWidth()
        {
            return Mathf.Max(120f, _currentDockWidth - 32f);
        }

        private void DrawEnvironmentSection()
        {
            BeginSection("ENVIRONMENT / WEATHER");
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("SIM INPUT", _dataMode == "sim" ? _buttonActive : _button)) SetDataMode("sim");
            if (GUILayout.Button("LIVE INPUT", _dataMode == "live" ? _buttonActive : _button)) SetDataMode("live");
            GUILayout.EndHorizontal();
            GUILayout.Space(4f);
            if (_dataMode != "sim")
            {
                DrawLiveSourceSummary();
                GUILayout.Label("READ ONLY · live telemetry cannot mutate the simulator or arm robot control.", _muted);
                EndSection();
                return;
            }

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("CLEAR", _button)) ApplyWeatherPreset("clear");
            if (GUILayout.Button("STORM", _button)) ApplyWeatherPreset("storm");
            if (GUILayout.Button("WHITEOUT", _button)) ApplyWeatherPreset("whiteout");
            if (GUILayout.Button("WIND", _button)) ApplyWeatherPreset("wind");
            GUILayout.EndHorizontal();
            GUILayout.Space(4f);

            Slider("Temperature", ref _temperature, -45f, 8f, "0.0", "°C");
            Slider("Wind", ref _windSpeed, 0f, 45f, "0.0", "m/s");
            Slider("Direction", ref _windDirection, 0f, 360f, "0", "°");
            Slider("Snowfall", ref _snowfall, 0f, 60f, "0.0", "mm/h");
            Slider("Visibility", ref _visibility, 0.04f, 1f, "0.00", "");

            GUILayout.Space(5f);
            GUILayout.Label("VOLUMETRIC CLOUDS", _sectionTitle);
            Slider("Density", ref _cloudDensity, 0f, 1f, "0.00", "");
            Slider("Coverage", ref _cloudCoverage, 0f, 1f, "0.00", "");
            Slider("Radius", ref _cloudRadius, 15f, 600f, "0", "m");
            Slider("Altitude", ref _cloudAltitude, 5f, 300f, "0", "m");
            Slider("Thickness", ref _cloudThickness, 5f, 180f, "0", "m");
            Slider("Speed", ref _cloudSpeed, 0f, 2f, "0.00", "");
            Slider("Quality", ref _cloudQuality, 0f, 1f, "0.00", "");

            GUILayout.Space(5f);
            if (GUILayout.Button("APPLY WEATHER", _buttonActive)) ApplyEnvironment();
            GUILayout.Label("Wind, friction and snowfall forcing round-trip through the backend. Cloud/fog rendering is presentation-only.", _muted);
            EndSection();
        }

        private void DrawPhysicsWindowSection()
        {
            BeginSection("PHYSICS WINDOW");
            var wasEnabled = GUI.enabled;
            GUI.enabled = wasEnabled && _dataMode == "sim";
            Slider("Radius", ref _physicsRadius, 0.75f, 6f, "0.00", "m");
            Slider("Min voxel", ref _mpmVoxelSize, 0.05f, 0.25f, "0.000", "m");
            SliderInt("Target cells", ref _physicsDetailCells, 24, 96);
            Slider("MPM coupling", ref _mpmCouplingHz, 2f, 30f, "0", "Hz");
            Slider("Contact refine", ref _mpmContactRefineRadius, 0.30f, 1.25f, "0.00", "m");
            SliderInt("Coarse stride", ref _mpmCoarseStride, 1, 4);
            Slider("Recenter", ref _patchRecenter, 0.25f, 0.75f, "0.00", "× radius");

            GUILayout.BeginHorizontal();
            if (GUILayout.Button(_snowAccumulationEnabled ? "ACCUMULATION ON" : "ACCUMULATION OFF",
                    _snowAccumulationEnabled ? _buttonActive : _button))
                _snowAccumulationEnabled = !_snowAccumulationEnabled;
            GUILayout.Label($"time ×{_weatherTimeScale:0}", _muted, GUILayout.Width(66f));
            GUILayout.EndHorizontal();
            LogSlider("Weather time", ref _weatherTimeScale, 1f, 600f, "×");
            if (GUILayout.Button("APPLY PHYSICS WINDOW", _buttonActive)) ApplySimulationSettings();

            var state = _backend.LatestState;
            var settings = state?["simulation_settings"] as JObject;
            var newton = state?["newton"] as JObject;
            GUILayout.BeginVertical(_metric);
            var activeRadius = settings?.Value<float?>("physics_radius_m") ?? _physicsRadius;
            var voxel = newton?.Value<float?>("voxel_size_m");
            var particles = newton?.Value<int?>("particle_count") ?? 0;
            GUILayout.Label($"ACTIVE  r {activeRadius:0.00} m   ·   {particles:N0} particles", _label);
            GUILayout.Label(voxel.HasValue ? $"voxel {voxel.Value:0.000} m · terrain conforming" : "rigid surface / no MPM", _tiny);
            GUILayout.EndVertical();
            GUILayout.Label("Only this moving window is high-cost Newton MPM. The surrounding snow/ice/rock shell is visual DEM LOD.", _muted);
            GUI.enabled = wasEnabled;
            EndSection();
        }

        private void DrawMaterialSection()
        {
            BeginSection("TERRAIN MATERIAL");
            var wasEnabled = GUI.enabled;
            GUI.enabled = wasEnabled && _dataMode == "sim";
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("SNOW / MPM", _surface == "snow" ? _buttonActive : _button)) SetSurface("snow");
            if (GUILayout.Button("RIGID ICE", _surface == "ice" ? _buttonActive : _button)) SetSurface("ice");
            if (GUILayout.Button("BARE ROCK", _surface == "rock" ? _buttonActive : _button)) SetSurface("rock");
            GUILayout.EndHorizontal();

            if (_surface == "ice")
            {
                Slider("Ice friction μ", ref _iceFriction, 0.01f, 0.45f, "0.00", "");
                if (GUILayout.Button("APPLY ICE CONTACT", _buttonActive))
                {
                    _backend.SendSurface("ice");
                    _backend.SendSurfaceFriction(_iceFriction);
                }
                GUILayout.Label("Ice uses the authoritative DEM collider with backend friction. The local active surface is rendered from the same sampled heightfield.", _muted);
                GUI.enabled = wasEnabled;
                EndSection();
                return;
            }

            if (_surface == "rock")
            {
                Slider("Rock friction μ", ref _rockFriction, 0.35f, 1.25f, "0.00", "");
                if (GUILayout.Button("APPLY ROCK CONTACT", _buttonActive))
                {
                    _backend.SendSurface("rock");
                    _backend.SendSurfaceFriction(_rockFriction);
                }
                GUILayout.Label("Bare rock is rigid MuJoCo DEM contact; no fake deformable rock layer is created in Unity.", _muted);
                GUI.enabled = wasEnabled;
                EndSection();
                return;
            }

            Slider("Surface friction μ", ref _surfaceFriction, 0.05f, 0.90f, "0.00", "");
            GUILayout.Space(4f);
            GUILayout.Label("MULTILAYER SNOW / FIRN / ICE", _sectionTitle);
            GUILayout.BeginHorizontal();
            var layerNames = new string[_layers.Count];
            for (var i = 0; i < _layers.Count; ++i) layerNames[i] = $"L{i + 1}";
            _selectedLayer = Mathf.Clamp(GUILayout.Toolbar(_selectedLayer, layerNames, _button), 0, _layers.Count - 1);
            if (GUILayout.Button("+", _button, GUILayout.Width(28f)) && _layers.Count < 6)
            {
                _layers.Add(NewSettledLayer());
                _selectedLayer = _layers.Count - 1;
            }
            if (GUILayout.Button("−", _button, GUILayout.Width(28f)) && _layers.Count > 1)
            {
                _layers.RemoveAt(_selectedLayer);
                _selectedLayer = Mathf.Clamp(_selectedLayer, 0, _layers.Count - 1);
            }
            GUILayout.EndHorizontal();

            var layer = _layers[_selectedLayer];
            var typeIndex = Mathf.Max(0, Array.IndexOf(LayerTypes, layer.Type));
            var nextTypeIndex = GUILayout.Toolbar(typeIndex, _compactDock ? CompactLayerTypes : LayerTypes, _button);
            var nextType = LayerTypes[Mathf.Clamp(nextTypeIndex, 0, LayerTypes.Length - 1)];
            if (nextType != layer.Type) ApplyLayerPreset(layer, nextType);
            Slider("Thickness", ref layer.Thickness, 0.01f, 1f, "0.00", "m");
            Slider("Density", ref layer.Density, 60f, 950f, "0", "kg/m³");
            LogSlider("Stiffness", ref layer.Stiffness, 20000f, 100000000f, "Pa");
            LogSlider("Compression", ref layer.Compressive, 1000f, 10000000f, "Pa");
            LogSlider("Shear", ref layer.Shear, 300f, 5000000f, "Pa");
            Slider("Hardening", ref layer.Hardening, 0f, 40f, "0.0", "");
            LogSlider("Bond below", ref layer.Bond, 500f, 5000000f, "Pa");
            if (GUILayout.Button("APPLY LAYERS TO NEWTON", _buttonActive)) ApplySnow();
            GUI.enabled = wasEnabled;
            EndSection();
        }

        private void DrawTerrainSection()
        {
            BeginSection("DEM / VISUAL LOD");
            SliderInt("Local LOD step", ref _localLod, 1, 16);
            SliderInt("Macro LOD step", ref _macroLod, 2, 32);
            if (GUILayout.Button("APPLY VISUAL LOD", _button))
            {
                _terrain?.SetLod(_localLod, _macroLod);
                _visualTerrain?.SetLod(_localLod, _macroLod);
            }
            GUILayout.Label("Base DEM wireframe + terrain-draped material shell. No decorative planes; the physical window replaces the shell locally.", _muted);
            EndSection();
        }

        private void DrawControlSection()
        {
            BeginSection("ROBOT / CAMERA");
            var wasEnabled = GUI.enabled;
            var supervisor = _backend.LatestState?["policy"]?["supervisor"] as JObject;
            var waitingCheckpoint = supervisor?.Value<string>("stage") == "waiting_checkpoint";
            GUI.enabled = wasEnabled && _dataMode == "sim" && !waitingCheckpoint;
            var manual = _runtime != null && _runtime.ManualControlEnabled;
            var cheat = _runtime != null && _runtime.CheatModeEnabled;
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(manual ? "RELEASE CONTROL" : "TAKE CONTROL", manual ? _button : _buttonActive))
                _runtime?.SetManualControl(!manual);
            if (GUILayout.Button(cheat ? "CHEAT ON" : "CHEAT OFF", cheat ? _buttonDanger : _button))
                _runtime?.SetCheatMode(!cheat);
            GUILayout.EndHorizontal();
            Slider("Cheat speed", ref _cheatSpeed, 0.1f, 5f, "0.00", "m/s");
            Slider("Cheat yaw", ref _cheatYawRate, 0.1f, 4f, "0.00", "rad/s");
            if (GUILayout.Button("APPLY CHEAT SPEED", _button)) ApplySimulationSettings();
            GUILayout.Label(cheat
                ? "NON-PHYSICAL: A/D strafe · S/F move · Q/E yaw. Newton window follows the transported robot."
                : "Force control: W/S nudge · A/D turn · Space stop. MuJoCo + Newton remain authoritative.", cheat ? _warn : _muted);

            GUI.enabled = wasEnabled;

            GUILayout.Space(5f);
            GUILayout.Label("CAMERA", _sectionTitle);
            GUILayout.BeginHorizontal();
            var orbit = _camera == null || _camera.Mode == EverestCameraMode.Orbit;
            if (GUILayout.Button("ORBIT", orbit ? _buttonActive : _button)) _camera?.SetMode(EverestCameraMode.Orbit);
            if (GUILayout.Button("FREE", !orbit ? _buttonActive : _button)) _camera?.SetMode(EverestCameraMode.Free);
            if (GUILayout.Button("RESET CAM", _button)) _camera?.ResetCamera();
            GUILayout.EndHorizontal();
            GUILayout.Label(
                orbit
                    ? "LMB drag orbit · RMB drag pan · wheel zoom. Camera renders only in the visible viewport."
                    : "LMB look · arrow keys move · Shift+WASD/QE also moves. RESET CAM returns to the robot.",
                _muted);

            GUILayout.Space(4f);
            DrawFeet(_robot?.LatestFeet);
            EndSection();
        }

        private void DrawSystemSection()
        {
            BeginSection("EDITOR / SYSTEM");
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("160%", Mathf.Abs(_uiScale - 1.6f) < 0.01f ? _buttonActive : _button)) SetUiScale(1.6f);
            if (GUILayout.Button("180%", Mathf.Abs(_uiScale - 1.8f) < 0.01f ? _buttonActive : _button)) SetUiScale(1.8f);
            if (GUILayout.Button("200%", Mathf.Abs(_uiScale - 2.0f) < 0.01f ? _buttonActive : _button)) SetUiScale(2.0f);
            GUILayout.EndHorizontal();
            var next = GUILayout.HorizontalSlider(_uiScale, 1f, 2.4f);
            if (Mathf.Abs(next - _uiScale) > 0.002f) SetUiScale(next);
            GUILayout.Label($"UI scale {_uiScale * 100f:0}%", _muted);

            GUILayout.Space(5f);
            GUILayout.Label("PANEL WIDTH", _sectionTitle);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("NARROW", _button)) SetDockWidthRatio(0.30f);
            if (GUILayout.Button("DEFAULT", _buttonActive)) SetDockWidthRatio(0.37f);
            if (GUILayout.Button("WIDE", _button)) SetDockWidthRatio(0.50f);
            GUILayout.EndHorizontal();
            var nextDockRatio = GUILayout.HorizontalSlider(_dockWidthRatio, 0.24f, 0.72f);
            if (Mathf.Abs(nextDockRatio - _dockWidthRatio) > 0.002f) SetDockWidthRatio(nextDockRatio);
            GUILayout.Label($"dock {_currentDockWidth:0} px virtual · drag its left edge to resize", _muted);

            var state = _backend.LatestState;
            if (state != null)
            {
                GUILayout.Space(4f);
                GUILayout.Label($"engine  {state.Value<string>("engine")}", _tiny);
                var newton = state["newton"] as JObject;
                if (newton != null)
                    GUILayout.Label($"newton  {newton.Value<string>("solver")} · {newton.Value<string>("device")}", _tiny);
            }
            var ack = _backend.LatestControlAck;
            if (ack != null)
                GUILayout.Label(ack.Value<bool?>("ok") == true ? "● control acknowledged" : $"control error · {ack.Value<string>("message")}",
                    ack.Value<bool?>("ok") == true ? _good : _warn);
            EndSection();
        }

        private void DrawStatusBar(Rect rect)
        {
            var compact = rect.width < 620f;
            var narrow = rect.width < 430f;
            GUILayout.BeginArea(rect, _chrome);
            GUILayout.BeginHorizontal();
            var state = _backend.LatestState;
            var simTime = state?.Value<float?>("sim_time") ?? 0f;
            GUILayout.Label(
                narrow ? (_backend.Connected ? "●" : "○") : (_backend.Connected ? "● BACKEND" : "○ OFFLINE"),
                _backend.Connected ? _good : _warn,
                GUILayout.Width(narrow ? 18f : 70f));
            var source = state?["source"] as JObject;
            var sourceStatus = source?.Value<string>("status") ?? "disconnected";
            var sourceKind = source?.Value<string>("kind") ?? "unknown";
            var ageMs = source?.Value<float?>("age_ms");
            var standLock = state?["simulation_settings"]?.Value<bool?>("stand_lock_active") ?? false;
            var modeLabel = _dataMode == "live"
                ? $"LIVE {sourceStatus.ToUpperInvariant()}"
                : standLock ? $"SIM HOLD {simTime:0.00}s" : $"SIM {simTime:0.00}s";
            GUILayout.Label(modeLabel, sourceStatus == "connected" ? _tiny : _warn, GUILayout.Width(narrow ? 92f : 124f));
            GUILayout.Label(_surface.ToUpperInvariant(), _tiny, GUILayout.Width(narrow ? 42f : 48f));
            if (_dataMode == "live" && !compact)
                GUILayout.Label($"{sourceKind} · {(ageMs.HasValue ? ageMs.Value.ToString("0") + " ms" : "no robot sample")} · READ ONLY", _tiny, GUILayout.Width(230f));
            if (!compact && _snow != null && _snow.Sequence >= 0)
                GUILayout.Label($"{_snow.LayerCount} layers · comp {_snow.MaxCompaction:P0}", _tiny, GUILayout.Width(120f));
            GUILayout.FlexibleSpace();
            if (!compact)
            {
                GUILayout.Label(
                    _runtime != null && _runtime.CheatModeEnabled
                        ? "NON-PHYSICAL TRANSPORT ACTIVE"
                        : "LMB orbit/look · RMB pan · P pause · R reset",
                    _runtime != null && _runtime.CheatModeEnabled ? _warn : _tiny);
            }
            else if (_runtime != null && _runtime.CheatModeEnabled)
            {
                GUILayout.Label("CHEAT", _warn, GUILayout.Width(40f));
            }
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        private void DrawLiveSourceSummary()
        {
            var source = _backend.LatestState?["source"] as JObject;
            if (source == null)
            {
                GUILayout.Label("LIVE DISCONNECTED · source metadata pending", _warn);
                return;
            }
            var status = source.Value<string>("status") ?? "disconnected";
            var kind = source.Value<string>("kind") ?? "unknown";
            var name = source.Value<string>("name") ?? kind;
            var age = source.Value<float?>("age_ms");
            GUILayout.Label(
                $"LIVE {status.ToUpperInvariant()} · {name} · {(age.HasValue ? age.Value.ToString("0") + " ms" : "no robot sample")}",
                status == "connected" ? _good : _warn);
            var channels = source["channels"] as JObject;
            if (channels == null) return;
            foreach (var channelName in new[] { "robot", "weather", "terrain", "snow", "sensors" })
            {
                var channel = channels[channelName] as JObject;
                var channelStatus = channel?.Value<string>("status") ?? "unavailable";
                if (channelStatus == "connected") continue;
                GUILayout.Label($"{channelName.ToUpperInvariant()} {channelStatus.ToUpperInvariant()}", _warn);
            }
            var robot = channels["robot"] as JObject;
            var missingBodies = robot?["missing_bodies"] as JArray;
            var missingJoints = robot?["missing_joints"] as JArray;
            if ((missingBodies?.Count ?? 0) > 0 || (missingJoints?.Count ?? 0) > 0)
                GUILayout.Label($"INCOMPLETE ROBOT LAYOUT · {missingBodies?.Count ?? 0} bodies · {missingJoints?.Count ?? 0} joints missing", _warn);
        }

        private void ApplyWeatherPreset(string preset)
        {
            switch (preset)
            {
                case "storm":
                    _temperature = -22f; _windSpeed = 24f; _snowfall = 22f; _visibility = 0.38f;
                    _cloudDensity = 0.68f; _cloudCoverage = 0.86f; _cloudAltitude = 26f; _cloudThickness = 54f; _cloudQuality = 0.68f;
                    break;
                case "whiteout":
                    _temperature = -19f; _windSpeed = 15f; _snowfall = 45f; _visibility = 0.10f;
                    _cloudDensity = 0.86f; _cloudCoverage = 0.96f; _cloudAltitude = 18f; _cloudThickness = 72f; _cloudQuality = 0.72f;
                    break;
                case "wind":
                    _temperature = -28f; _windSpeed = 38f; _snowfall = 2f; _visibility = 0.62f;
                    _cloudDensity = 0.34f; _cloudCoverage = 0.46f; _cloudAltitude = 58f; _cloudThickness = 32f; _cloudQuality = 0.50f;
                    break;
                default:
                    _temperature = -16f; _windSpeed = 5f; _snowfall = 0f; _visibility = 1f;
                    _cloudDensity = 0.16f; _cloudCoverage = 0.24f; _cloudAltitude = 70f; _cloudThickness = 24f; _cloudQuality = 0.45f;
                    break;
            }
        }

        private void ApplyEnvironment()
        {
            _backend.SendWeather(new JObject
            {
                ["temperature_c"] = _temperature,
                ["wind_speed_m_s"] = _windSpeed,
                ["wind_direction_deg"] = _windDirection,
                ["snowfall_mm_h"] = _snowfall,
                ["visibility_scale"] = _visibility,
                ["cloud_density"] = _cloudDensity,
                ["cloud_coverage"] = _cloudCoverage,
                ["cloud_radius_m"] = _cloudRadius,
                ["cloud_altitude_m"] = _cloudAltitude,
                ["cloud_thickness_m"] = _cloudThickness,
                ["cloud_speed"] = _cloudSpeed,
                ["cloud_quality"] = _cloudQuality,
                ["movement_allowed"] = true
            });
        }

        private void ApplySimulationSettings()
        {
            _backend.SendSimulationSettings(new JObject
            {
                ["physics_radius_m"] = _physicsRadius,
                ["mpm_min_voxel_size_m"] = _mpmVoxelSize,
                ["physics_detail_cells"] = _physicsDetailCells,
                ["mpm_coupling_hz"] = _mpmCouplingHz,
                ["mpm_contact_refine_radius_m"] = _mpmContactRefineRadius,
                ["mpm_coarse_stride"] = _mpmCoarseStride,
                ["patch_recenter_fraction"] = _patchRecenter,
                ["snow_accumulation_enabled"] = _snowAccumulationEnabled,
                ["weather_time_scale"] = _weatherTimeScale,
                ["cheat_speed_m_s"] = _cheatSpeed,
                ["cheat_yaw_rate_rad_s"] = _cheatYawRate
            });
        }

        private void ApplySnow()
        {
            var layers = new JArray();
            foreach (var layer in _layers) layers.Add(layer.ToJson());
            _backend.SendSurface("snow");
            _backend.SendSnowParameters(new JObject
            {
                ["surface_friction"] = _surfaceFriction,
                ["snowfall_mm_h"] = _snowfall,
                ["wind_speed_m_s"] = _windSpeed,
                ["wind_direction_deg"] = _windDirection,
                ["temperature_c"] = _temperature,
                ["slope_deg"] = 18f,
                ["layers"] = layers
            });
        }

        private void SetSurface(string surface)
        {
            if (_surface == surface) return;
            _surface = surface;
            _backend.SendSurface(surface);
        }

        private void SetDataMode(string mode)
        {
            if (_dataMode == mode) return;
            _backend.SendMode(mode);
        }

        private void SyncDraftFromBackendOnce()
        {
            var state = _backend.LatestState;
            if (state != null)
            {
                var backendMode = state.Value<string>("data_mode") ?? _dataMode;
                if (_dataMode == "live" && backendMode == "sim") _draftInitialized = false;
                _dataMode = backendMode;
            }
            if (_draftInitialized) return;
            if (state != null)
            {
                _dataMode = state.Value<string>("data_mode") ?? _dataMode;
                _surface = state.Value<string>("surface") ?? _surface;
                var settings = state["simulation_settings"] as JObject;
                if (settings != null)
                {
                    _physicsRadius = settings.Value<float?>("physics_radius_m") ?? _physicsRadius;
                    _mpmVoxelSize = settings.Value<float?>("mpm_min_voxel_size_m") ?? settings.Value<float?>("mpm_voxel_size_m") ?? _mpmVoxelSize;
                    _physicsDetailCells = settings.Value<int?>("physics_detail_cells") ?? _physicsDetailCells;
                    _mpmCouplingHz = settings.Value<float?>("mpm_coupling_hz") ?? _mpmCouplingHz;
                    _mpmContactRefineRadius = settings.Value<float?>("mpm_contact_refine_radius_m") ?? _mpmContactRefineRadius;
                    _mpmCoarseStride = settings.Value<int?>("mpm_coarse_stride") ?? _mpmCoarseStride;
                    _patchRecenter = settings.Value<float?>("patch_recenter_fraction") ?? _patchRecenter;
                    _snowAccumulationEnabled = settings.Value<bool?>("snow_accumulation_enabled") ?? _snowAccumulationEnabled;
                    _weatherTimeScale = settings.Value<float?>("weather_time_scale") ?? _weatherTimeScale;
                    _cheatSpeed = settings.Value<float?>("cheat_speed_m_s") ?? _cheatSpeed;
                    _cheatYawRate = settings.Value<float?>("cheat_yaw_rate_rad_s") ?? _cheatYawRate;
                }
            }

            var env = _backend.LatestEnvironment;
            if (env != null)
            {
                _temperature = env.Value<float?>("temperature_c") ?? _temperature;
                _windSpeed = env.Value<float?>("wind_speed_m_s") ?? _windSpeed;
                _windDirection = env.Value<float?>("wind_direction_deg") ?? _windDirection;
                _snowfall = env.Value<float?>("snowfall_mm_h") ?? _snowfall;
                _visibility = env.Value<float?>("visibility_scale") ?? _visibility;
                _cloudDensity = env.Value<float?>("cloud_density") ?? _cloudDensity;
                _cloudCoverage = env.Value<float?>("cloud_coverage") ?? _cloudCoverage;
                _cloudRadius = env.Value<float?>("cloud_radius_m") ?? _cloudRadius;
                _cloudAltitude = env.Value<float?>("cloud_altitude_m") ?? _cloudAltitude;
                _cloudThickness = env.Value<float?>("cloud_thickness_m") ?? _cloudThickness;
                _cloudSpeed = env.Value<float?>("cloud_speed") ?? _cloudSpeed;
                _cloudQuality = env.Value<float?>("cloud_quality") ?? _cloudQuality;
            }

            var snow = _backend.LatestSnow;
            var rawLayers = snow?["layers"] as JArray;
            if (rawLayers != null && rawLayers.Count > 0)
            {
                _layers.Clear();
                foreach (var token in rawLayers)
                {
                    if (!(token is JObject raw)) continue;
                    _layers.Add(new LayerDraft
                    {
                        Type = raw.Value<string>("type") ?? "POWDER",
                        Label = raw.Value<string>("label") ?? "Snow layer",
                        Thickness = raw.Value<float?>("thickness_m") ?? 0.08f,
                        Density = raw.Value<float?>("density_kg_m3") ?? 120f,
                        Stiffness = raw.Value<float?>("stiffness_pa") ?? 35000f,
                        Compressive = raw.Value<float?>("compressive_strength_pa") ?? 3500f,
                        Shear = raw.Value<float?>("shear_strength_pa") ?? 1800f,
                        Hardening = raw.Value<float?>("compaction_hardening") ?? 12f,
                        Bond = raw.Value<float?>("bond_strength_below_pa") ?? 2500f
                    });
                }
                _surfaceFriction = snow.Value<float?>("surface_friction") ?? _surfaceFriction;
            }
            _draftInitialized = state != null && env != null && snow != null
                && (env.Value<string>("data_mode") ?? _dataMode) == _dataMode
                && (snow.Value<string>("data_mode") ?? _dataMode) == _dataMode;
        }

        private void DrawFeet(JObject feet)
        {
            if (feet == null)
            {
                GUILayout.Label("foot telemetry pending", _tiny);
                return;
            }
            DrawFoot("L", feet["left"] as JObject);
            DrawFoot("R", feet["right"] as JObject);
        }

        private void DrawFoot(string side, JObject foot)
        {
            if (foot == null) return;
            var contact = foot.Value<bool?>("contact") ?? false;
            var force = foot.Value<float?>("normal_force_n") ?? 0f;
            var sink = foot.Value<float?>("penetration_m") ?? 0f;
            var slip = foot.Value<float?>("slip_speed_m_s") ?? 0f;
            GUILayout.Label($"{side} {(contact ? "CONTACT" : "air")} · {force:0} N · sink {sink * 100f:0.0} cm · slip {slip:0.000} m/s", contact ? _label : _tiny);
        }

        private void SetUiScale(float value)
        {
            var clamped = Mathf.Clamp(value, 1f, 2.4f);
            if (Mathf.Abs(clamped - _uiScale) < 0.001f) return;
            _uiScale = clamped;
            PlayerPrefs.SetFloat(UiScalePlayerPref, _uiScale);
            PlayerPrefs.Save();
        }

        private void BeginSection(string title)
        {
            GUILayout.BeginVertical(_section);
            GUILayout.Label(title, _sectionTitle);
        }

        private static void EndSection() => GUILayout.EndVertical();

        private void Slider(string label, ref float value, float min, float max, string format, string suffix)
        {
            var labelWidth = _compactDock ? 72f : 92f;
            var valueWidth = _compactDock ? 56f : 70f;
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, _label, GUILayout.Width(labelWidth));
            value = GUILayout.HorizontalSlider(value, min, max, GUILayout.ExpandWidth(true));
            GUILayout.Label($"{value.ToString(format)} {suffix}", _tiny, GUILayout.Width(valueWidth));
            GUILayout.EndHorizontal();
        }

        private void SliderInt(string label, ref int value, int min, int max)
        {
            var labelWidth = _compactDock ? 72f : 92f;
            var valueWidth = _compactDock ? 56f : 70f;
            float asFloat = value;
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, _label, GUILayout.Width(labelWidth));
            asFloat = GUILayout.HorizontalSlider(asFloat, min, max, GUILayout.ExpandWidth(true));
            value = Mathf.Clamp(Mathf.RoundToInt(asFloat), min, max);
            GUILayout.Label(value.ToString(), _tiny, GUILayout.Width(valueWidth));
            GUILayout.EndHorizontal();
        }

        private void LogSlider(string label, ref float value, float min, float max, string suffix)
        {
            var labelWidth = _compactDock ? 72f : 92f;
            var valueWidth = _compactDock ? 56f : 70f;
            var logMin = Mathf.Log10(min);
            var logMax = Mathf.Log10(max);
            var log = Mathf.Log10(Mathf.Clamp(value, min, max));
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, _label, GUILayout.Width(labelWidth));
            log = GUILayout.HorizontalSlider(log, logMin, logMax, GUILayout.ExpandWidth(true));
            value = Mathf.Pow(10f, log);
            GUILayout.Label(FormatEngineering(value, suffix), _tiny, GUILayout.Width(valueWidth));
            GUILayout.EndHorizontal();
        }

        private static string FormatEngineering(float value, string suffix)
        {
            if (suffix == "×") return $"{value:0}×";
            if (value >= 1000000f) return $"{value / 1000000f:0.##}M {suffix}";
            if (value >= 1000f) return $"{value / 1000f:0.##}k {suffix}";
            return $"{value:0} {suffix}";
        }

        private static void ApplyLayerPreset(LayerDraft layer, string type)
        {
            layer.Type = type;
            layer.Label = type.Replace('_', ' ');
            switch (type)
            {
                case "POWDER":
                    layer.Density = 120f; layer.Stiffness = 35000f; layer.Compressive = 3500f; layer.Shear = 1800f; layer.Hardening = 12f; layer.Bond = 2500f;
                    break;
                case "WIND_PACK":
                    layer.Density = 360f; layer.Stiffness = 750000f; layer.Compressive = 85000f; layer.Shear = 32000f; layer.Hardening = 22f; layer.Bond = 12000f;
                    break;
                case "CRUST":
                    layer.Density = 460f; layer.Stiffness = 1800000f; layer.Compressive = 180000f; layer.Shear = 70000f; layer.Hardening = 28f; layer.Bond = 9000f;
                    break;
                case "FIRN":
                    layer.Density = 650f; layer.Stiffness = 6500000f; layer.Compressive = 650000f; layer.Shear = 220000f; layer.Hardening = 32f; layer.Bond = 180000f;
                    break;
                case "ICE":
                    layer.Density = 917f; layer.Stiffness = 60000000f; layer.Compressive = 5000000f; layer.Shear = 1800000f; layer.Hardening = 8f; layer.Bond = 900000f;
                    break;
                default:
                    layer.Density = 320f; layer.Stiffness = 500000f; layer.Compressive = 55000f; layer.Shear = 20000f; layer.Hardening = 18f; layer.Bond = 7500f;
                    break;
            }
        }

        private void ResetDraftLayers()
        {
            _layers.Clear();
            _layers.Add(new LayerDraft());
            _layers.Add(NewSettledLayer());
        }

        private static LayerDraft NewSettledLayer()
        {
            return new LayerDraft
            {
                Type = "DENSE_SNOW", Label = "Settled snow", Thickness = 0.32f, Density = 320f,
                Stiffness = 500000f, Compressive = 55000f, Shear = 20000f, Hardening = 18f, Bond = 7500f
            };
        }

        private static Color Hex(string hex)
        {
            Color color;
            return ColorUtility.TryParseHtmlString("#" + hex, out color) ? color : Color.white;
        }

        private static Texture2D MakeTexture(Color color)
        {
            var texture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            texture.SetPixel(0, 0, color);
            texture.Apply();
            return texture;
        }

        private void OnDestroy()
        {
            if (_backend != null) _backend.SubsetViewReceived -= OnSubsetView;
            Destroy(_subsetTexture);
            Destroy(_chromeTex); Destroy(_dockTex); Destroy(_sectionTex); Destroy(_buttonTex);
            Destroy(_buttonHoverTex); Destroy(_accentTex); Destroy(_dangerTex); Destroy(_metricTex);
            Destroy(_splitterTex);
        }
    }
}
