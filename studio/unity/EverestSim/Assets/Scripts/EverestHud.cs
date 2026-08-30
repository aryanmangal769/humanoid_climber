using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EverestSim
{
    public sealed class EverestHud : MonoBehaviour
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
                if (Type == "ICE") color = new JArray(0.28, 0.68, 0.88);
                else if (Type == "CRUST") color = new JArray(0.78, 0.88, 0.96);
                else if (Type == "FIRN") color = new JArray(0.64, 0.78, 0.88);
                else color = new JArray(0.94, 0.97, 1.0);
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
        private EverestEnvironmentRenderer _environment;
        private EverestCameraController _camera;
        private EverestRuntime _runtime;

        private GUIStyle _panel;
        private GUIStyle _section;
        private GUIStyle _title;
        private GUIStyle _eyebrow;
        private GUIStyle _label;
        private GUIStyle _muted;
        private GUIStyle _button;
        private GUIStyle _buttonActive;
        private GUIStyle _statusGood;
        private GUIStyle _statusWarn;
        private Texture2D _panelTex;
        private Texture2D _sectionTex;
        private Texture2D _buttonTex;
        private Texture2D _buttonActiveTex;

        private Vector2 _scroll;
        private int _workspaceSelection = 3;
        private int _selectedLayer;
        private bool _draftInitialized;
        private string _surface = "snow";
        private string _dataMode = "sim";
        private float _temperature = -18f;
        private float _windSpeed = 8f;
        private float _windDirection = 250f;
        private float _snowfall;
        private float _visibility = 1f;
        private float _cloudDensity = 0.28f;
        private float _cloudCoverage = 0.42f;
        private float _cloudRadius = 120f;
        private float _cloudAltitude = 42f;
        private float _cloudThickness = 30f;
        private float _cloudSpeed = 0.35f;
        private float _cloudQuality = 0.55f;
        private float _surfaceFriction = 0.36f;
        private float _iceFriction = 0.08f;
        private float _physicsRadius = 2f;
        private float _mpmVoxelSize = 0.08f;
        private int _physicsDetailCells = 50;
        private float _patchRecenter = 0.45f;
        private float _cheatSpeed = 1.6f;
        private float _cheatYawRate = 1.4f;
        private float _uiScale = 1f;
        private int _localLod = 3;
        private int _macroLod = 7;
        private readonly List<LayerDraft> _layers = new List<LayerDraft>();

        private static readonly string[] LayerTypes = { "POWDER", "WIND_PACK", "CRUST", "DENSE_SNOW", "FIRN", "ICE" };

        private const string UiScalePlayerPref = "EverestSim.UiScale";

        private void Awake()
        {
            _uiScale = Mathf.Clamp(PlayerPrefs.GetFloat(UiScalePlayerPref, 1f), 0.65f, 1.60f);
        }

        public void Initialize(
            EverestBackendClient backend,
            EverestRobotRenderer robot,
            EverestSnowRenderer snow,
            EverestTerrainRenderer terrain,
            EverestEnvironmentRenderer environment,
            EverestCameraController camera,
            EverestRuntime runtime)
        {
            _backend = backend;
            _robot = robot;
            _snow = snow;
            _terrain = terrain;
            _environment = environment;
            _camera = camera;
            _runtime = runtime;
            ResetDraftLayers();
        }

        private void OnDestroy()
        {
            Destroy(_panelTex);
            Destroy(_sectionTex);
            Destroy(_buttonTex);
            Destroy(_buttonActiveTex);
        }

        private void EnsureStyles()
        {
            if (_panel != null) return;

            // VS Code / Unity-editor inspired shell. Keep interaction instantaneous:
            // these controls are used repeatedly while tuning physics.
            _panelTex = MakeTexture(new Color(0.075f, 0.082f, 0.092f, 0.985f));
            _sectionTex = MakeTexture(new Color(0.105f, 0.115f, 0.128f, 0.98f));
            _buttonTex = MakeTexture(new Color(0.135f, 0.145f, 0.158f, 1f));
            _buttonActiveTex = MakeTexture(new Color(0.055f, 0.43f, 0.63f, 1f));

            _panel = new GUIStyle(GUI.skin.box)
            {
                normal = { background = _panelTex },
                padding = new RectOffset(10, 10, 8, 8),
                margin = new RectOffset(0, 0, 0, 0)
            };
            _section = new GUIStyle(GUI.skin.box)
            {
                normal = { background = _sectionTex },
                padding = new RectOffset(11, 11, 9, 10),
                margin = new RectOffset(0, 0, 5, 7)
            };
            _title = new GUIStyle(GUI.skin.label)
            {
                fontSize = 17,
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.96f, 0.97f, 0.98f) }
            };
            _eyebrow = new GUIStyle(GUI.skin.label)
            {
                fontSize = 10,
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.39f, 0.77f, 0.94f) }
            };
            _label = new GUIStyle(GUI.skin.label)
            {
                fontSize = 12,
                normal = { textColor = new Color(0.88f, 0.90f, 0.92f) }
            };
            _muted = new GUIStyle(_label)
            {
                fontSize = 11,
                normal = { textColor = new Color(0.55f, 0.59f, 0.64f) }
            };
            _button = new GUIStyle(GUI.skin.button)
            {
                fontSize = 11,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                fixedHeight = 27f,
                normal = { background = _buttonTex, textColor = new Color(0.80f, 0.83f, 0.86f) },
                hover = { background = _sectionTex, textColor = Color.white },
                active = { background = _buttonActiveTex, textColor = Color.white }
            };
            _buttonActive = new GUIStyle(_button)
            {
                normal = { background = _buttonActiveTex, textColor = Color.white }
            };
            _statusGood = new GUIStyle(_label) { normal = { textColor = new Color(0.40f, 0.88f, 0.62f) } };
            _statusWarn = new GUIStyle(_label) { normal = { textColor = new Color(1f, 0.60f, 0.38f) } };
        }

        private static Texture2D MakeTexture(Color color)
        {
            var texture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            texture.SetPixel(0, 0, color);
            texture.Apply();
            return texture;
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

            const float topH = 38f;
            const float tuningH = 34f;
            const float chromeH = topH + tuningH;
            const float leftW = 226f;
            const float rightW = 390f;
            const float bottomH = 26f;

            DrawTopToolbar(new Rect(0f, 0f, virtualWidth, topH));
            DrawTuningToolbar(new Rect(0f, topH, virtualWidth, tuningH));
            DrawHierarchy(new Rect(0f, chromeH, leftW, virtualHeight - chromeH - bottomH));
            DrawInspector(new Rect(virtualWidth - rightW, chromeH, rightW, virtualHeight - chromeH - bottomH));
            DrawBottomStatus(new Rect(0f, virtualHeight - bottomH, virtualWidth, bottomH));
            GUI.matrix = oldMatrix;
        }

        private void DrawTopToolbar(Rect rect)
        {
            GUILayout.BeginArea(rect, _panel);
            GUILayout.BeginHorizontal();
            GUILayout.Label("EVEREST / SIM STUDIO", _eyebrow, GUILayout.Width(172f));
            GUILayout.Space(6f);
            if (GUILayout.Button("SIM", _dataMode == "sim" ? _buttonActive : _button, GUILayout.Width(62f))) SetDataMode("sim");
            if (GUILayout.Button("LIVE", _dataMode == "live" ? _buttonActive : _button, GUILayout.Width(62f))) SetDataMode("live");
            GUILayout.Space(9f);

            var paused = _runtime == null || _runtime.Paused;
            if (GUILayout.Button(paused ? "▶ RUN" : "Ⅱ PAUSE", paused ? _buttonActive : _button, GUILayout.Width(86f)))
                _backend.SendPause(!paused);
            if (GUILayout.Button("↺ RESET", _button, GUILayout.Width(78f)))
                _backend.SendReset();

            GUILayout.FlexibleSpace();

            var orbit = _camera == null || _camera.Mode == EverestCameraMode.Orbit;
            if (GUILayout.Button("ORBIT", orbit ? _buttonActive : _button, GUILayout.Width(70f)))
                _camera?.SetMode(EverestCameraMode.Orbit);
            if (GUILayout.Button("FREE", !orbit ? _buttonActive : _button, GUILayout.Width(70f)))
                _camera?.SetMode(EverestCameraMode.Free);

            var cheat = _runtime != null && _runtime.CheatModeEnabled;
            if (GUILayout.Button(cheat ? "CHEAT ON" : "CHEAT", cheat ? _buttonActive : _button, GUILayout.Width(82f)))
                _runtime?.SetCheatMode(!cheat);

            var manual = _runtime != null && _runtime.ManualControlEnabled;
            if (GUILayout.Button(manual ? "CONTROL ON" : "TAKE CONTROL", manual ? _buttonActive : _button, GUILayout.Width(116f)))
                _runtime?.SetManualControl(!manual);
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        private void DrawTuningToolbar(Rect rect)
        {
            GUILayout.BeginArea(rect, _panel);
            GUILayout.BeginHorizontal();

            GUILayout.Label("UI", _eyebrow, GUILayout.Width(22f));
            if (GUILayout.Button("−", _button, GUILayout.Width(28f))) SetUiScale(_uiScale - 0.10f);
            var nextScale = GUILayout.HorizontalSlider(_uiScale, 0.65f, 1.60f, GUILayout.Width(92f));
            if (Mathf.Abs(nextScale - _uiScale) > 0.002f) SetUiScale(nextScale);
            GUILayout.Label($"{_uiScale * 100f:0}%", _muted, GUILayout.Width(42f));
            if (GUILayout.Button("+", _button, GUILayout.Width(28f))) SetUiScale(_uiScale + 0.10f);
            if (GUILayout.Button("100%", _button, GUILayout.Width(48f))) SetUiScale(1f);

            GUILayout.Space(18f);
            GUILayout.Label("PHYSICS RADIUS", _eyebrow, GUILayout.Width(100f));
            _physicsRadius = GUILayout.HorizontalSlider(_physicsRadius, 0.75f, 6.0f, GUILayout.Width(160f));
            GUILayout.Label($"{_physicsRadius:0.00} m", _label, GUILayout.Width(58f));
            if (GUILayout.Button("APPLY", _buttonActive, GUILayout.Width(62f))) ApplySimulationSettings();

            var state = _backend.LatestState;
            var newton = state?["newton"] as JObject;
            if (newton != null && rect.width > 820f)
            {
                var effectiveVoxel = newton.Value<float?>("voxel_size_m");
                var particles = newton.Value<int?>("particle_count") ?? 0;
                GUILayout.Space(10f);
                GUILayout.Label(
                    rect.width > 1040f && effectiveVoxel.HasValue
                        ? $"backend voxel {effectiveVoxel.Value:0.000} m · {particles:N0} particles"
                        : $"{particles:N0} particles",
                    _muted,
                    GUILayout.Width(rect.width > 1040f ? 230f : 100f));
            }

            GUILayout.FlexibleSpace();
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        private void SetUiScale(float value)
        {
            var clamped = Mathf.Clamp(value, 0.65f, 1.60f);
            if (Mathf.Abs(clamped - _uiScale) < 0.001f) return;
            _uiScale = clamped;
            PlayerPrefs.SetFloat(UiScalePlayerPref, _uiScale);
            PlayerPrefs.Save();
        }

        private void DrawHierarchy(Rect rect)
        {
            GUILayout.BeginArea(rect, _panel);
            GUILayout.Label("SCENE", _eyebrow);
            GUILayout.Space(5f);
            HierarchyRow(0, "▾  Everest Scene");
            HierarchyRow(1, "   ◇ Unitree G1");
            HierarchyRow(2, "   ◇ Atmosphere");
            HierarchyRow(3, "   ◆ Snowpack / Ice");
            HierarchyRow(4, "   ◇ Terrain");
            HierarchyRow(5, "   ◇ Backend");

            GUILayout.Space(14f);
            GUILayout.Label("ASSETS", _eyebrow);
            GUILayout.Label("  HMA2 · 2 m DEM", _muted);
            GUILayout.Label("  Poly Haven snow / rock", _muted);
            GUILayout.Label("  Unitree G1 meshes", _muted);

            GUILayout.FlexibleSpace();
            GUILayout.Label(_backend.Connected ? "● backend online" : "○ reconnecting",
                _backend.Connected ? _statusGood : _statusWarn);
            GUILayout.EndArea();
        }

        private void HierarchyRow(int id, string label)
        {
            var style = _workspaceSelection == id ? _buttonActive : _button;
            if (GUILayout.Button(label, style, GUILayout.Height(30f)))
            {
                _workspaceSelection = id;
                _scroll = Vector2.zero;
            }
        }

        private void DrawInspector(Rect rect)
        {
            GUILayout.BeginArea(rect, _panel);
            GUILayout.Label("INSPECTOR", _eyebrow);
            GUILayout.Label(InspectorTitle(), _title);
            GUILayout.Space(6f);

            _scroll = GUILayout.BeginScrollView(_scroll, false, true);
            switch (_workspaceSelection)
            {
                case 1: DrawRobotInspector(); break;
                case 2: DrawEnvironmentTab(); break;
                case 3: DrawSurfaceTab(); break;
                case 4: DrawTerrainInspector(); break;
                case 5: DrawSystemTab(); break;
                default: DrawSceneInspector(); break;
            }
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        private string InspectorTitle()
        {
            switch (_workspaceSelection)
            {
                case 1: return "Unitree G1";
                case 2: return "Atmosphere";
                case 3: return "Snowpack / Ice";
                case 4: return "Everest Terrain";
                case 5: return "Simulation Backend";
                default: return "Everest Scene";
            }
        }

        private void DrawSceneInspector()
        {
            SectionStart("WORKSPACE");
            GUILayout.Label("Everest local physics + macro visual terrain", _label);
            GUILayout.Label("MuJoCo owns G1 dynamics. Newton owns deformable snow. Unity is renderer + editor UI.", _muted);
            SectionEnd();

            SectionStart("QUICK ACTIONS");
            if (GUILayout.Button("FOCUS ROBOT", _buttonActive))
                _camera?.SetMode(EverestCameraMode.Orbit);
            if (GUILayout.Button("RESET SIMULATION", _button))
                _backend.SendReset();
            SectionEnd();
        }

        private void DrawRobotInspector()
        {
            SectionStart("CONTROL");
            var cheat = _runtime != null && _runtime.CheatModeEnabled;
            if (GUILayout.Button(cheat ? "DISABLE CHEAT MODE" : "ENABLE CHEAT MODE", cheat ? _buttonActive : _button))
                _runtime?.SetCheatMode(!cheat);
            GUILayout.Label(cheat
                ? "NON-PHYSICAL root transport. A/D strafe · S/F back/forward · Q/E yaw. Newton material still receives the moved feet."
                : "Normal mode keeps the articulated G1 under MuJoCo + policy/contact dynamics.", _muted);
            GUILayout.Space(6f);
            var manual = _runtime != null && _runtime.ManualControlEnabled;
            GUILayout.Label(manual ? "Manual velocity control enabled" : "Manual commands locked",
                manual ? _statusGood : _muted);
            if (GUILayout.Button(manual ? "RELEASE CONTROL" : "TAKE CONTROL", manual ? _button : _buttonActive))
                _runtime?.SetManualControl(!manual);
            GUILayout.Label("WASD move · Q/E yaw · Space stop", _muted);
            SectionEnd();

            SectionStart("FOOT TELEMETRY");
            DrawFeet(_robot?.LatestFeet);
            SectionEnd();
        }

        private void DrawTerrainInspector()
        {
            SectionStart("BASE DEM · WIREFRAME");
            GUILayout.Label("HMA2 2 m DEM · render context only", _label);
            GUILayout.Label("MuJoCo still owns the full-resolution terrain collider.", _muted);
            SliderInt("Local LOD step", ref _localLod, 1, 16);
            SliderInt("Macro LOD step", ref _macroLod, 2, 32);
            GUILayout.Space(6f);
            if (GUILayout.Button("APPLY WIREFRAME LOD", _button))
                _terrain?.SetLod(_localLod, _macroLod);
            SectionEnd();

            SectionStart("ACTIVE MATERIAL RADIUS");
            Slider("Physics radius", ref _physicsRadius, 0.75f, 6f, "0.00", "m");
            Slider("Minimum MPM voxel", ref _mpmVoxelSize, 0.05f, 0.25f, "0.000", "m");
            SliderInt("Target cells / side", ref _physicsDetailCells, 24, 96);
            Slider("Recenter fraction", ref _patchRecenter, 0.25f, 0.75f, "0.00", "");
            GUILayout.Space(6f);
            if (GUILayout.Button("APPLY PHYSICS WINDOW", _buttonActive))
                ApplySimulationSettings();
            GUILayout.Label("Outside this moving radius the DEM stays wireframe/context; only the local material layer is snow/ice.", _muted);

            var state = _backend.LatestState;
            var newton = state?["newton"] as JObject;
            if (newton != null)
            {
                GUILayout.Label($"{newton.Value<int?>("particle_count") ?? 0:N0} MPM particles", _label);
                GUILayout.Label("Terrain-conforming Newton window around the G1", _muted);
            }
            SectionEnd();
        }

        private void DrawBottomStatus(Rect rect)
        {
            GUILayout.BeginArea(rect, _panel);
            GUILayout.BeginHorizontal();
            GUILayout.Label(_backend.Connected ? "● CONNECTED" : "○ OFFLINE",
                _backend.Connected ? _statusGood : _statusWarn, GUILayout.Width(110f));

            var state = _backend.LatestState;
            var simTime = state?.Value<float?>("sim_time") ?? 0f;
            GUILayout.Label($"SIM {simTime:0.00}s", _muted, GUILayout.Width(86f));
            GUILayout.Label($"MODE {_dataMode.ToUpperInvariant()}", _muted, GUILayout.Width(90f));

            if (_snow != null && _snow.Sequence >= 0)
                GUILayout.Label($"{_snow.SurfaceKind.ToUpperInvariant()} r={_physicsRadius:0.0}m · {_snow.LayerCount} layers · comp {_snow.MaxCompaction:P0}",
                    _muted, GUILayout.Width(250f));

            GUILayout.FlexibleSpace();
            GUILayout.Label(_runtime != null && _runtime.CheatModeEnabled
                ? "CHEAT: A/D strafe · S/F move · Q/E yaw"
                : "RMB orbit/look · wheel speed/zoom · P pause · R reset", _muted);
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        private void DrawEnvironmentTab()
        {
            SectionStart("SIMULATED ENVIRONMENT");
            if (_dataMode != "sim")
            {
                GUILayout.Label("LIVE input channel is armed. Switch to SIM to edit the local environment.", _muted);
                SectionEnd();
                return;
            }

            Slider("Temperature", ref _temperature, -45f, 8f, "0.0", "°C");
            Slider("Wind speed", ref _windSpeed, 0f, 45f, "0.0", "m/s");
            Slider("Wind direction", ref _windDirection, 0f, 360f, "0", "°");
            Slider("Snowfall", ref _snowfall, 0f, 60f, "0.0", "mm/h");
            Slider("Visibility", ref _visibility, 0.05f, 1f, "0.00", "");
            GUILayout.Space(7f);
            GUILayout.Label("VOLUMETRIC CLOUDS", _eyebrow);
            Slider("Cloud density", ref _cloudDensity, 0f, 1f, "0.00", "");
            Slider("Cloud coverage", ref _cloudCoverage, 0f, 1f, "0.00", "");
            Slider("Cloud radius", ref _cloudRadius, 15f, 600f, "0", "m");
            Slider("Cloud altitude", ref _cloudAltitude, 5f, 300f, "0", "m");
            Slider("Cloud thickness", ref _cloudThickness, 5f, 180f, "0", "m");
            Slider("Cloud speed", ref _cloudSpeed, 0f, 2f, "0.00", "");
            Slider("Cloud quality", ref _cloudQuality, 0f, 1f, "0.00", "");
            GUILayout.Space(8f);

            if (GUILayout.Button("APPLY ENVIRONMENT", _buttonActive))
                ApplyEnvironment();

            GUILayout.Label("Wind force remains backend/MuJoCo physics. Cloud controls round-trip through the backend environment channel and only affect rendering.", _muted);
            SectionEnd();
        }

        private void DrawSurfaceTab()
        {
            SectionStart("ACTIVE PHYSICS WINDOW");
            GUILayout.Label("This is the actual backend Newton/MuJoCo material radius, not a Unity-only render mask.", _muted);
            Slider("Physics radius", ref _physicsRadius, 0.75f, 6.0f, "0.00", "m");
            Slider("Minimum MPM voxel", ref _mpmVoxelSize, 0.05f, 0.20f, "0.000", "m");
            SliderInt("Target cells / side", ref _physicsDetailCells, 24, 80);
            Slider("Recenter fraction", ref _patchRecenter, 0.25f, 0.75f, "0.00", "");
            GUILayout.Space(6f);
            if (GUILayout.Button("APPLY PHYSICS RADIUS", _buttonActive))
                ApplySimulationSettings();
            var state = _backend.LatestState;
            var backendSettings = state?["simulation_settings"] as JObject;
            var newton = state?["newton"] as JObject;
            if (backendSettings != null)
            {
                var backendRadius = backendSettings.Value<float?>("physics_radius_m") ?? _physicsRadius;
                var effectiveVoxel = newton?.Value<float?>("voxel_size_m");
                var particles = newton?.Value<int?>("particle_count") ?? 0;
                GUILayout.Label(
                    effectiveVoxel.HasValue
                        ? $"Backend active: r={backendRadius:0.00} m · voxel={effectiveVoxel.Value:0.000} m · {particles:N0} particles"
                        : $"Backend active: r={backendRadius:0.00} m · {particles:N0} particles",
                    _statusGood);
            }
            SectionEnd();

            SectionStart("SURFACE MODE");
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("LAYERED MPM", _surface == "snow" ? _buttonActive : _button))
                SetSurface("snow");
            if (GUILayout.Button("RIGID ICE", _surface == "ice" ? _buttonActive : _button))
                SetSurface("ice");
            GUILayout.EndHorizontal();

            if (_surface == "ice")
            {
                Slider("Ice friction μ", ref _iceFriction, 0.01f, 0.45f, "0.00", "");
                GUILayout.Space(7f);
                if (GUILayout.Button("APPLY ICE CONTACT", _buttonActive))
                {
                    _backend.SendSurface("ice");
                    _backend.SendSurfaceFriction(_iceFriction);
                }
                GUILayout.Label("The active-radius ice layer is rendered from the backend. MuJoCo supplies the rigid contact/friction while the DEM remains the base collider.", _muted);
                SectionEnd();
                return;
            }

            Slider("Surface friction μ", ref _surfaceFriction, 0.05f, 0.90f, "0.00", "");
            SectionEnd();

            SectionStart("MULTILAYER SNOW / FIRN / ICE");
            GUILayout.BeginHorizontal();
            var layerNames = new string[_layers.Count];
            for (var i = 0; i < _layers.Count; i++) layerNames[i] = $"L{i + 1}";
            _selectedLayer = Mathf.Clamp(GUILayout.Toolbar(_selectedLayer, layerNames, _button), 0, _layers.Count - 1);

            if (GUILayout.Button("+", _button, GUILayout.Width(34f)) && _layers.Count < 6)
            {
                _layers.Add(NewSettledLayer());
                _selectedLayer = _layers.Count - 1;
            }
            if (GUILayout.Button("−", _button, GUILayout.Width(34f)) && _layers.Count > 1)
            {
                _layers.RemoveAt(_selectedLayer);
                _selectedLayer = Mathf.Clamp(_selectedLayer, 0, _layers.Count - 1);
            }
            GUILayout.EndHorizontal();

            var layer = _layers[_selectedLayer];
            GUILayout.Label($"Layer {_selectedLayer + 1} · top → bottom", _muted);

            var typeIndex = Mathf.Max(0, Array.IndexOf(LayerTypes, layer.Type));
            var nextTypeIndex = GUILayout.Toolbar(typeIndex, LayerTypes, _button);
            var nextType = LayerTypes[Mathf.Clamp(nextTypeIndex, 0, LayerTypes.Length - 1)];
            if (nextType != layer.Type) ApplyLayerPreset(layer, nextType);

            Slider("Thickness", ref layer.Thickness, 0.01f, 1.20f, "0.00", "m");
            Slider("Density", ref layer.Density, 60f, 950f, "0", "kg/m³");
            LogSlider("Stiffness", ref layer.Stiffness, 20000f, 100000000f, "Pa");
            LogSlider("Compression yield", ref layer.Compressive, 1000f, 10000000f, "Pa");
            LogSlider("Shear strength", ref layer.Shear, 300f, 5000000f, "Pa");
            Slider("Hardening", ref layer.Hardening, 0f, 40f, "0.0", "");
            LogSlider("Bond below", ref layer.Bond, 500f, 5000000f, "Pa");

            GUILayout.Space(8f);
            if (GUILayout.Button("APPLY TO NEWTON", _buttonActive))
                ApplySnow();

            GUILayout.Label("ICE layers use the same backend Newton MPM path with their own density/stiffness/yield values. Rigid Ice above is the cheaper non-deforming shortcut.", _muted);
            SectionEnd();
        }

        private void DrawSystemTab()
        {
            SectionStart("PLAYBACK");
            var paused = _runtime == null || _runtime.Paused;
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "RUN PHYSICS" : "PAUSE", paused ? _buttonActive : _button))
                _backend.SendPause(!paused);
            if (GUILayout.Button("RESET", _button))
                _backend.SendReset();
            GUILayout.EndHorizontal();
            SectionEnd();

            SectionStart("EDITOR UI");
            var nextUiScale = GUILayout.HorizontalSlider(_uiScale, 0.65f, 1.60f);
            if (Mathf.Abs(nextUiScale - _uiScale) > 0.002f) SetUiScale(nextUiScale);
            GUILayout.BeginHorizontal();
            GUILayout.Label($"UI scale {_uiScale * 100f:0}%", _label);
            if (GUILayout.Button("−", _button, GUILayout.Width(32f))) SetUiScale(_uiScale - 0.10f);
            if (GUILayout.Button("100%", _button, GUILayout.Width(54f))) SetUiScale(1f);
            if (GUILayout.Button("+", _button, GUILayout.Width(32f))) SetUiScale(_uiScale + 0.10f);
            GUILayout.EndHorizontal();
            GUILayout.Label("UI scaling is renderer-local and never changes backend physics.", _muted);
            SectionEnd();

            SectionStart("CHEAT TRANSPORT");
            var cheat = _runtime != null && _runtime.CheatModeEnabled;
            if (GUILayout.Button(cheat ? "CHEAT MODE ON" : "CHEAT MODE OFF", cheat ? _buttonActive : _button))
                _runtime?.SetCheatMode(!cheat);
            Slider("Slide speed", ref _cheatSpeed, 0.1f, 5f, "0.00", "m/s");
            Slider("Yaw rate", ref _cheatYawRate, 0.1f, 4f, "0.00", "rad/s");
            if (GUILayout.Button("APPLY CHEAT SPEED", _button))
                ApplySimulationSettings();
            GUILayout.Label("Cheat mode is explicitly non-physical: the floating base is translated directly while the local snow/ice renderer and Newton contact window follow it.", _muted);
            SectionEnd();

            SectionStart("BACKEND");
            var state = _backend.LatestState;
            if (state != null)
            {
                GUILayout.Label($"Data mode: {state.Value<string>("data_mode") ?? _dataMode}", _label);
                GUILayout.Label($"Surface: {state.Value<string>("surface") ?? _surface}", _label);
                var newton = state["newton"] as JObject;
                if (newton != null)
                {
                    GUILayout.Label($"{newton.Value<string>("solver")} · {newton.Value<string>("device")}", _muted);
                    GUILayout.Label($"{newton.Value<int?>("particle_count") ?? 0:N0} particles · step {newton.Value<int?>("steps") ?? 0}", _muted);
                }
            }

            var ack = _backend.LatestControlAck;
            if (ack != null)
                GUILayout.Label(ack.Value<bool?>("ok") == true ? "Last control acknowledged" : $"Control error: {ack.Value<string>("message")}",
                    ack.Value<bool?>("ok") == true ? _statusGood : _statusWarn);
            SectionEnd();

            SectionStart("SHORTCUTS");
            GUILayout.Label("Robot: WASD · Q/E yaw · Space stop", _muted);
            GUILayout.Label("Cheat: A/D strafe · S/F back/forward · Q/E yaw", _muted);
            GUILayout.Label("Orbit: RMB drag · wheel zoom", _muted);
            GUILayout.Label("Free: RMB look · Shift+WASD · Shift+Q/E vertical", _muted);
            GUILayout.Label("P pause · R reset", _muted);
            SectionEnd();
        }

        private void SectionStart(string title)
        {
            GUILayout.BeginVertical(_section);
            GUILayout.Label(title, _eyebrow);
        }

        private static void SectionEnd()
        {
            GUILayout.EndVertical();
        }

        private void Slider(string label, ref float value, float min, float max, string format, string suffix)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, _label, GUILayout.Width(138f));
            value = GUILayout.HorizontalSlider(value, min, max, GUILayout.ExpandWidth(true));
            GUILayout.Label($"{value.ToString(format)} {suffix}", _muted, GUILayout.Width(88f));
            GUILayout.EndHorizontal();
        }

        private void SliderInt(string label, ref int value, int min, int max)
        {
            float asFloat = value;
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, _label, GUILayout.Width(138f));
            asFloat = GUILayout.HorizontalSlider(asFloat, min, max, GUILayout.ExpandWidth(true));
            value = Mathf.Clamp(Mathf.RoundToInt(asFloat), min, max);
            GUILayout.Label(value.ToString(), _muted, GUILayout.Width(88f));
            GUILayout.EndHorizontal();
        }

        private void LogSlider(string label, ref float value, float min, float max, string suffix)
        {
            var logMin = Mathf.Log10(min);
            var logMax = Mathf.Log10(max);
            var log = Mathf.Log10(Mathf.Clamp(value, min, max));

            GUILayout.BeginHorizontal();
            GUILayout.Label(label, _label, GUILayout.Width(138f));
            log = GUILayout.HorizontalSlider(log, logMin, logMax, GUILayout.ExpandWidth(true));
            value = Mathf.Pow(10f, log);
            GUILayout.Label(FormatEngineering(value, suffix), _muted, GUILayout.Width(88f));
            GUILayout.EndHorizontal();
        }

        private static string FormatEngineering(float value, string suffix)
        {
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
                    layer.Density = 120f;
                    layer.Stiffness = 35000f;
                    layer.Compressive = 3500f;
                    layer.Shear = 1800f;
                    layer.Hardening = 12f;
                    layer.Bond = 2500f;
                    break;
                case "WIND_PACK":
                    layer.Density = 360f;
                    layer.Stiffness = 750000f;
                    layer.Compressive = 85000f;
                    layer.Shear = 32000f;
                    layer.Hardening = 22f;
                    layer.Bond = 12000f;
                    break;
                case "CRUST":
                    layer.Density = 460f;
                    layer.Stiffness = 1800000f;
                    layer.Compressive = 180000f;
                    layer.Shear = 70000f;
                    layer.Hardening = 28f;
                    layer.Bond = 9000f;
                    break;
                case "FIRN":
                    layer.Density = 650f;
                    layer.Stiffness = 6500000f;
                    layer.Compressive = 650000f;
                    layer.Shear = 220000f;
                    layer.Hardening = 32f;
                    layer.Bond = 180000f;
                    break;
                case "ICE":
                    layer.Density = 917f;
                    layer.Stiffness = 60000000f;
                    layer.Compressive = 5000000f;
                    layer.Shear = 1800000f;
                    layer.Hardening = 8f;
                    layer.Bond = 900000f;
                    break;
                default:
                    layer.Density = 320f;
                    layer.Stiffness = 500000f;
                    layer.Compressive = 55000f;
                    layer.Shear = 20000f;
                    layer.Hardening = 18f;
                    layer.Bond = 7500f;
                    break;
            }
        }

        private void DrawFeet(JObject feet)
        {
            if (feet == null)
            {
                GUILayout.Label("Waiting for foot telemetry…", _muted);
                return;
            }
            DrawFoot("Left", feet["left"] as JObject);
            DrawFoot("Right", feet["right"] as JObject);
        }

        private void DrawFoot(string prefix, JObject foot)
        {
            if (foot == null) return;
            var contact = foot.Value<bool?>("contact") ?? false;
            var force = foot.Value<float?>("normal_force_n") ?? 0f;
            var sink = foot.Value<float?>("penetration_m") ?? 0f;
            var slip = foot.Value<float?>("slip_speed_m_s") ?? 0f;
            GUILayout.Label($"{prefix} · {(contact ? "CONTACT" : "air")} · {force:0} N", contact ? _label : _muted);
            GUILayout.Label($"sink {sink * 100f:0.0} cm · slip {slip:0.000} m/s", _muted);
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
                ["patch_recenter_fraction"] = _patchRecenter,
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
            _dataMode = mode;
            _backend.SendMode(mode);
        }

        private void SyncDraftFromBackendOnce()
        {
            if (_draftInitialized) return;

            var state = _backend.LatestState;
            if (state != null)
            {
                _dataMode = state.Value<string>("data_mode") ?? _dataMode;
                _surface = state.Value<string>("surface") ?? _surface;
                var settings = state["simulation_settings"] as JObject;
                if (settings != null)
                {
                    _physicsRadius = settings.Value<float?>("physics_radius_m") ?? _physicsRadius;
                    _mpmVoxelSize = settings.Value<float?>("mpm_min_voxel_size_m")
                        ?? settings.Value<float?>("mpm_voxel_size_m")
                        ?? _mpmVoxelSize;
                    _physicsDetailCells = settings.Value<int?>("physics_detail_cells") ?? _physicsDetailCells;
                    _patchRecenter = settings.Value<float?>("patch_recenter_fraction") ?? _patchRecenter;
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

            _draftInitialized = state != null && env != null && snow != null;
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
                Type = "DENSE_SNOW",
                Label = "Settled snow",
                Thickness = 0.32f,
                Density = 320f,
                Stiffness = 500000f,
                Compressive = 55000f,
                Shear = 20000f,
                Hardening = 18f,
                Bond = 7500f
            };
        }
    }
}
