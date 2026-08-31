using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace EverestSim
{
    /// <summary>
    /// Renders only the active backend material layer: Newton multilayer snow/firn/ice
    /// or a rigid MuJoCo ice patch. The DEM itself is rendered separately as wireframe.
    /// </summary>
    public sealed class EverestSnowRenderer : MonoBehaviour
    {
        private GameObject _surfaceObject;
        private Mesh _mesh;
        private MeshRenderer _renderer;
        private GameObject _volumeObject;
        private Mesh _volumeMesh;
        private MeshRenderer _volumeRenderer;
        private Material _material;
        private Material _layerMaterial;
        private readonly List<GameObject> _historyObjects = new List<GameObject>();
        private long _historySequence = -1;
        private int _width;
        private int _height;
        private long _sequence = -1;
        private long _sourceEpoch = -1;
        private Vector3[] _targetVertices;
        private Vector3[] _visualBaseVertices;
        private Color[] _targetColors;
        private Vector3[] _targetVolumeVertices;
        private Color[] _targetVolumeColors;
        private bool _hasSurface;
        private bool _visualOnly;
        private readonly List<Vector3> _visualFeet = new List<Vector3>();

        public int LayerCount { get; private set; }
        public float SurfaceDepthM { get; private set; }
        public float SurfaceFriction { get; private set; }
        public float MaxCompaction { get; private set; }
        public float MaxSinkageM { get; private set; }
        public float PredictedStaticSinkageM { get; private set; }
        public float DepositedDepthM { get; private set; }
        public bool NewtonActive { get; private set; }
        public string SurfaceKind { get; private set; } = "snow";
        public long Sequence => _sequence;

        public void SetSourceEpoch(long epoch)
        {
            if (epoch == _sourceEpoch) return;
            _sourceEpoch = epoch;
            _sequence = -1;
            _historySequence = -1;
            _hasSurface = false;
            if (_surfaceObject != null) _surfaceObject.SetActive(false);
            if (_volumeObject != null) _volumeObject.SetActive(false);
            foreach (var historyObject in _historyObjects)
                if (historyObject != null) historyObject.SetActive(false);
        }

        private void Awake()
        {
            _surfaceObject = new GameObject("Active Snow / Ice Surface");
            _surfaceObject.transform.SetParent(transform, false);
            _surfaceObject.AddComponent<MeshFilter>();
            _renderer = _surfaceObject.AddComponent<MeshRenderer>();
            _renderer.shadowCastingMode = ShadowCastingMode.On;
            _renderer.receiveShadows = true;

            _volumeObject = new GameObject("Active Multilayer Snow Volume");
            _volumeObject.transform.SetParent(transform, false);
            _volumeObject.AddComponent<MeshFilter>();
            _volumeRenderer = _volumeObject.AddComponent<MeshRenderer>();
            // The skirt is a visualization of the live layer boundaries. It
            // must not self-shadow the top surface into a black cutout in the
            // forward WebGL renderer.
            _volumeRenderer.shadowCastingMode = ShadowCastingMode.Off;
            _volumeRenderer.receiveShadows = false;
            _volumeObject.SetActive(false);

            _material = EverestRuntimeMaterials.Load("EverestTerrain", "Everest/Terrain");
            _layerMaterial = EverestRuntimeMaterials.Load("EverestSnowLayers", "Everest/SnowLayers");
            if (_material != null && _material.shader != null && _material.shader.name == "Everest/Terrain")
            {
                var albedo = Resources.Load<Texture2D>("Textures/Everest/snow_field_aerial_col_2k");
                var roughness = Resources.Load<Texture2D>("Textures/Everest/snow_field_aerial_rough_2k");
                var rock = Resources.Load<Texture2D>("Textures/Everest/rocky_terrain_02_diff_2k");
                var rockRoughness = Resources.Load<Texture2D>("Textures/Everest/rocky_terrain_02_rough_2k");
                if (albedo != null) _material.SetTexture("_SnowTex", albedo);
                if (roughness != null) _material.SetTexture("_SnowRoughness", roughness);
                if (rock != null) _material.SetTexture("_RockTex", rock);
                if (rockRoughness != null) _material.SetTexture("_RockRoughness", rockRoughness);
                _material.SetFloat("_SnowScale", 0.0075f);
                _material.SetFloat("_RockScale", 0.055f);
                _material.SetFloat("_MacroScale", 0.010f);
                _material.SetFloat("_Stylize", 0.36f);
                _material.SetFloat("_ActiveMaterial", 1f);
            }
            if (_material != null)
            {
                _renderer.sharedMaterial = _material;
            }
            if (_layerMaterial != null) _volumeRenderer.sharedMaterial = _layerMaterial;
        }

        public void OnSnow(JObject surface)
        {
            SetSourceEpoch(surface.Value<long?>("source_epoch") ?? _sourceEpoch);
            var sequence = surface.Value<long?>("sequence") ?? -1;
            if (sequence <= _sequence) return;
            _sequence = sequence;

            SurfaceKind = (surface.Value<string>("surface_kind") ?? "snow").ToLowerInvariant();
            SurfaceDepthM = surface.Value<float?>("surface_depth_m") ?? 0f;
            SurfaceFriction = surface.Value<float?>("surface_friction") ?? 0f;
            var layers = surface["layers"] as JArray;
            LayerCount = layers?.Count ?? 0;
            var mpm = surface["mpm"] as JObject;
            NewtonActive = mpm?.Value<bool?>("active") ?? false;
            MaxSinkageM = mpm?.Value<float?>("max_sinkage_m") ?? 0f;
            PredictedStaticSinkageM = mpm?.Value<float?>("predicted_static_sinkage_m") ?? 0f;
            DepositedDepthM = mpm?.Value<float?>("deposited_depth_m") ?? 0f;
            if (_material != null && _material.HasProperty("_SurfaceMode"))
                _material.SetFloat("_SurfaceMode", SurfaceKind == "ice" ? 1f : SurfaceKind == "rock" ? 2f : 0f);
            _surfaceObject.SetActive(true);

            var resolution = surface["resolution"] as JArray;
            var origin = surface["origin"] as JArray;
            var size = surface["size"] as JArray;
            var heights = surface["heights"] as JArray;
            var backendVertices = surface["vertices"] as JArray;
            var compaction = surface["compaction"] as JArray;
            var materialIds = surface["material_ids"] as JArray;
            var baseHeights = surface["base_heights"] as JArray;
            var layerHeights = surface["layer_heights"] as JArray;
            var layerVertices = surface["layer_vertices"] as JArray;
            var substrateVertices = surface["substrate_vertices"] as JArray;
            if (resolution == null || origin == null || size == null || heights == null) return;

            var width = resolution[0].Value<int>();
            var height = resolution[1].Value<int>();
            if (width < 2 || height < 2 || heights.Count < width * height) return;

            var layerColors = BuildLayerColors(layers);
            EnsureMesh(width, height);
            if (_targetVertices == null || _targetVertices.Length != width * height)
                _targetVertices = new Vector3[width * height];
            if (_visualBaseVertices == null || _visualBaseVertices.Length != width * height)
                _visualBaseVertices = new Vector3[width * height];
            if (_targetColors == null || _targetColors.Length != width * height)
                _targetColors = new Color[width * height];

            var backendOriginX = origin[0].Value<float>();
            var backendOriginY = origin[1].Value<float>();
            var sizeX = size[0].Value<float>();
            var sizeY = size[1].Value<float>();
            Shader.SetGlobalVector(
                "_EverestActiveCenter",
                new Vector4(backendOriginX + sizeX * 0.5f, 0f, backendOriginY + sizeY * 0.5f, 0f));
            Shader.SetGlobalFloat("_EverestActiveRadius", Mathf.Min(sizeX, sizeY) * 0.5f);
            var dx = sizeX / width;
            var dz = sizeY / height;
            MaxCompaction = 0f;

            for (var row = 0; row < height; ++row)
            {
                for (var col = 0; col < width; ++col)
                {
                    var i = row * width + col;
                    var backendX = backendOriginX + (col + 0.5f) * dx;
                    var backendY = backendOriginY + (row + 0.5f) * dz;
                    var backendZ = heights[i].Value<float>();
                    _visualBaseVertices[i] = ReadBackendVertex(
                        backendVertices,
                        i,
                        new Vector3(backendX, backendZ, backendY)) + Vector3.up * 0.004f;
                    _targetVertices[i] = _visualBaseVertices[i];

                    var compact = compaction != null && i < compaction.Count
                        ? Mathf.Clamp01(compaction[i].Value<float>())
                        : 0f;
                    MaxCompaction = Mathf.Max(MaxCompaction, compact);

                    var materialId = materialIds != null && i < materialIds.Count
                        ? Mathf.Max(0, materialIds[i].Value<int>())
                        : 0;
                    var layerColor = materialId < layerColors.Count
                        ? layerColors[materialId]
                        : (SurfaceKind == "ice" ? new Color(0.28f, 0.68f, 0.86f) : Color.white);
                    _targetColors[i] = new Color(layerColor.r, layerColor.g, layerColor.b, compact);
                }
            }

            UpdateVolumeTargets(
                width,
                height,
                backendOriginX,
                backendOriginY,
                sizeX,
                sizeY,
                heights,
                baseHeights,
                layerHeights,
                layerVertices,
                substrateVertices,
                layerColors);
            ApplyVisualOnlyFootprints();

            if (!_hasSurface)
            {
                _mesh.vertices = _targetVertices;
                _mesh.colors = _targetColors;
                _mesh.RecalculateNormals();
                _mesh.RecalculateBounds();
                _hasSurface = true;
            }
        }

        public void SetVisualOnly(bool enabled)
        {
            if (_visualOnly == enabled) return;
            _visualOnly = enabled;
            ApplyVisualOnlyFootprints();
        }

        public void OnFeet(JObject feet)
        {
            _visualFeet.Clear();
            if (feet != null)
            {
                foreach (var side in new[] { "left", "right" })
                {
                    var foot = feet[side] as JObject;
                    if (foot?["position"] != null)
                        _visualFeet.Add(EverestCoordinates.Position(foot["position"]));
                }
            }
            if (_visualOnly) ApplyVisualOnlyFootprints();
        }

        private void ApplyVisualOnlyFootprints()
        {
            if (_targetVertices == null || _visualBaseVertices == null
                || _targetVertices.Length != _visualBaseVertices.Length) return;
            for (var i = 0; i < _targetVertices.Length; ++i)
            {
                var vertex = _visualBaseVertices[i];
                var sink = 0f;
                if (_visualOnly)
                {
                    foreach (var foot in _visualFeet)
                    {
                        var distance = Vector2.Distance(
                            new Vector2(vertex.x, vertex.z),
                            new Vector2(foot.x, foot.z));
                        var influence = Mathf.Clamp01(1f - distance / 0.19f);
                        sink = Mathf.Max(sink, 0.038f * influence * influence);
                    }
                }
                _targetVertices[i] = vertex + Vector3.down * sink;
            }
        }

        public void OnSnowHistory(JObject history)
        {
            var sequence = history.Value<long?>("sequence") ?? -1;
            if (sequence == _historySequence) return;
            _historySequence = sequence;
            foreach (var item in _historyObjects)
                if (item != null) Destroy(item);
            _historyObjects.Clear();

            var patches = history["patches"] as JArray;
            if (patches == null || patches.Count == 0)
            {
                Shader.SetGlobalInt("_EverestHistoryCount", 0);
                return;
            }

            const int maxVisiblePatches = 16;
            var first = Mathf.Max(0, patches.Count - maxVisiblePatches);
            var centers = new Vector4[maxVisiblePatches];
            var visible = 0;
            for (var patchIndex = first; patchIndex < patches.Count; ++patchIndex)
            {
                if (!(patches[patchIndex] is JObject patch)) continue;
                var resolution = patch["resolution"] as JArray;
                var origin = patch["origin"] as JArray;
                var size = patch["size"] as JArray;
                var heights = patch["heights"] as JArray;
                var backendVertices = patch["vertices"] as JArray;
                var compaction = patch["compaction"] as JArray;
                var materialIds = patch["material_ids"] as JArray;
                var layers = patch["layers"] as JArray;
                if (resolution == null || origin == null || size == null || heights == null) continue;
                var width = resolution[0].Value<int>();
                var height = resolution[1].Value<int>();
                if (width < 2 || height < 2 || heights.Count < width * height) continue;

                var colorsByLayer = BuildLayerColors(layers);
                var vertices = new Vector3[width * height];
                var colors = new Color[vertices.Length];
                var triangles = new int[(width - 1) * (height - 1) * 6];
                var originX = origin[0].Value<float>();
                var originY = origin[1].Value<float>();
                var sizeX = size[0].Value<float>();
                var sizeY = size[1].Value<float>();
                for (var row = 0; row < height; ++row)
                for (var col = 0; col < width; ++col)
                {
                    var i = row * width + col;
                    var fallback = new Vector3(
                        originX + (col + 0.5f) * sizeX / width,
                        heights[i].Value<float>() + 0.004f,
                        originY + (row + 0.5f) * sizeY / height);
                    vertices[i] = ReadBackendVertex(backendVertices, i, fallback - Vector3.up * 0.004f)
                        + Vector3.up * 0.004f;
                    var materialId = materialIds != null && i < materialIds.Count ? Mathf.Max(0, materialIds[i].Value<int>()) : 0;
                    var color = materialId < colorsByLayer.Count ? colorsByLayer[materialId] : Color.white;
                    var compact = compaction != null && i < compaction.Count ? Mathf.Clamp01(compaction[i].Value<float>()) : 0f;
                    colors[i] = new Color(color.r, color.g, color.b, compact);
                }
                var t = 0;
                for (var row = 0; row < height - 1; ++row)
                for (var col = 0; col < width - 1; ++col)
                {
                    var i = row * width + col;
                    triangles[t++] = i; triangles[t++] = i + width; triangles[t++] = i + 1;
                    triangles[t++] = i + 1; triangles[t++] = i + width; triangles[t++] = i + width + 1;
                }
                var mesh = new Mesh { name = $"Persistent Newton Snow Trail {patchIndex}" };
                mesh.vertices = vertices;
                mesh.colors = colors;
                mesh.triangles = triangles;
                mesh.RecalculateNormals();
                mesh.RecalculateBounds();
                var go = new GameObject(mesh.name);
                go.transform.SetParent(transform, false);
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                var renderer = go.AddComponent<MeshRenderer>();
                renderer.sharedMaterial = _material;
                renderer.shadowCastingMode = ShadowCastingMode.On;
                renderer.receiveShadows = true;
                _historyObjects.Add(go);
                centers[visible++] = new Vector4(originX + sizeX * 0.5f, originY + sizeY * 0.5f, Mathf.Min(sizeX, sizeY) * 0.5f, 0f);
            }
            Shader.SetGlobalVectorArray("_EverestHistoryCenters", centers);
            Shader.SetGlobalInt("_EverestHistoryCount", visible);
        }

        private void UpdateVolumeTargets(
            int width,
            int height,
            float originX,
            float originY,
            float sizeX,
            float sizeY,
            JArray surfaceHeights,
            JArray baseHeights,
            JArray layerHeights,
            JArray layerVertices,
            JArray substrateVertices,
            List<Color> layerColors)
        {
            var layerCount = Mathf.Min(layerColors.Count, layerVertices?.Count ?? 0);
            if (!NewtonActive || SurfaceKind != "snow" || layerCount == 0)
            {
                _volumeObject.SetActive(false);
                return;
            }

            var ring = BuildBoundaryRing(width, height);
            var vertices = new Vector3[layerCount * ring.Count * 2];
            var colors = new Color[vertices.Length];
            var triangles = new int[layerCount * ring.Count * 6];

            for (var layer = 0; layer < layerCount; ++layer)
            {
                var upper = layerVertices[layer] as JArray;
                var lower = layer + 1 < layerCount
                    ? layerVertices[layer + 1] as JArray
                    : substrateVertices;
                var upperHeights = layer == 0 ? surfaceHeights : layerHeights?[layer] as JArray;
                var lowerHeights = layer + 1 < layerCount ? layerHeights?[layer + 1] as JArray : null;

                for (var segment = 0; segment < ring.Count; ++segment)
                {
                    var sample = ring[segment];
                    var col = sample % width;
                    var row = sample / width;
                    var fallbackX = originX + (col + 0.5f) * sizeX / width;
                    var fallbackY = originY + (row + 0.5f) * sizeY / height;
                    var fallbackTop = upperHeights != null && sample < upperHeights.Count
                        ? upperHeights[sample].Value<float>()
                        : surfaceHeights[sample].Value<float>();
                    var fallbackBottom = lowerHeights != null && sample < lowerHeights.Count
                        ? lowerHeights[sample].Value<float>()
                        : (baseHeights != null && sample < baseHeights.Count
                            ? baseHeights[sample].Value<float>() - SurfaceDepthM
                            : fallbackTop - SurfaceDepthM);

                    var top = ReadBackendVertex(
                        upper,
                        sample,
                        new Vector3(fallbackX, fallbackTop, fallbackY));
                    var bottom = ReadBackendVertex(
                        lower,
                        sample,
                        new Vector3(fallbackX, fallbackBottom, fallbackY));
                    var vertex = (layer * ring.Count + segment) * 2;
                    vertices[vertex] = top - Vector3.up * 0.002f;
                    vertices[vertex + 1] = bottom;
                    colors[vertex] = colors[vertex + 1] = new Color(
                        layerColors[layer].r,
                        layerColors[layer].g,
                        layerColors[layer].b,
                        0f);

                    var next = (layer * ring.Count + (segment + 1) % ring.Count) * 2;
                    var triangle = (layer * ring.Count + segment) * 6;
                    triangles[triangle] = vertex;
                    triangles[triangle + 1] = next;
                    triangles[triangle + 2] = vertex + 1;
                    triangles[triangle + 3] = vertex + 1;
                    triangles[triangle + 4] = next;
                    triangles[triangle + 5] = next + 1;
                }
            }

            if (_volumeMesh == null || _volumeMesh.vertexCount != vertices.Length)
            {
                if (_volumeMesh != null) Destroy(_volumeMesh);
                _volumeMesh = new Mesh { name = "Live Multilayer Snow Volume" };
                _volumeMesh.MarkDynamic();
                _volumeMesh.indexFormat = vertices.Length > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
                _volumeMesh.vertices = vertices;
                _volumeMesh.colors = colors;
                _volumeMesh.triangles = triangles;
                _volumeObject.GetComponent<MeshFilter>().sharedMesh = _volumeMesh;
            }
            _targetVolumeVertices = vertices;
            _targetVolumeColors = colors;
            if (!_volumeObject.activeSelf)
            {
                _volumeMesh.vertices = vertices;
                _volumeMesh.colors = colors;
                _volumeMesh.RecalculateNormals();
                _volumeMesh.RecalculateBounds();
            }
            _volumeObject.SetActive(true);
        }

        private static List<int> BuildBoundaryRing(int width, int height)
        {
            var ring = new List<int>(2 * width + 2 * height - 4);
            for (var col = 0; col < width; ++col) ring.Add(col);
            for (var row = 1; row < height; ++row) ring.Add(row * width + width - 1);
            for (var col = width - 2; col >= 0; --col) ring.Add((height - 1) * width + col);
            for (var row = height - 2; row > 0; --row) ring.Add(row * width);
            return ring;
        }

        private static Vector3 ReadBackendVertex(JArray points, int index, Vector3 fallback)
        {
            if (points == null || index < 0 || index >= points.Count || !(points[index] is JArray point) || point.Count < 3)
                return fallback;
            // Backend is right-handed Z-up; Unity stores the backend Y axis in
            // transform Z. Keep the actual Newton X/Y displacement instead of
            // rebuilding a fixed horizontal grid from height samples.
            return new Vector3(
                point[0].Value<float>(),
                point[2].Value<float>(),
                point[1].Value<float>());
        }

        private static List<Color> BuildLayerColors(JArray layers)
        {
            var result = new List<Color>();
            if (layers == null) return result;
            foreach (var token in layers)
            {
                var layer = token as JObject;
                var color = layer?["color"] as JArray;
                if (color != null && color.Count >= 3)
                    result.Add(new Color(color[0].Value<float>(), color[1].Value<float>(), color[2].Value<float>()));
                else
                    result.Add(Color.white);
            }
            return result;
        }

        private void Update()
        {
            if (!_hasSurface || _mesh == null || _targetVertices == null) return;
            var vertices = _mesh.vertices;
            var colors = _mesh.colors;
            if (vertices.Length != _targetVertices.Length) return;
            if (colors == null || colors.Length != vertices.Length) colors = new Color[vertices.Length];

            // Newton-on targets are backend-authored. In explicit visual-only
            // mode the target also includes bounded, non-physical boot dimples.
            var blend = 1f - Mathf.Exp(-20f * Time.unscaledDeltaTime);
            var geometryDirty = false;
            var colorDirty = false;
            for (var i = 0; i < vertices.Length; ++i)
            {
                var next = Vector3.LerpUnclamped(vertices[i], _targetVertices[i], blend);
                geometryDirty |= (next - vertices[i]).sqrMagnitude > 1e-10f;
                vertices[i] = next;
                var nextColor = Color.LerpUnclamped(colors[i], _targetColors[i], blend);
                colorDirty |= Mathf.Abs(nextColor.r - colors[i].r) > 1e-4f
                    || Mathf.Abs(nextColor.g - colors[i].g) > 1e-4f
                    || Mathf.Abs(nextColor.b - colors[i].b) > 1e-4f
                    || Mathf.Abs(nextColor.a - colors[i].a) > 1e-4f;
                colors[i] = nextColor;
            }
            if (geometryDirty)
            {
                _mesh.vertices = vertices;
                _mesh.RecalculateNormals();
                _mesh.RecalculateBounds();
            }
            if (colorDirty) _mesh.colors = colors;

            if (_volumeObject.activeSelf && _volumeMesh != null && _targetVolumeVertices != null)
            {
                var volumeVertices = _volumeMesh.vertices;
                var volumeDirty = false;
                for (var i = 0; i < volumeVertices.Length && i < _targetVolumeVertices.Length; ++i)
                {
                    var next = Vector3.LerpUnclamped(volumeVertices[i], _targetVolumeVertices[i], blend);
                    volumeDirty |= (next - volumeVertices[i]).sqrMagnitude > 1e-10f;
                    volumeVertices[i] = next;
                }
                if (volumeDirty)
                {
                    _volumeMesh.vertices = volumeVertices;
                    _volumeMesh.RecalculateNormals();
                    _volumeMesh.RecalculateBounds();
                }
                if (_targetVolumeColors != null && _targetVolumeColors.Length == volumeVertices.Length)
                    _volumeMesh.colors = _targetVolumeColors;
            }
        }

        private void EnsureMesh(int width, int height)
        {
            if (_mesh != null && width == _width && height == _height) return;
            if (_mesh != null) Destroy(_mesh);

            _width = width;
            _height = height;
            _hasSurface = false;
            _targetVertices = null;
            _visualBaseVertices = null;
            _targetColors = null;

            var vertices = new Vector3[width * height];
            var uv = new Vector2[vertices.Length];
            var triangles = new int[(width - 1) * (height - 1) * 6];
            for (var row = 0; row < height; ++row)
            {
                for (var col = 0; col < width; ++col)
                {
                    var i = row * width + col;
                    uv[i] = new Vector2(col / (float)(width - 1), row / (float)(height - 1));
                }
            }

            var t = 0;
            for (var row = 0; row < height - 1; ++row)
            {
                for (var col = 0; col < width - 1; ++col)
                {
                    var i = row * width + col;
                    triangles[t++] = i;
                    triangles[t++] = i + width;
                    triangles[t++] = i + 1;
                    triangles[t++] = i + 1;
                    triangles[t++] = i + width;
                    triangles[t++] = i + width + 1;
                }
            }

            _mesh = new Mesh { name = "Active Snow / Ice Surface" };
            _mesh.indexFormat = vertices.Length > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
            _mesh.MarkDynamic();
            _mesh.vertices = vertices;
            _mesh.uv = uv;
            _mesh.triangles = triangles;
            _surfaceObject.GetComponent<MeshFilter>().sharedMesh = _mesh;
        }
    }
}
