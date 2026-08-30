using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace EverestSim
{
    /// <summary>
    /// Cheap DEM-draped material shell outside the active Newton window.
    /// Geometry comes only from the authoritative Everest DEM. The shell is
    /// visual LOD; MuJoCo/Newton remain the only physics owners.
    /// </summary>
    public sealed class EverestVisualTerrainRenderer : MonoBehaviour
    {
        private readonly List<Material> _materials = new List<Material>();
        private JObject _localData;
        private JObject _macroData;
        private GameObject _local;
        private GameObject _macro;
        private int _localLodStep = 1;
        private int _macroLodStep = 7;
        private float _snowDepth = 0.40f;
        private float _surfaceMode;
        private float _snowfall;
        private float _temperature = -18f;

        public void OnLocalTerrain(JObject terrain)
        {
            _localData = terrain;
            RebuildLocal();
        }

        public void OnMacroTerrain(JObject terrain)
        {
            _macroData = terrain;
            RebuildMacro();
        }

        public void OnSnow(JObject surface)
        {
            _snowDepth = Mathf.Clamp(surface.Value<float?>("surface_depth_m") ?? _snowDepth, 0f, 2.5f);
            var surfaceKind = (surface.Value<string>("surface_kind") ?? "snow").ToLowerInvariant();
            _surfaceMode = surfaceKind == "ice" ? 1f : surfaceKind == "rock" ? 2f : 0f;

            var layers = surface["layers"] as JArray;
            if (_surfaceMode < 0.5f && layers != null && layers.Count > 0 && layers[0] is JObject top)
            {
                var type = (top.Value<string>("type") ?? string.Empty).ToUpperInvariant();
                if (type == "ICE") _surfaceMode = 1f;
            }
            UpdateMaterials();
        }

        public void OnEnvironment(JObject environment)
        {
            _snowfall = Mathf.Max(0f, environment.Value<float?>("snowfall_mm_h") ?? _snowfall);
            _temperature = environment.Value<float?>("temperature_c") ?? _temperature;
            UpdateMaterials();
        }

        public void SetLod(int localStep, int macroStep)
        {
            localStep = Mathf.Clamp(localStep, 1, 16);
            macroStep = Mathf.Clamp(macroStep, 2, 32);
            if (localStep == _localLodStep && macroStep == _macroLodStep) return;
            _localLodStep = localStep;
            _macroLodStep = macroStep;
            RebuildLocal();
            RebuildMacro();
        }

        private void RebuildLocal()
        {
            if (_localData == null) return;
            if (_local != null) Destroy(_local);
            _local = BuildLod("Everest Visual Snow · Local", _localData, _localLodStep, false);
        }

        private void RebuildMacro()
        {
            if (_macroData == null) return;
            if (_macro != null) Destroy(_macro);
            _macro = BuildLod("Everest Visual Snow · Macro", _macroData, _macroLodStep, true);
        }

        private GameObject BuildLod(string name, JObject terrain, int baseStep, bool macro)
        {
            var parent = new GameObject(name);
            parent.transform.SetParent(transform, false);
            var multipliers = new[] { 1, 2, 4 };
            var thresholds = macro
                ? new[] { 0.20f, 0.055f, 0.012f }
                : new[] { 0.36f, 0.11f, 0.025f };
            var lods = new LOD[multipliers.Length];

            for (var i = 0; i < multipliers.Length; ++i)
            {
                var child = BuildMesh(
                    $"{name} · LOD{i}",
                    terrain,
                    Mathf.Clamp(baseStep * multipliers[i], 1, 64),
                    macro);
                if (child == null) continue;
                child.transform.SetParent(parent.transform, true);
                lods[i] = new LOD(thresholds[i], new[] { child.GetComponent<MeshRenderer>() });
            }

            var group = parent.AddComponent<LODGroup>();
            group.fadeMode = LODFadeMode.CrossFade;
            group.animateCrossFading = true;
            group.SetLODs(lods);
            group.RecalculateBounds();
            return parent;
        }

        private GameObject BuildMesh(string name, JObject terrain, int step, bool macro)
        {
            var width = terrain.Value<int?>("grid_width") ?? 0;
            var height = terrain.Value<int?>("grid_height") ?? 0;
            var worldWidth = terrain.Value<float?>("world_width_m") ?? 0f;
            var worldDepth = terrain.Value<float?>("world_depth_m") ?? 0f;
            var center = EverestCoordinates.Position(terrain["terrain_center"]);
            var heights = terrain["heights"] as JArray;
            if (width < 2 || height < 2 || heights == null || heights.Count < width * height) return null;

            var columns = SampleIndices(width, step);
            var rows = SampleIndices(height, step);
            var sampledWidth = columns.Count;
            var sampledHeight = rows.Count;
            var vertices = new Vector3[sampledWidth * sampledHeight];
            var triangles = new List<int>((sampledWidth - 1) * (sampledHeight - 1) * 6);
            var x0 = -worldWidth * 0.5f;
            var z0 = -worldDepth * 0.5f;

            for (var row = 0; row < sampledHeight; ++row)
            {
                var sourceRow = rows[row];
                var z = z0 + sourceRow * worldDepth / (height - 1);
                for (var col = 0; col < sampledWidth; ++col)
                {
                    var sourceColumn = columns[col];
                    var sourceIndex = sourceRow * width + sourceColumn;
                    var x = x0 + sourceColumn * worldWidth / (width - 1);
                    vertices[row * sampledWidth + col] = center + new Vector3(x, heights[sourceIndex].Value<float>() + 0.012f, z);
                }
            }

            Vector3 localCenter = Vector3.zero;
            var localHalfWidth = 0f;
            var localHalfDepth = 0f;
            if (macro && _localData != null)
            {
                localCenter = EverestCoordinates.Position(_localData["terrain_center"]);
                localHalfWidth = (_localData.Value<float?>("world_width_m") ?? 0f) * 0.505f;
                localHalfDepth = (_localData.Value<float?>("world_depth_m") ?? 0f) * 0.505f;
            }

            for (var row = 0; row < sampledHeight - 1; ++row)
            {
                for (var col = 0; col < sampledWidth - 1; ++col)
                {
                    var a = row * sampledWidth + col;
                    var b = a + 1;
                    var c = a + sampledWidth;
                    var d = c + 1;
                    if (macro && localHalfWidth > 0f)
                    {
                        var aInside = Mathf.Abs(vertices[a].x - localCenter.x) <= localHalfWidth
                            && Mathf.Abs(vertices[a].z - localCenter.z) <= localHalfDepth;
                        var bInside = Mathf.Abs(vertices[b].x - localCenter.x) <= localHalfWidth
                            && Mathf.Abs(vertices[b].z - localCenter.z) <= localHalfDepth;
                        var cInside = Mathf.Abs(vertices[c].x - localCenter.x) <= localHalfWidth
                            && Mathf.Abs(vertices[c].z - localCenter.z) <= localHalfDepth;
                        var dInside = Mathf.Abs(vertices[d].x - localCenter.x) <= localHalfWidth
                            && Mathf.Abs(vertices[d].z - localCenter.z) <= localHalfDepth;
                        // Keep partially overlapping macro quads. Removing by
                        // center created large sky-colored holes at coarse LOD.
                        if (aInside && bInside && cInside && dInside)
                            continue;
                    }
                    triangles.Add(a);
                    triangles.Add(c);
                    triangles.Add(b);
                    triangles.Add(b);
                    triangles.Add(c);
                    triangles.Add(d);
                }
            }

            var mesh = new Mesh { name = name };
            mesh.indexFormat = vertices.Length > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
            mesh.vertices = vertices;
            mesh.SetTriangles(triangles, 0, false);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            var go = new GameObject(name);
            go.transform.SetParent(transform, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = go.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = BuildMaterial(macro);
            renderer.shadowCastingMode = macro ? ShadowCastingMode.Off : ShadowCastingMode.On;
            renderer.receiveShadows = true;
            return go;
        }

        private Material BuildMaterial(bool macro)
        {
            var material = EverestRuntimeMaterials.Load("EverestTerrain", "Everest/Terrain");
            if (material == null) return null;
            material.name = macro ? "Everest Macro Material Shell" : "Everest Local Material Shell";
            if (material.shader != null && material.shader.name == "Everest/Terrain")
            {
                // Use one world-space material at every distance. Switching
                // photo sets at the local/macro boundary was visible as a
                // broad snow-color seam.
                var snow = Resources.Load<Texture2D>("Textures/Everest/snow_field_aerial_col_2k");
                var snowRoughness = Resources.Load<Texture2D>("Textures/Everest/snow_field_aerial_rough_2k");
                var rock = Resources.Load<Texture2D>("Textures/Everest/rocky_terrain_02_diff_2k");
                var rockRoughness = Resources.Load<Texture2D>("Textures/Everest/rocky_terrain_02_rough_2k");
                if (snow != null) material.SetTexture("_SnowTex", snow);
                if (snowRoughness != null) material.SetTexture("_SnowRoughness", snowRoughness);
                if (rock != null) material.SetTexture("_RockTex", rock);
                if (rockRoughness != null) material.SetTexture("_RockRoughness", rockRoughness);
                material.SetFloat("_SnowScale", 0.0075f);
                material.SetFloat("_RockScale", 0.055f);
                material.SetFloat("_MacroScale", 0.010f);
                material.SetFloat("_Stylize", 0.36f);
                material.SetFloat("_ActiveMaterial", 0f);
                material.SetFloat("_RockSlopeStart", 0.35f);
                material.SetFloat("_RockSlopeEnd", 0.72f);
                material.SetFloat("_IceSlopeStart", 0.18f);
                material.SetFloat("_IceSlopeEnd", 0.74f);
            }
            _materials.Add(material);
            ApplyMaterialState(material);
            return material;
        }

        private void UpdateMaterials()
        {
            for (var i = _materials.Count - 1; i >= 0; --i)
            {
                var material = _materials[i];
                if (material == null)
                {
                    _materials.RemoveAt(i);
                    continue;
                }
                ApplyMaterialState(material);
            }
        }

        private void ApplyMaterialState(Material material)
        {
            if (material.HasProperty("_SnowDepth")) material.SetFloat("_SnowDepth", _snowDepth);
            if (material.HasProperty("_SurfaceMode")) material.SetFloat("_SurfaceMode", _surfaceMode);
            if (material.HasProperty("_SnowfallIntensity")) material.SetFloat("_SnowfallIntensity", Mathf.Clamp01(_snowfall / 40f));
            if (material.HasProperty("_TemperatureC")) material.SetFloat("_TemperatureC", _temperature);
        }

        private static List<int> SampleIndices(int count, int step)
        {
            var result = new List<int>();
            for (var i = 0; i < count; i += Mathf.Max(1, step)) result.Add(i);
            if (result[result.Count - 1] != count - 1) result.Add(count - 1);
            return result;
        }

        private void OnDestroy()
        {
            foreach (var material in _materials)
                if (material != null) Destroy(material);
        }
    }
}
