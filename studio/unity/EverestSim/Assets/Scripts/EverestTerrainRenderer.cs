using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace EverestSim
{
    /// <summary>
    /// Renders the authoritative DEM as a lightweight wireframe context layer.
    /// Physics never lives here: MuJoCo owns the full DEM and Newton owns the
    /// active snow/ice material window rendered by EverestSnowRenderer.
    /// </summary>
    public sealed class EverestTerrainRenderer : MonoBehaviour
    {
        private JObject _localData;
        private JObject _macroData;
        private GameObject _local;
        private GameObject _macro;
        private int _localLodStep = 1;
        private int _macroLodStep = 7;

        public int LocalLodStep => _localLodStep;
        public int MacroLodStep => _macroLodStep;

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
            _local = BuildWireTerrainLod(
                "Everest DEM · Local Wireframe",
                _localData,
                _localLodStep,
                new Color(0.24f, 0.60f, 0.76f, 0.28f));
        }

        private void RebuildMacro()
        {
            if (_macroData == null) return;
            if (_macro != null) Destroy(_macro);
            _macro = BuildWireTerrainLod(
                "Everest DEM · Macro Wireframe",
                _macroData,
                _macroLodStep,
                new Color(0.18f, 0.34f, 0.42f, 0.16f));
        }

        private GameObject BuildWireTerrainLod(string name, JObject terrain, int baseStep, Color color)
        {
            var parent = new GameObject(name);
            parent.transform.SetParent(transform, false);
            var multipliers = new[] { 1, 2, 4 };
            var thresholds = new[] { 0.32f, 0.10f, 0.02f };
            var lods = new LOD[multipliers.Length];

            for (var i = 0; i < multipliers.Length; ++i)
            {
                var child = BuildWireTerrain(
                    $"{name} · LOD{i}",
                    terrain,
                    Mathf.Clamp(baseStep * multipliers[i], 1, 64),
                    new Color(color.r, color.g, color.b, color.a * (1f - i * 0.12f)));
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

        private GameObject BuildWireTerrain(string name, JObject terrain, int step, Color color)
        {
            var width = terrain.Value<int?>("grid_width") ?? 0;
            var height = terrain.Value<int?>("grid_height") ?? 0;
            var worldWidth = terrain.Value<float?>("world_width_m") ?? 0f;
            var worldDepth = terrain.Value<float?>("world_depth_m") ?? 0f;
            var center = EverestCoordinates.Position(terrain["terrain_center"]);
            var heights = terrain["heights"] as JArray;
            if (width < 2 || height < 2 || heights == null || heights.Count < width * height)
            {
                Debug.LogWarning($"Invalid {name} payload");
                return null;
            }

            var columns = SampleIndices(width, step);
            var rows = SampleIndices(height, step);
            var sampledWidth = columns.Count;
            var sampledHeight = rows.Count;
            var vertices = new Vector3[sampledWidth * sampledHeight];
            var lineIndices = new List<int>(sampledWidth * sampledHeight * 4);
            var x0 = -worldWidth * 0.5f;
            var z0 = -worldDepth * 0.5f;

            for (var r = 0; r < sampledHeight; ++r)
            {
                var sourceRow = rows[r];
                var z = z0 + sourceRow * worldDepth / (height - 1);
                for (var c = 0; c < sampledWidth; ++c)
                {
                    var sourceColumn = columns[c];
                    var sourceIndex = sourceRow * width + sourceColumn;
                    var x = x0 + sourceColumn * worldWidth / (width - 1);
                    vertices[r * sampledWidth + c] = center + new Vector3(
                        x,
                        heights[sourceIndex].Value<float>() + 0.024f,
                        z);
                }
            }

            for (var r = 0; r < sampledHeight; ++r)
            {
                for (var c = 0; c < sampledWidth; ++c)
                {
                    var i = r * sampledWidth + c;
                    if (c + 1 < sampledWidth)
                    {
                        lineIndices.Add(i);
                        lineIndices.Add(i + 1);
                    }
                    if (r + 1 < sampledHeight)
                    {
                        lineIndices.Add(i);
                        lineIndices.Add(i + sampledWidth);
                    }
                }
            }

            var mesh = new Mesh { name = name };
            mesh.indexFormat = vertices.Length > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16;
            mesh.vertices = vertices;
            mesh.SetIndices(lineIndices, MeshTopology.Lines, 0, false);
            mesh.RecalculateBounds();

            var go = new GameObject(name);
            go.transform.SetParent(transform, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = go.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = BuildWireMaterial(color);
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            renderer.receiveShadows = false;
            return go;
        }

        private static List<int> SampleIndices(int count, int step)
        {
            var result = new List<int>();
            for (var i = 0; i < count; i += Mathf.Max(1, step)) result.Add(i);
            if (result[result.Count - 1] != count - 1) result.Add(count - 1);
            return result;
        }

        private static Material BuildWireMaterial(Color color)
        {
            var material = EverestRuntimeMaterials.Load("EverestWireTerrain", "Everest/WireTerrain");
            if (material == null) return null;
            material.name = "Everest DEM Wireframe";
            if (material.HasProperty("_Color")) material.SetColor("_Color", color);
            material.color = color;
            return material;
        }
    }
}
