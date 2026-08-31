using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EverestSim
{
    /// <summary>
    /// Presentation-only cues for the autonomous showcase. Route, ice-region,
    /// and force vectors all consume backend-authored coordinates/state; this
    /// component never moves the robot or deforms terrain.
    /// </summary>
    public sealed class EverestDemoRenderer : MonoBehaviour
    {
        private GameObject _root;
        private LineRenderer _route;
        private LineRenderer _frictionRegion;
        private readonly List<LineRenderer> _forces = new List<LineRenderer>();
        private Material _lineTemplate;

        public void OnState(JObject state)
        {
            var demo = state?["demo"] as JObject;
            var active = demo?.Value<bool?>("active") == true;
            EnsureRoot();
            _root.SetActive(active);
            if (!active) return;

            DrawRoute(demo?["route_points"] as JArray);
            DrawFrictionRegion(demo?["low_friction_region"] as JObject);
            DrawForces(demo?["force_vectors"] as JArray);
        }

        private void EnsureRoot()
        {
            if (_root != null) return;
            _root = new GameObject("Autonomous Demo Cues");
            _root.transform.SetParent(transform, false);
            _lineTemplate = EverestRuntimeMaterials.Load("EverestWireTerrain", "Everest/WireTerrain");
        }

        private LineRenderer CreateLine(string name, Color color, float width)
        {
            var go = new GameObject(name);
            go.transform.SetParent(_root.transform, false);
            var line = go.AddComponent<LineRenderer>();
            line.useWorldSpace = true;
            line.loop = false;
            line.widthMultiplier = width;
            line.numCapVertices = 3;
            line.numCornerVertices = 3;
            line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            line.receiveShadows = false;
            if (_lineTemplate != null)
            {
                line.material = new Material(_lineTemplate) { name = $"{name} Material" };
                if (line.material.HasProperty("_Color")) line.material.SetColor("_Color", color);
                line.material.color = color;
            }
            line.startColor = color;
            line.endColor = color;
            return line;
        }

        private void DrawRoute(JArray points)
        {
            if (_route == null)
                _route = CreateLine("Planned ascent route", new Color(0.20f, 0.86f, 1.0f, 0.82f), 0.035f);
            if (points == null || points.Count < 2)
            {
                _route.positionCount = 0;
                return;
            }
            _route.positionCount = points.Count;
            for (var index = 0; index < points.Count; ++index)
                _route.SetPosition(index, EverestCoordinates.Position(points[index]));
        }

        private void DrawFrictionRegion(JObject region)
        {
            if (_frictionRegion == null)
                _frictionRegion = CreateLine("Low-friction snow over ice", new Color(0.24f, 0.58f, 1.0f, 0.70f), 0.055f);
            var centerToken = region?["center"];
            var size = region?["size"] as JArray;
            if (centerToken == null || size == null || size.Count < 2)
            {
                _frictionRegion.positionCount = 0;
                return;
            }
            var center = EverestCoordinates.Position(centerToken);
            var halfX = size[0].Value<float>() * 0.5f;
            var halfZ = size[1].Value<float>() * 0.5f;
            var rotation = Quaternion.Euler(0f, -(region.Value<float?>("yaw_deg") ?? 0f), 0f);
            var lift = Vector3.up * 0.055f;
            var corners = new[]
            {
                center + rotation * new Vector3(-halfX, 0f, -halfZ) + lift,
                center + rotation * new Vector3( halfX, 0f, -halfZ) + lift,
                center + rotation * new Vector3( halfX, 0f,  halfZ) + lift,
                center + rotation * new Vector3(-halfX, 0f,  halfZ) + lift,
                center + rotation * new Vector3(-halfX, 0f, -halfZ) + lift,
            };
            _frictionRegion.positionCount = corners.Length;
            _frictionRegion.SetPositions(corners);
            var physicallyActive = region.Value<bool?>("physical_friction_active") == true;
            var color = physicallyActive
                ? new Color(0.34f, 0.76f, 1.0f, 0.92f)
                : new Color(0.24f, 0.58f, 1.0f, 0.46f);
            _frictionRegion.startColor = color;
            _frictionRegion.endColor = color;
            if (_frictionRegion.material != null)
            {
                if (_frictionRegion.material.HasProperty("_Color")) _frictionRegion.material.SetColor("_Color", color);
                _frictionRegion.material.color = color;
            }
        }

        private void DrawForces(JArray vectors)
        {
            var count = vectors?.Count ?? 0;
            while (_forces.Count < count)
                _forces.Add(CreateLine("Applied force", new Color(1.0f, 0.68f, 0.30f, 0.68f), 0.022f));
            for (var index = 0; index < _forces.Count; ++index)
            {
                var line = _forces[index];
                if (index >= count)
                {
                    line.positionCount = 0;
                    continue;
                }
                var item = vectors[index] as JObject;
                var origin = EverestCoordinates.Position(item?["origin"]);
                var force = EverestCoordinates.Direction(item?["force_n"]);
                var wind = item?.Value<string>("kind") == "wind";
                // Keep the backend vector physically exact (16 N in the
                // showcase) while giving wind enough presentation scale to
                // remain legible against the Everest terrain.
                var visualScale = wind ? 0.012f : 0.006f;
                var length = Mathf.Clamp(force.magnitude * visualScale, 0.18f, 1.45f);
                var endpoint = origin + (force.sqrMagnitude > 1e-8f ? force.normalized * length : Vector3.zero);
                var direction = force.sqrMagnitude > 1e-8f ? force.normalized : Vector3.forward;
                var side = Vector3.Cross(direction, Vector3.up);
                if (side.sqrMagnitude < 1e-6f) side = Vector3.right;
                side.Normalize();
                var wingBack = direction * Mathf.Min(0.22f, length * 0.28f);
                var wingSide = side * Mathf.Min(0.13f, length * 0.17f);
                line.widthMultiplier = wind ? 0.040f : 0.022f;
                line.positionCount = 5;
                line.SetPosition(0, origin);
                line.SetPosition(1, endpoint);
                line.SetPosition(2, endpoint - wingBack + wingSide);
                line.SetPosition(3, endpoint);
                line.SetPosition(4, endpoint - wingBack - wingSide);
                var color = wind
                    ? new Color(0.15f, 0.92f, 1.0f, 0.96f)
                    : new Color(1.0f, 0.68f, 0.30f, 0.54f);
                line.startColor = color;
                line.endColor = new Color(color.r, color.g, color.b, 0.18f);
            }
        }

        private void OnDestroy()
        {
            if (_lineTemplate != null) Destroy(_lineTemplate);
        }
    }
}
