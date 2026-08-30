using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EverestSim
{
    public sealed class EverestRobotRenderer : MonoBehaviour
    {
        private sealed class BodyTarget
        {
            public Transform Transform;
            public Vector3 Position;
            public Quaternion Rotation;
            public bool HasPose;
        }

        private readonly Dictionary<string, BodyTarget> _bodies = new Dictionary<string, BodyTarget>();
        private Transform _root;
        private long _sequence = -1;
        private long _sourceEpoch = -1;

        public float InterpolationSharpness = 28f;
        public JObject LatestFeet { get; private set; }
        public double SimTime { get; private set; }
        public long Sequence => _sequence;

        public void SetSourceEpoch(long epoch)
        {
            if (epoch == _sourceEpoch) return;
            _sourceEpoch = epoch;
            _sequence = -1;
            // The next source sample must snap. Interpolating from a simulated
            // pose into unrelated physical/replay telemetry is misleading.
            foreach (var body in _bodies.Values)
            {
                body.HasPose = false;
                body.Transform.gameObject.SetActive(false);
            }
        }

        private void Awake()
        {
            _root = new GameObject("G1 Authoritative Bodies").transform;
            _root.SetParent(transform, false);
        }

        public void OnScene(JObject scene)
        {
            foreach (Transform child in _root) Destroy(child.gameObject);
            _bodies.Clear();

            var names = scene["body_names"] as JArray;
            if (names != null)
            {
                foreach (var token in names)
                {
                    var name = token.Value<string>();
                    if (string.IsNullOrWhiteSpace(name)) continue;
                    CreateBody(name);
                }
            }

            var visuals = scene["visuals"] as JArray;
            if (visuals == null) return;
            foreach (var token in visuals)
            {
                if (!(token is JObject visual)) continue;
                var bodyName = visual.Value<string>("body");
                if (string.IsNullOrWhiteSpace(bodyName)) continue;
                if (!_bodies.TryGetValue(bodyName, out var body)) body = CreateBody(bodyName);
                AttachVisual(body.Transform, visual);
            }
        }

        private BodyTarget CreateBody(string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(_root, false);
            var body = new BodyTarget
            {
                Transform = go.transform,
                Position = Vector3.zero,
                Rotation = Quaternion.identity,
                HasPose = false
            };
            _bodies[name] = body;
            return body;
        }

        private static void AttachVisual(Transform body, JObject visual)
        {
            var asset = visual.Value<string>("asset") ?? visual.Value<string>("mesh") ?? string.Empty;
            var stem = Path.GetFileNameWithoutExtension(asset);
            var prefab = Resources.Load<GameObject>($"G1/{stem}");
            GameObject instance;
            if (prefab != null)
            {
                instance = Instantiate(prefab, body, false);
                instance.name = stem;
            }
            else
            {
                // Keep protocol debugging usable before the one-time G1 asset
                // conversion has been run.
                instance = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                instance.name = $"{stem} (missing asset)";
                instance.transform.SetParent(body, false);
                instance.transform.localScale = Vector3.one * 0.045f;
                var collider = instance.GetComponent<Collider>();
                if (collider != null) Destroy(collider);
            }

            instance.transform.localPosition = EverestCoordinates.Position(visual["position"]);
            instance.transform.localRotation = EverestCoordinates.RotationWxyz(visual["quaternion"]);
            if (visual["scale"] != null)
                instance.transform.localScale = Vector3.Scale(instance.transform.localScale, EverestCoordinates.Scale(visual["scale"]));
        }

        public void OnFrame(JObject frame)
        {
            SetSourceEpoch(frame.Value<long?>("source_epoch") ?? _sourceEpoch);
            var sequence = frame.Value<long?>("sequence") ?? -1;
            if (sequence <= _sequence) return;
            _sequence = sequence;
            SimTime = frame.Value<double?>("sim_time") ?? SimTime;
            LatestFeet = frame["feet"] as JObject;

            var names = frame["body_names"] as JArray;
            var positions = frame["body_pos_w"] as JArray;
            var rotations = frame["body_quat_w"] as JArray;
            if (names == null || positions == null || rotations == null) return;

            var count = Math.Min(names.Count, Math.Min(positions.Count, rotations.Count));
            for (var i = 0; i < count; ++i)
            {
                var name = names[i].Value<string>();
                if (string.IsNullOrWhiteSpace(name)) continue;
                if (!_bodies.TryGetValue(name, out var body)) body = CreateBody(name);
                body.Transform.gameObject.SetActive(true);
                body.Position = EverestCoordinates.Position(positions[i]);
                body.Rotation = EverestCoordinates.RotationWxyz(rotations[i]);
                if (!body.HasPose)
                {
                    body.Transform.position = body.Position;
                    body.Transform.rotation = body.Rotation;
                    body.HasPose = true;
                }
            }
        }

        private void LateUpdate()
        {
            var blend = 1f - Mathf.Exp(-InterpolationSharpness * Time.unscaledDeltaTime);
            foreach (var body in _bodies.Values)
            {
                if (!body.HasPose) continue;
                body.Transform.position = Vector3.Lerp(body.Transform.position, body.Position, blend);
                body.Transform.rotation = Quaternion.Slerp(body.Transform.rotation, body.Rotation, blend);
            }
        }

        public bool TryGetBodyPosition(string name, out Vector3 position)
        {
            if (_bodies.TryGetValue(name, out var body) && body.HasPose)
            {
                position = body.Transform.position;
                return true;
            }
            position = Vector3.zero;
            return false;
        }
    }
}
