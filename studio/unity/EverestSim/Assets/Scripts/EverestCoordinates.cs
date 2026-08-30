using Newtonsoft.Json.Linq;
using UnityEngine;

namespace EverestSim
{
    /// <summary>
    /// Converts the renderer-neutral backend frame (RH, Z-up, wxyz) into
    /// Unity's left-handed Y-up frame. Backend data is never mutated.
    /// </summary>
    public static class EverestCoordinates
    {
        public static Vector3 Position(JToken value)
        {
            if (value == null || !value.HasValues) return Vector3.zero;
            return new Vector3(
                value[0]!.Value<float>(),
                value[2]!.Value<float>(),
                value[1]!.Value<float>());
        }

        public static Vector3 Direction(JToken value)
        {
            return Position(value);
        }

        public static Vector3 Scale(JToken value)
        {
            if (value == null || !value.HasValues) return Vector3.one;
            return new Vector3(
                value[0]!.Value<float>(),
                value[2]!.Value<float>(),
                value[1]!.Value<float>());
        }

        public static Quaternion RotationWxyz(JToken value)
        {
            if (value == null || !value.HasValues) return Quaternion.identity;

            var w = value[0]!.Value<float>();
            var x = value[1]!.Value<float>();
            var y = value[2]!.Value<float>();
            var z = value[3]!.Value<float>();

            // For the reflection M:(x,y,z)->(x,z,y), an axial vector transforms
            // with det(M)M. Conjugating the backend rotation by M therefore gives
            // Unity quaternion (x,y,z,w)=(-x,-z,-y,w).
            var q = new Quaternion(-x, -z, -y, w);
            var magnitude = Mathf.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            if (magnitude <= 1e-8f) return Quaternion.identity;
            q.x /= magnitude;
            q.y /= magnitude;
            q.z /= magnitude;
            q.w /= magnitude;
            return q;
        }
    }
}
