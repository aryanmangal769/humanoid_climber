using UnityEngine;

namespace EverestSim
{
    /// <summary>
    /// Loads renderer materials from Resources so their shaders are concrete
    /// build dependencies. Runtime-only Shader.Find calls are stripped from
    /// WebGL builds and produce Unity's magenta error material.
    /// </summary>
    public static class EverestRuntimeMaterials
    {
        public static Material Load(string resourceName, string fallbackShaderName)
        {
            var template = Resources.Load<Material>($"Materials/{resourceName}");
            if (template != null)
                return new Material(template) { name = $"{resourceName} (Runtime)" };

            var shader = Shader.Find(fallbackShaderName);
            if (shader == null)
            {
                Debug.LogError(
                    $"Everest material '{resourceName}' and shader '{fallbackShaderName}' are missing. " +
                    "Run the Everest WebGL build so runtime materials are generated and included.");
                return null;
            }

            Debug.LogWarning(
                $"Everest runtime material '{resourceName}' was not found; using editor shader fallback '{fallbackShaderName}'.");
            return new Material(shader) { name = $"{resourceName} (Fallback)" };
        }
    }
}
