using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace EverestSim.Editor
{
    public static class EverestWebBuild
    {
        private const string ScenePath = "Assets/Scenes/Everest.unity";

        [MenuItem("Everest/Build WebGL")]
        public static void BuildWebGL()
        {
            EnsureScene();
            EnsureRuntimeMaterials();

            PlayerSettings.companyName = "Everest Dream";
            PlayerSettings.productName = "Everest Dream";
            PlayerSettings.WebGL.compressionFormat = WebGLCompressionFormat.Disabled;
            PlayerSettings.WebGL.dataCaching = true;
            PlayerSettings.WebGL.initialMemorySize = 256;
            PlayerSettings.WebGL.memoryGrowthMode = WebGLMemoryGrowthMode.Geometric;
            PlayerSettings.WebGL.maximumMemorySize = 1024;
            PlayerSettings.WebGL.geometricMemoryGrowthStep = 0.20f;
            PlayerSettings.WebGL.memoryGeometricGrowthCap = 64;
            PlayerSettings.WebGL.powerPreference = WebGLPowerPreference.HighPerformance;
            PlayerSettings.WebGL.showDiagnostics = true;
            PlayerSettings.WebGL.template = "PROJECT:EverestFull";
            PlayerSettings.runInBackground = true;

            var output = Environment.GetEnvironmentVariable("EVEREST_WEBGL_OUT");
            if (string.IsNullOrWhiteSpace(output))
                output = Path.Combine(Directory.GetCurrentDirectory(), "Builds", "WebGL");
            Directory.CreateDirectory(output);

            var options = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = output,
                target = BuildTarget.WebGL,
                options = BuildOptions.None
            };

            var report = BuildPipeline.BuildPlayer(options);
            var summary = report.summary;
            Debug.Log($"Everest WebGL build: {summary.result}; {summary.totalSize / (1024f * 1024f):0.0} MiB -> {output}");
            if (summary.result != BuildResult.Succeeded)
                throw new InvalidOperationException($"WebGL build failed: {summary.result}");
        }

        private static void EnsureRuntimeMaterials()
        {
            const string folder = "Assets/Resources/Materials";
            Directory.CreateDirectory(Path.Combine(Directory.GetCurrentDirectory(), folder));

            EnsureRuntimeMaterial(folder, "EverestSnow", "Everest/Snow");
            EnsureRuntimeMaterial(folder, "EverestTerrain", "Everest/Terrain");
            EnsureRuntimeMaterial(folder, "EverestWireTerrain", "Everest/WireTerrain");
            EnsureRuntimeMaterial(folder, "EverestVolumetricClouds", "Everest/VolumetricClouds");
            EnsureRuntimeMaterial(folder, "EverestSnowfall", "Everest/Snowfall");
            EnsureRuntimeMaterial(folder, "EverestSnowLayers", "Everest/SnowLayers");

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
        }

        private static void EnsureRuntimeMaterial(string folder, string assetName, string shaderName)
        {
            var shader = Shader.Find(shaderName);
            if (shader == null || !shader.isSupported)
                throw new InvalidOperationException($"Required WebGL shader is missing or unsupported: {shaderName}");

            var path = $"{folder}/{assetName}.mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                material = new Material(shader) { name = assetName };
                AssetDatabase.CreateAsset(material, path);
            }
            else if (material.shader != shader)
            {
                material.shader = shader;
                EditorUtility.SetDirty(material);
            }
        }

        private static void EnsureScene()
        {
            var absolute = Path.Combine(Directory.GetCurrentDirectory(), ScenePath);
            Directory.CreateDirectory(Path.GetDirectoryName(absolute) ?? "Assets/Scenes");
            if (!File.Exists(absolute))
            {
                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                EditorSceneManager.SaveScene(scene, ScenePath);
            }
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            AssetDatabase.SaveAssets();
        }
    }
}
