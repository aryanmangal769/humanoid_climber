#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace EverestSim.Editor
{
    public sealed class EverestTextureImporter : AssetPostprocessor
    {
        private void OnPreprocessTexture()
        {
            if (!assetPath.Contains("/Resources/Textures/Everest/")) return;

            var importer = (TextureImporter)assetImporter;
            importer.wrapMode = TextureWrapMode.Repeat;
            importer.filterMode = FilterMode.Trilinear;
            importer.mipmapEnabled = true;
            importer.anisoLevel = 6;
            importer.maxTextureSize = 2048;
            importer.textureCompression = TextureImporterCompression.CompressedHQ;

            var webgl = importer.GetPlatformTextureSettings("WebGL");
            webgl.overridden = true;
            webgl.maxTextureSize = 1024;
            webgl.format = TextureImporterFormat.Automatic;
            webgl.textureCompression = TextureImporterCompression.Compressed;
            importer.SetPlatformTextureSettings(webgl);

            if (assetPath.Contains("_nor_gl_"))
            {
                importer.textureType = TextureImporterType.NormalMap;
                importer.sRGBTexture = false;
            }
            else if (assetPath.Contains("_rough_"))
            {
                importer.textureType = TextureImporterType.Default;
                importer.sRGBTexture = false;
            }
            else
            {
                importer.textureType = TextureImporterType.Default;
                importer.sRGBTexture = true;
            }
        }
    }
}
#endif
