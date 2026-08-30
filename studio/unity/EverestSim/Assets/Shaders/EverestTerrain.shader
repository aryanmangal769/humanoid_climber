Shader "Everest/Terrain"
{
    Properties
    {
        _SnowTex ("Snow Albedo", 2D) = "white" {}
        _SnowRoughness ("Snow Roughness", 2D) = "white" {}
        _RockTex ("Rock Albedo", 2D) = "gray" {}
        _RockRoughness ("Rock Roughness", 2D) = "white" {}

        _SnowLight ("Snow Light", Color) = (0.76, 0.87, 0.96, 1)
        _SnowShadow ("Snow Shadow", Color) = (0.16, 0.30, 0.46, 1)
        _IceLight ("Ice Light", Color) = (0.34, 0.67, 0.82, 1)
        _IceDeep ("Ice Deep", Color) = (0.055, 0.18, 0.28, 1)
        _RockLight ("Rock Light", Color) = (0.43, 0.43, 0.42, 1)
        _RockShadow ("Rock Shadow", Color) = (0.12, 0.14, 0.16, 1)

        _SnowScale ("Snow World Scale", Float) = 0.0075
        _RockScale ("Rock World Scale", Float) = 0.055
        _MacroScale ("Macro Variation Scale", Float) = 0.01
        _RockSlopeStart ("Rock Slope Start", Range(0,1)) = 0.54
        _RockSlopeEnd ("Rock Slope End", Range(0,1)) = 0.90
        _IceSlopeStart ("Ice Slope Start", Range(0,1)) = 0.28
        _IceSlopeEnd ("Ice Slope End", Range(0,1)) = 0.76
        _RimStrength ("Ice/Snow Rim", Range(0,1)) = 0.18
        _Stylize ("Stylize", Range(0,1)) = 0.38

        _SnowDepth ("Snow Depth", Float) = 0.4
        _SurfaceMode ("Surface Mode", Range(0,2)) = 0
        _SnowfallIntensity ("Snowfall Intensity", Range(0,1)) = 0
        _TemperatureC ("Temperature C", Float) = -18
        _ActiveMaterial ("Backend Active Material", Range(0,1)) = 0
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 300

        CGPROGRAM
        #pragma surface surf EverestAlpine fullforwardshadows addshadow
        #pragma target 3.0

        sampler2D _SnowTex;
        sampler2D _SnowRoughness;
        sampler2D _RockTex;
        sampler2D _RockRoughness;

        fixed4 _SnowLight;
        fixed4 _SnowShadow;
        fixed4 _IceLight;
        fixed4 _IceDeep;
        fixed4 _RockLight;
        fixed4 _RockShadow;

        float _SnowScale;
        float _RockScale;
        float _MacroScale;
        float _RockSlopeStart;
        float _RockSlopeEnd;
        float _IceSlopeStart;
        float _IceSlopeEnd;
        half _RimStrength;
        half _Stylize;
        float _SnowDepth;
        half _SurfaceMode;
        half _SnowfallIntensity;
        float _TemperatureC;
        half _ActiveMaterial;

        float4 _EverestActiveCenter;
        float _EverestActiveRadius;
        float4 _EverestHistoryCenters[16];
        int _EverestHistoryCount;
        float4 _EverestWindDir;
        float _EverestWindStrength;

        struct Input
        {
            float3 worldPos;
            float3 worldNormal;
            float3 viewDir;
            float4 color : COLOR;
        };

        inline float3 TriWeights(float3 normal)
        {
            float3 w = pow(abs(normal), 7.0);
            return w / max(w.x + w.y + w.z, 1e-4);
        }

        inline fixed4 SampleTriplanar(sampler2D tex, float3 p, float3 n, float scale)
        {
            float3 w = TriWeights(n);
            fixed4 x = tex2D(tex, p.zy * scale);
            fixed4 y = tex2D(tex, p.xz * scale);
            fixed4 z = tex2D(tex, p.xy * scale);
            return x * w.x + y * w.y + z * w.z;
        }

        inline float2 RotateUv(float2 uv, float angle)
        {
            float s = sin(angle);
            float c = cos(angle);
            return float2(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
        }

        inline float ValueNoise(float2 p);

        // Blend three independently rotated/offset projections. A single
        // world-space photo repeats eventually; this inexpensive stochastic
        // blend keeps its broad aerial character while hiding tile edges and
        // making repetition much less legible at both LODs.
        inline fixed4 SampleSnowStochastic(sampler2D tex, float3 p, float3 n, float scale)
        {
            float3 w = TriWeights(n);
            float2 uvX = p.zy;
            float2 uvY = p.xz;
            float2 uvZ = p.xy;
            float selector = ValueNoise(p.xz * 0.0017 + 17.0);
            float blend1 = smoothstep(0.22, 0.55, selector);
            float blend2 = smoothstep(0.55, 0.88, selector);
            float weight0 = 1.0 - blend1;
            float weight1 = blend1 * (1.0 - blend2);
            float weight2 = blend2;

            float2 x0 = RotateUv(uvX * scale, 0.17) + float2(0.13, 0.71);
            float2 y0 = RotateUv(uvY * scale, 0.17) + float2(0.13, 0.71);
            float2 z0 = RotateUv(uvZ * scale, 0.17) + float2(0.13, 0.71);
            float2 x1 = RotateUv(uvX * (scale * 0.79), -0.31) + float2(0.47, 0.19);
            float2 y1 = RotateUv(uvY * (scale * 0.79), -0.31) + float2(0.47, 0.19);
            float2 z1 = RotateUv(uvZ * (scale * 0.79), -0.31) + float2(0.47, 0.19);
            float2 x2 = RotateUv(uvX * (scale * 1.37), 0.53) + float2(0.83, 0.37);
            float2 y2 = RotateUv(uvY * (scale * 1.37), 0.53) + float2(0.83, 0.37);
            float2 z2 = RotateUv(uvZ * (scale * 1.37), 0.53) + float2(0.83, 0.37);

            fixed4 p0 = tex2D(tex, x0) * w.x + tex2D(tex, y0) * w.y + tex2D(tex, z0) * w.z;
            fixed4 p1 = tex2D(tex, x1) * w.x + tex2D(tex, y1) * w.y + tex2D(tex, z1) * w.z;
            fixed4 p2 = tex2D(tex, x2) * w.x + tex2D(tex, y2) * w.y + tex2D(tex, z2) * w.z;
            return p0 * weight0 + p1 * weight1 + p2 * weight2;
        }

        inline float Hash21(float2 p)
        {
            p = frac(p * float2(123.34, 456.21));
            p += dot(p, p + 45.32);
            return frac(p.x * p.y);
        }

        inline float ValueNoise(float2 p)
        {
            float2 i = floor(p);
            float2 f = frac(p);
            f = f * f * (3.0 - 2.0 * f);
            float a = Hash21(i);
            float b = Hash21(i + float2(1,0));
            float c = Hash21(i + float2(0,1));
            float d = Hash21(i + 1.0);
            return lerp(lerp(a,b,f.x), lerp(c,d,f.x), f.y);
        }

        inline half4 LightingEverestAlpine(SurfaceOutput s, half3 lightDir, half3 viewDir, half atten)
        {
            half ndl = saturate(dot(s.Normal, lightDir));
            half wrap = saturate((dot(s.Normal, lightDir) + 0.25h) / 1.25h);
            half graphic = smoothstep(0.14h, 0.36h, ndl * atten) * 0.28h
                         + smoothstep(0.52h, 0.80h, ndl * atten) * 0.34h;
            half smoothRamp = 0.30h + 0.70h * wrap;
            half ramp = lerp(smoothRamp, 0.31h + graphic + 0.30h * ndl, _Stylize);

            half3 h = normalize(lightDir + viewDir);
            half nh = saturate(dot(s.Normal, h));
            half specPower = lerp(18.0h, 120.0h, s.Gloss);
            half specRaw = pow(nh, specPower);
            half spec = lerp(specRaw, smoothstep(0.50h, 0.80h, specRaw), _Stylize)
                      * s.Gloss * atten;

            // Keep snow below the HDR clamp so terrain relief and wind bands
            // remain readable in WebGL's LDR forward path.
            half3 ambient = UNITY_LIGHTMODEL_AMBIENT.rgb * s.Albedo * 0.40h;
            half3 direct = s.Albedo * _LightColor0.rgb * ramp * atten * 0.82h;
            return half4(ambient + direct + spec * _LightColor0.rgb + s.Emission, s.Alpha);
        }

        void surf(Input IN, inout SurfaceOutput o)
        {
            // The visual DEM shell yields to the backend-authored Newton/rigid
            // patch inside the active radius. No extra plane is rendered.
            if (_ActiveMaterial < 0.5h && _EverestActiveRadius > 0.01 && _SurfaceMode < 1.5h)
            {
                float2 delta = IN.worldPos.xz - _EverestActiveCenter.xz;
                float keep = length(delta) - _EverestActiveRadius * 0.94;
                [unroll]
                for (int historyIndex = 0; historyIndex < 16; ++historyIndex)
                {
                    if (historyIndex < _EverestHistoryCount)
                    {
                        float4 history = _EverestHistoryCenters[historyIndex];
                        float2 historyDelta = IN.worldPos.xz - history.xy;
                        keep = min(keep, length(historyDelta) - history.z * 0.94);
                    }
                }
                clip(keep);
            }

            float3 n = normalize(IN.worldNormal);
            float up = saturate(n.y);
            float slope = 1.0 - up;
            float macroA = ValueNoise(IN.worldPos.xz * _MacroScale);
            float macroB = ValueNoise(IN.worldPos.xz * (_MacroScale * 3.7) + 31.4);
            float macro = saturate(macroA * 0.68 + macroB * 0.32);

            fixed3 snowPhoto = SampleSnowStochastic(_SnowTex, IN.worldPos, n, _SnowScale).rgb;
            half snowRough = SampleTriplanar(_SnowRoughness, IN.worldPos, n, _SnowScale).r;
            fixed3 rockPhoto = SampleTriplanar(_RockTex, IN.worldPos, n, _RockScale).rgb;
            half rockRough = SampleTriplanar(_RockRoughness, IN.worldPos, n, _RockScale).r;

            float depthCoverage = saturate(_SnowDepth / 0.30);
            float windFacing = abs(dot(normalize(n.xz + float2(1e-4, 1e-4)), normalize(_EverestWindDir.xz + float2(1e-4, 1e-4))));
            float scour = saturate(_EverestWindStrength * windFacing * slope * 1.4);
            float snowRetention = saturate(depthCoverage * smoothstep(0.10, 0.70, up) - scour * 0.52);
            snowRetention = saturate(snowRetention + _SnowfallIntensity * 0.18);

            float snowShade = saturate(0.38 + up * 0.50 + macro * 0.17);
            fixed3 snow = lerp(_SnowShadow.rgb, _SnowLight.rgb, snowShade);
            snow *= lerp(0.78.xxx, snowPhoto, 0.66);
            snow *= lerp(0.88, 1.08, macro);

            float iceIn = smoothstep(_IceSlopeStart, _IceSlopeStart + 0.18, slope);
            float iceOut = 1.0 - smoothstep(_IceSlopeEnd - 0.12, _IceSlopeEnd, slope);
            float coldIce = saturate((-_TemperatureC - 2.0) / 24.0);
            float iceMask = saturate(iceIn * iceOut * (0.62 + macro * 0.48) * (1.0 - snowRetention * 0.72));
            iceMask = saturate(iceMask + scour * 0.42 + (1.0 - depthCoverage) * coldIce * 0.24);

            float rockSlope = smoothstep(_RockSlopeStart, _RockSlopeEnd, slope);
            float exposed = saturate((1.0 - snowRetention) * (0.34 + slope * 0.96));
            float rockMask = saturate(rockSlope * (0.70 + (1.0 - macro) * 0.32) + exposed * 0.38);
            half compact = _ActiveMaterial > 0.5h ? saturate(IN.color.a) : 0.0h;
            fixed3 layerTint = saturate(IN.color.rgb);
            half layeredIce = _ActiveMaterial > 0.5h
                ? smoothstep(0.16h, 0.38h, layerTint.b - layerTint.r)
                : 0.0h;

            if (_SurfaceMode > 1.5h)
            {
                rockMask = 1.0;
                iceMask = 0.0;
                snowRetention = 0.0;
            }
            else if (_SurfaceMode > 0.5h)
            {
                iceMask = 1.0;
                rockMask = saturate(rockSlope * 0.28);
                snowRetention = 0.0;
            }
            else if (_ActiveMaterial > 0.5h)
            {
                iceMask = max(iceMask, layeredIce);
                snowRetention = saturate(snowRetention * (1.0 - layeredIce));
            }

            float iceDepth = saturate(0.22 + slope * 0.82 + (1.0 - macro) * 0.16);
            fixed3 ice = lerp(_IceLight.rgb, _IceDeep.rgb, iceDepth);
            ice *= lerp(0.94.xxx, snowPhoto, 0.10);

            float rockShade = saturate(0.30 + up * 0.48 + macro * 0.21);
            fixed3 rock = lerp(_RockShadow.rgb, _RockLight.rgb, rockShade);
            rock *= lerp(0.72.xxx, rockPhoto, 0.64);

            fixed3 albedo = lerp(ice, snow, snowRetention);
            albedo = lerp(albedo, rock, rockMask);
            if (_ActiveMaterial > 0.5h)
            {
                albedo *= lerp(1.0.xxx, layerTint, 0.24);
                albedo *= lerp(1.0h, 0.72h, compact);
            }

            half snowSmooth = lerp(0.08h, 0.25h, 1.0h - snowRough);
            half iceSmooth = 0.70h;
            half rockSmooth = lerp(0.04h, 0.16h, 1.0h - rockRough);
            half smoothness = lerp(iceSmooth, snowSmooth, snowRetention);
            smoothness = lerp(smoothness, rockSmooth, rockMask);
            smoothness = lerp(smoothness, 0.44h, compact);

            float2 wind = normalize(_EverestWindDir.xz + float2(1e-4, 1e-4));
            float across = dot(IN.worldPos.xz, float2(-wind.y, wind.x));
            float along = dot(IN.worldPos.xz, wind);
            float streak = 0.5 + 0.5 * sin(across * 8.5 + sin(along * 0.75) * 1.4);
            albedo *= 1.0 - snowRetention * _EverestWindStrength * (1.0 - streak) * 0.045;

            float fresnel = pow(1.0 - saturate(dot(normalize(IN.viewDir), n)), 3.0);
            fixed3 rimColor = lerp(_IceLight.rgb, _SnowLight.rgb, snowRetention);
            half rimMask = saturate((1.0 - rockMask) * fresnel * _RimStrength);

            o.Albedo = saturate(albedo);
            o.Gloss = smoothness;
            o.Specular = 0.22;
            o.Emission = rimColor * rimMask;
            o.Alpha = 1.0;
        }
        ENDCG
    }
    FallBack "Standard"
}
