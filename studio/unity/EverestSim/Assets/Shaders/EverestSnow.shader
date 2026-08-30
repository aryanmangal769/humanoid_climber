Shader "Everest/Snow"
{
    Properties
    {
        _MainTex ("Snow Albedo", 2D) = "white" {}
        _NormalMap ("Snow Normal", 2D) = "bump" {}
        _RoughnessMap ("Snow Roughness", 2D) = "white" {}

        _FreshLight ("Fresh Snow Light", Color) = (0.76, 0.87, 0.96, 1)
        _FreshShadow ("Fresh Snow Shadow", Color) = (0.16, 0.30, 0.46, 1)
        _CompactLight ("Compacted Light", Color) = (0.52, 0.70, 0.82, 1)
        _CompactShadow ("Compacted Shadow", Color) = (0.11, 0.25, 0.40, 1)

        _NormalStrength ("Normal Strength", Range(0,2)) = 0.75
        _FreshSmoothness ("Fresh Smoothness", Range(0,1)) = 0.16
        _CompactSmoothness ("Compacted Smoothness", Range(0,1)) = 0.50
        _TextureScale ("Texture Scale", Float) = 1.7
        _RimStrength ("Snow Rim", Range(0,0.5)) = 0.16
        _SparkleStrength ("Crystal Sparkle", Range(0,0.2)) = 0.05
        _WindRippleStrength ("Wind Ripple", Range(0,0.25)) = 0.07
        _SurfaceMode ("Surface Mode", Range(0,1)) = 0
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 300

        CGPROGRAM
        #pragma surface surf EverestSnowToon fullforwardshadows
        #pragma target 3.0

        sampler2D _MainTex;
        sampler2D _NormalMap;
        sampler2D _RoughnessMap;

        fixed4 _FreshLight;
        fixed4 _FreshShadow;
        fixed4 _CompactLight;
        fixed4 _CompactShadow;

        half _NormalStrength;
        half _FreshSmoothness;
        half _CompactSmoothness;
        float _TextureScale;
        half _RimStrength;
        half _SparkleStrength;
        half _WindRippleStrength;
        half _SurfaceMode;

        float4 _EverestWindDir;
        float _EverestWindStrength;
        float4 _EverestActiveCenter;
        float _EverestActiveRadius;

        struct Input
        {
            float2 uv_MainTex;
            float4 color : COLOR;
            float3 worldPos;
            float3 worldNormal;
            float3 viewDir;
            INTERNAL_DATA
        };

        inline float Hash31(float3 p)
        {
            p = frac(p * 0.1031);
            p += dot(p, p.yzx + 33.33);
            return frac((p.x + p.y) * p.z);
        }

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

        inline half4 LightingEverestSnowToon(SurfaceOutput s, half3 lightDir, half3 viewDir, half atten)
        {
            half compact = saturate(s.Specular);
            half ice = saturate(_SurfaceMode);
            half ndl = saturate(dot(s.Normal, lightDir));
            half b1 = smoothstep(0.15h, 0.34h, ndl * atten);
            half b2 = smoothstep(0.62h, 0.82h, ndl * atten);
            half ramp = 0.30h + 0.28h * b1 + 0.42h * b2;
            half3 h = normalize(lightDir + viewDir);
            half specRaw = pow(saturate(dot(s.Normal, h)), lerp(24.0h, 92.0h, s.Gloss));
            half spec = smoothstep(0.58h, 0.86h, specRaw) * s.Gloss * atten * lerp(1.0h, 1.65h, ice);
            half wrap = saturate((dot(s.Normal, lightDir) + 0.35h) / 1.35h);
            half3 scatter = lerp(_FreshLight.rgb, _CompactLight.rgb, compact) * (1.0h - b1) * wrap * 0.08h;
            half3 ambient = UNITY_LIGHTMODEL_AMBIENT.rgb * s.Albedo * 0.40h;
            return half4(ambient + s.Albedo * _LightColor0.rgb * ramp * atten * 0.82h + scatter + spec * _LightColor0.rgb + s.Emission, s.Alpha);
        }

        void surf(Input IN, inout SurfaceOutput o)
        {
            half compact = saturate(IN.color.a);
            fixed3 layerTint = saturate(IN.color.rgb);
            // ICE layer presets are deliberately much bluer than snow/firn.
            // This keeps the material identity per-cell, so an exposed deeper
            // ICE layer can visually become ice without a renderer-side physics guess.
            half layeredIce = smoothstep(0.16h, 0.38h, layerTint.b - layerTint.r);
            half iceMask = max(saturate(_SurfaceMode), layeredIce);
            float2 uv = IN.uv_MainTex * _TextureScale;
            float3 geometryNormal = normalize(IN.worldNormal);

            fixed3 tex = SampleTriplanar(_MainTex, IN.worldPos, geometryNormal, _TextureScale).rgb;
            tex = lerp(0.78.xxx, tex, 0.66);

            half3 normal = UnpackNormal(tex2D(_NormalMap, uv));
            half normalScale = _NormalStrength * lerp(1.0h, 0.30h, compact);
            normal.xy *= normalScale;
            normal.z = sqrt(saturate(1.0h - dot(normal.xy, normal.xy)));
            o.Normal = normal;

            float3 wn = normalize(WorldNormalVector(IN, o.Normal));
            float up = saturate(wn.y);
            float shade = saturate(up * 0.80 + 0.18);

            fixed3 fresh = lerp(_FreshShadow.rgb, _FreshLight.rgb, shade);
            fixed3 packed = lerp(_CompactShadow.rgb, _CompactLight.rgb, shade);
            fixed3 tint = lerp(fresh, packed, compact);
            fixed3 glacier = lerp(fixed3(0.035, 0.20, 0.34), fixed3(0.30, 0.76, 0.93), shade);
            tint = lerp(tint, glacier, iceMask);
            tint *= lerp(1.0.xxx, layerTint, 0.46);

            // Wind-aligned striations. These are purely visual; Newton still owns the
            // geometry. The global vector is driven by backend environment state.
            float2 wind = normalize(_EverestWindDir.xz + float2(0.0001, 0.0001));
            float acrossWind = dot(IN.worldPos.xz, float2(-wind.y, wind.x));
            float alongWind = dot(IN.worldPos.xz, wind);
            float ripples = sin(acrossWind * 15.0 + sin(alongWind * 1.3) * 1.7);
            ripples = ripples * 0.5 + 0.5;
            float rippleAmount = _WindRippleStrength * saturate(_EverestWindStrength)
                * (1.0 - compact * 0.75) * (1.0 - iceMask * 0.92);
            tint *= 1.0 - rippleAmount * (1.0 - ripples);

            o.Albedo = saturate(tex * tint);
            half rough = SampleTriplanar(_RoughnessMap, IN.worldPos, geometryNormal, _TextureScale).r;
            half freshSmooth = lerp(0.08h, _FreshSmoothness, 1.0h - rough);
            o.Gloss = lerp(lerp(freshSmooth, _CompactSmoothness, compact), 0.72h, iceMask);
            o.Specular = max(compact, iceMask * 0.72h);

            float3 view = normalize(IN.viewDir);
            float fresnel = pow(1.0 - saturate(dot(view, wn)), 3.0);
            float grain = Hash31(floor(IN.worldPos * 34.0));
            float sparkle = step(0.985, grain) * pow(saturate(dot(reflect(-view, wn), normalize(float3(0.25, 0.86, 0.44)))), 18.0);
            sparkle *= (1.0 - compact * 0.60);

            fixed3 rimBase = lerp(lerp(_FreshLight.rgb, _CompactLight.rgb, compact), fixed3(0.30, 0.76, 0.93), iceMask);
            fixed3 rim = rimBase * fresnel * _RimStrength;
            o.Emission = rim + _FreshLight.rgb * sparkle * _SparkleStrength;
            o.Alpha = 1.0;
        }
        ENDCG
    }
    FallBack "Standard"
}
