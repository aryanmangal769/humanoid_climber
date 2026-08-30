Shader "Everest/VolumetricClouds"
{
    Properties
    {
        _CloudColor ("Cloud Color", Color) = (0.72, 0.81, 0.90, 1)
        _ShadowColor ("Cloud Shadow", Color) = (0.18, 0.28, 0.40, 1)
        _CloudRadius ("Radius", Float) = 120
        _CloudThickness ("Thickness", Float) = 30
        _CloudDensity ("Density", Range(0,1)) = 0.28
        _CloudCoverage ("Coverage", Range(0,1)) = 0.42
        _CloudSpeed ("Speed", Range(0,2)) = 0.35
        _CloudQuality ("Quality", Range(0,1)) = 0.55
    }

    SubShader
    {
        Tags { "Queue"="Transparent+40" "RenderType"="Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #include "UnityCG.cginc"

            fixed4 _CloudColor;
            fixed4 _ShadowColor;
            float4 _CloudCenter;
            float4 _CloudWind;
            float _CloudRadius;
            float _CloudThickness;
            float _CloudDensity;
            float _CloudCoverage;
            float _CloudSpeed;
            float _CloudQuality;

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                return o;
            }

            float hash31(float3 p)
            {
                p = frac(p * 0.1031);
                p += dot(p, p.yzx + 33.33);
                return frac((p.x + p.y) * p.z);
            }

            float valueNoise(float3 p)
            {
                float3 i = floor(p);
                float3 f = frac(p);
                f = f * f * (3.0 - 2.0 * f);
                float n000 = hash31(i + float3(0,0,0));
                float n100 = hash31(i + float3(1,0,0));
                float n010 = hash31(i + float3(0,1,0));
                float n110 = hash31(i + float3(1,1,0));
                float n001 = hash31(i + float3(0,0,1));
                float n101 = hash31(i + float3(1,0,1));
                float n011 = hash31(i + float3(0,1,1));
                float n111 = hash31(i + float3(1,1,1));
                float nx00 = lerp(n000, n100, f.x);
                float nx10 = lerp(n010, n110, f.x);
                float nx01 = lerp(n001, n101, f.x);
                float nx11 = lerp(n011, n111, f.x);
                return lerp(lerp(nx00, nx10, f.y), lerp(nx01, nx11, f.y), f.z);
            }

            float fbm(float3 p)
            {
                float value = 0.0;
                float amp = 0.55;
                value += valueNoise(p) * amp;
                p = p * 2.03 + 11.7;
                amp *= 0.5;
                value += valueNoise(p) * amp;
                p = p * 2.01 + 7.2;
                amp *= 0.5;
                value += valueNoise(p) * amp;
                return value;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float3 ro = _WorldSpaceCameraPos;
                float3 rd = normalize(i.worldPos - ro);
                float3 oc = ro - _CloudCenter.xyz;
                float b = dot(oc, rd);
                float c = dot(oc, oc) - _CloudRadius * _CloudRadius;
                float h = b * b - c;
                if (h <= 0.0) discard;
                h = sqrt(h);
                float t0 = max(0.0, -b - h);
                float t1 = -b + h;
                if (t1 <= t0) discard;

                int steps = (int)lerp(6.0, 22.0, saturate(_CloudQuality));
                float dt = (t1 - t0) / max(1, steps);
                float alpha = 0.0;
                float lightAccum = 0.0;
                float3 wind = _CloudWind.xyz * (_Time.y * _CloudSpeed * 0.018);

                // A ray through a broad cloud layer crosses enough unrelated
                // 3D noise that its opacity otherwise converges to a nearly
                // uniform grey veil.  Keep a lower-frequency, ray-coherent
                // coverage field at the layer mid-plane so storm clouds retain
                // readable banks and holes instead of looking like extra fog.
                float planeDenominator = rd.y;
                float planeT = abs(planeDenominator) > 0.015
                    ? (_CloudCenter.y - ro.y) / planeDenominator
                    : (t0 + t1) * 0.5;
                planeT = clamp(planeT, t0, t1);
                float3 coveragePoint = ro + rd * planeT;
                float2 coverageUv = coveragePoint.xz * 0.013 + wind.xz * 0.55;
                float coverageNoise = fbm(float3(coverageUv.x, 4.7, coverageUv.y));
                coverageNoise = lerp(
                    coverageNoise,
                    valueNoise(float3(coverageUv * 2.17, 13.4)),
                    0.28);
                float coverageThreshold = lerp(0.68, 0.29, saturate(_CloudCoverage));
                float bankMask = smoothstep(
                    coverageThreshold - 0.10,
                    coverageThreshold + 0.10,
                    coverageNoise);

                [loop]
                for (int s = 0; s < 22; ++s)
                {
                    if (s >= steps || alpha > 0.96) break;
                    float t = t0 + (s + 0.5) * dt;
                    float3 p = ro + rd * t;
                    float vertical = abs(p.y - _CloudCenter.y) / max(1.0, _CloudThickness * 0.5);
                    float heightMask = saturate(1.0 - vertical);
                    heightMask = heightMask * heightMask * (3.0 - 2.0 * heightMask);
                    if (heightMask <= 0.001) continue;

                    float3 q = p * 0.032 + wind;
                    float n = fbm(q);
                    float threshold = lerp(0.76, 0.34, saturate(_CloudCoverage));
                    float density = saturate((n - threshold) * 4.2) * heightMask * _CloudDensity;
                    density *= lerp(0.025, 1.0, bankMask);
                    float sampleAlpha = 1.0 - exp(-density * dt * 0.20);
                    float remaining = 1.0 - alpha;
                    alpha += sampleAlpha * remaining;
                    lightAccum += sampleAlpha * remaining * saturate(0.35 + n * 0.9);
                }

                alpha *= smoothstep(0.0, 0.72, bankMask);
                if (alpha < 0.003) discard;
                float light = saturate(lightAccum / max(alpha, 0.001));
                fixed3 color = lerp(_ShadowColor.rgb, _CloudColor.rgb, light);
                return fixed4(color, alpha * 0.76);
            }
            ENDCG
        }
    }
}
