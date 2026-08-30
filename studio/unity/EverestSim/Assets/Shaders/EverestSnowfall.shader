Shader "Everest/Snowfall"
{
    Properties
    {
        _TintColor ("Tint", Color) = (0.92, 0.97, 1.0, 0.88)
        _Softness ("Edge Softness", Range(0.01, 0.5)) = 0.18
    }

    SubShader
    {
        Tags { "Queue"="Transparent+60" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Blend SrcAlpha OneMinusSrcAlpha
        ColorMask RGB
        Cull Off
        Lighting Off
        ZWrite Off
        ZTest LEqual

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 2.0
            #include "UnityCG.cginc"

            fixed4 _TintColor;
            half _Softness;

            struct appdata
            {
                float4 vertex : POSITION;
                fixed4 color : COLOR;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                fixed4 color : COLOR;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.color = v.color * _TintColor;
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float2 centered = i.uv * 2.0 - 1.0;
                float radius = length(centered);
                half alpha = 1.0 - smoothstep(1.0 - _Softness, 1.0, radius);
                clip(alpha - 0.01);
                return fixed4(i.color.rgb, i.color.a * alpha);
            }
            ENDCG
        }
    }
}
