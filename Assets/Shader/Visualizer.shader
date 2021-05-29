﻿Shader "Hidden/BlazePose/Visualizer"
{
    CGINCLUDE

    #include "UnityCG.cginc"

    StructuredBuffer<float4> _vertices;
    float2 _uiScale;
    float4 _linePair[35];

    struct v2f{
        float4 position: SV_POSITION;
        float4 color: COLOR;
    };

    v2f VertexPoint(uint vid : SV_VertexID, uint iid : SV_InstanceID)
    {   
        float4 p = _vertices[iid];
        float score = p.w;
        const float size = 0.01;

        float x = p.x + size * lerp(-0.5, 0.5, vid & 1);
        float y = p.y + size * lerp(-0.5, 0.5, vid < 2 || vid == 5);

        x = (2 * x - 1) * _uiScale.x / _ScreenParams.x;
        y = (2 * y - 1) * _uiScale.y / _ScreenParams.y;

        v2f o;
        o.position = float4(x, y, 0, 1);
        o.color = float4(1, 0, 0, score);
        return o;
    }

    v2f VertexLine(uint vid : SV_VertexID, uint iid : SV_InstanceID)
    {   
        uint2 pairIndex = (uint2)_linePair[iid].xy;
        float4 p0 = _vertices[pairIndex[0]];
        float4 p1 = _vertices[pairIndex[1]];

        float2 p0_p1 = p1.xy - p0.xy;
        float2 orthogonal = float2(-p0_p1.y, p0_p1.x);
        float len = length(p0_p1);
        const float size = 0.005;
        
        float2 p = p0.xy + len * lerp(0, 1, vid & 1) * normalize(p0_p1);
        p += size * lerp(-0.5, 0.5, vid < 2 || vid == 5) * normalize(orthogonal);

        float score = lerp(p0.w, p1.w, vid & 1);

        p = (2 * p - 1) * _uiScale / _ScreenParams.xy;

        v2f o;
        o.position = float4(p, 0, 1);
        o.color = float4(0, 1, 0, score);
        return o;
    }

    float4 Fragment(v2f i): SV_Target
    {
        return i.color;
    }

    ENDCG

    SubShader
    {
        ZWrite Off ZTest Always Cull Off
        Blend SrcAlpha OneMinusSrcAlpha
        Pass
        {
            CGPROGRAM
            #pragma vertex VertexPoint
            #pragma fragment Fragment
            ENDCG
        }
        Pass
        {
            CGPROGRAM
            #pragma vertex VertexLine
            #pragma fragment Fragment
            ENDCG
        }
    }
}