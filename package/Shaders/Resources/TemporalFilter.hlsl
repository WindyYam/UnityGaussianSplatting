// Based on very trimmed down version of Unity's URP TAA,
// https://github.com/Unity-Technologies/Graphics/blob/81a1ed4/Packages/com.unity.render-pipelines.universal/Shaders/PostProcessing/TemporalAA.hlsl
//
// Currently none of motion stuff is used, so this is not much more than
// just a stupid accumulation buffer type of thing.

#include "HLSLSupport.cginc"

#ifndef TAA_YCOCG
#define TAA_YCOCG 1
#endif

#define TAA_GAMMA_SPACE_POST 1 // splats are rendered in sRGB

#ifndef TAA_PERCEPTUAL_SPACE
#define TAA_PERCEPTUAL_SPACE 1
#endif

#define HALF_MIN 6.103515625e-5  // 2^-14, the same value for 10, 11 and 16-bit: https://www.khronos.org/opengl/wiki/Small_Float_Formats

// This function take a rgb color (best is to provide color in sRGB space)
// and return a YCoCg color in [0..1] space for 8bit (An offset is apply in the function)
// Ref: http://www.nvidia.com/object/real-time-ycocg-dxt-compression.html
#define YCOCG_CHROMA_BIAS (128.0 / 255.0)
half3 RGBToYCoCg(half3 rgb)
{
    half3 YCoCg;
    YCoCg.x = dot(rgb, half3(0.25, 0.5, 0.25));
    YCoCg.y = dot(rgb, half3(0.5, 0.0, -0.5)) + YCOCG_CHROMA_BIAS;
    YCoCg.z = dot(rgb, half3(-0.25, 0.5, -0.25)) + YCOCG_CHROMA_BIAS;
    return YCoCg;
}

half3 YCoCgToRGB(half3 YCoCg)
{
    half Y = YCoCg.x;
    half Co = YCoCg.y - YCOCG_CHROMA_BIAS;
    half Cg = YCoCg.z - YCOCG_CHROMA_BIAS;
    half3 rgb;
    rgb.r = Y + Co - Cg;
    rgb.g = Y + Cg;
    rgb.b = Y - Co - Cg;
    return rgb;
}

Texture2D _CameraDepthTexture;
Texture2D _GaussianSplatRT;
float4 _GaussianSplatRT_TexelSize;
Texture2D _TaaMotionVectorTex; // RG motion (alpha-weighted NDC delta from previous to current)
Texture2D _TaaAccumulationTex;

cbuffer TemporalAAData {
    float4 _TaaMotionVectorTex_TexelSize;   // (1/w, 1/h, w, h)
    float4 _TaaAccumulationTex_TexelSize;   // (1/w, 1/h, w, h)

    half _TaaFrameInfluence;
    half _TaaVarianceClampScale;
}
SamplerState sampler_LinearClamp, sampler_PointClamp;

// Per-pixel camera backwards velocity
half2 GetVelocityWithOffset(float2 uv, half2 depthOffsetUv)
{
    half2 mv = _TaaMotionVectorTex.Sample(sampler_LinearClamp, uv + _TaaMotionVectorTex_TexelSize.xy * depthOffsetUv).xy;
    // NDC to UV: uv = ndc*0.5 + 0.5, but UV y grows up in our splat NDC while texture UV.y grows down, so flip Y.
    half2 vel = half2(mv.x * 0.5, -mv.y * 0.5);
    return vel;
}

half4 PostFxSpaceToLinear(float4 src)
{
// gamma 2.0 is a good enough approximation
#if TAA_GAMMA_SPACE_POST
    return half4(src.xyz * src.xyz, src.w);
#else
    return src;
#endif
}

half4 LinearToPostFxSpace(float4 src)
{
#if TAA_GAMMA_SPACE_POST
    return half4(sqrt(src.xyz), src.w);
#else
    return src;
#endif
}

// Working Space: The color space that we will do the calculation in.
// Scene: The incoming/outgoing scene color. Either linear or gamma space
half4 SceneToWorkingSpace(half4 src)
{
    half4 linColor = PostFxSpaceToLinear(src);
#if TAA_YCOCG
    half4 dst = half4(RGBToYCoCg(linColor.xyz), linColor.w);
#else
    half4 dst = src;
#endif
    return dst;
}

half4 WorkingSpaceToScene(half4 src)
{
#if TAA_YCOCG
    half4 linColor = half4(YCoCgToRGB(src.xyz), src.w);
#else
    half4 linColor = src;
#endif

    half4 dst = LinearToPostFxSpace(linColor);
    return dst;
}

half4 SampleColorPoint(float2 uv, int2 texelOffset)
{
    return _GaussianSplatRT.Sample(sampler_PointClamp, uv, texelOffset);
}

// Simple temporal blend: historyFrame * (1 - _TaaFrameInfluence) + currentFrame * _TaaFrameInfluence
half4 DoTemporalAA(float2 uv)
{
    // Current frame color (in working space)
    half4 currentFrame = SceneToWorkingSpace(SampleColorPoint(uv, int2(0,0)));

    // Sample motion vectors to locate the previous-frame UV for this pixel
    half2 velocity = GetVelocityWithOffset(uv, half2(0.0, 0.0)); // already current-prev in UV space
    float2 historyUv = uv - velocity;

    // If history UV is out of buffer bounds, use current frame directly as history to avoid sampling invalid history.
    half4 historyFrame;
    if (historyUv.x < 0.0 || historyUv.x > 1.0 || historyUv.y < 0.0 || historyUv.y > 1.0)
    {
        historyFrame = currentFrame;
    }
    else
    {
        // Simplified history sampling: always use a single linear sample and convert to working space
        historyFrame = SceneToWorkingSpace(_TaaAccumulationTex.Sample(sampler_LinearClamp, historyUv));
    }

    half frameInfluence = _TaaFrameInfluence;

    // Blend in working space
    half4 workingColor = historyFrame * (1.0 - frameInfluence) + currentFrame * frameInfluence;

    half4 dstSceneColor = WorkingSpaceToScene(workingColor);

    return half4(max(dstSceneColor, 0.0));
}
