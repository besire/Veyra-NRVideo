// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 DosukaSOL and the PushFPS contributors (https://github.com/DosukaSOL/PushFPS).
// Ported into Veyra on 2026-11-19 from PushFPS shaders/PushHudLockCompose.hlsl (commit 3be641ca) and adapted: renamed and wired
// into Veyra's settings, engine and interface.
// Live HUD Lock, step 2 (pipeline/HudLock.h): a generated frame takes its locked
// pixels from the held real frame. A pixel is locked when it has been still for about a second
// itself and it, or a neighbour that is just as still, has an edge: the inside of a thin stroke
// and the pixel beside an edge belong to the overlay too. A pixel that moves is never locked.
Texture2D<float4> generatedTex : register(t0);
Texture2D<float4> heldTex : register(t1);
Texture2D<float2> stateTex : register(t2);
RWTexture2D<float4> outputTex : register(u0);
cbuffer Constants : register(b0) {
    uint width, height, outputMode, inPlace;
    uint pad1, pad2, pad3, pad4;
};
float stillOf(float2 s) { return smoothstep(0.80, 0.95, s.x); }
float edgeOf(float2 s) {
    const float edge = outputMode == 1 ? 0.03 : 0.04;
    return smoothstep(edge, edge * 2.5, s.y);
}
float scoreAt(int2 p) {
    const float2 q = stateTex[clamp(p, int2(0, 0), int2(int(width) - 1, int(height) - 1))];
    return edgeOf(q) * stillOf(q);
}

groupshared uint gAnyStill;
groupshared float sScore[20][20];

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID, uint3 gid : SV_GroupID, uint gidx : SV_GroupIndex) {
    if (gidx == 0) gAnyStill = 0;
    GroupMemoryBarrierWithGroupSync();

    const bool inBounds = (id.x < width && id.y < height);
    const int2 clampedId = clamp(int2(id.xy), int2(0, 0), int2(int(width) - 1, int(height) - 1));
    const float2 st = stateTex[clampedId];
    const float still = stillOf(st);
    if (inBounds && still > 0) InterlockedOr(gAnyStill, 1u);
    GroupMemoryBarrierWithGroupSync();

    if (gAnyStill == 0) {
        if (inBounds && inPlace == 0) outputTex[id.xy] = generatedTex[id.xy];
        return;
    }

    sScore[tid.y + 2][tid.x + 2] = edgeOf(st) * still;
    const int2 base = int2(gid.xy * 16u);
    if (height >= 1800) {
        if (gidx < 144) {
            uint hx = 0, hy = 0;
            if (gidx < 40) {
                hx = gidx % 20; hy = gidx / 20;
            } else if (gidx < 80) {
                const uint k = gidx - 40;
                hx = k % 20; hy = 18 + k / 20;
            } else if (gidx < 112) {
                const uint k = gidx - 80;
                hx = k & 1u; hy = 2 + (k >> 1);
            } else {
                const uint k = gidx - 112;
                hx = 18 + (k & 1u); hy = 2 + (k >> 1);
            }
            sScore[hy][hx] = scoreAt(base + int2(int(hx) - 2, int(hy) - 2));
        }
    } else {
        if (gidx < 68) {
            uint hx = 0, hy = 0;
            if (gidx < 18) {
                hx = 1 + gidx; hy = 1;
            } else if (gidx < 36) {
                hx = 1 + (gidx - 18); hy = 18;
            } else if (gidx < 52) {
                hx = 1; hy = 2 + (gidx - 36);
            } else {
                hx = 18; hy = 2 + (gidx - 52);
            }
            sScore[hy][hx] = scoreAt(base + int2(int(hx) - 2, int(hy) - 2));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (!inBounds) return;
    float edge = 0;
    if (still > 0) {
        if (height >= 1800) {
            [unroll] for (int y = -2; y <= 2; ++y)
                [unroll] for (int x = -2; x <= 2; ++x)
                    edge = max(edge, sScore[int(tid.y) + 2 + y][int(tid.x) + 2 + x]);
        } else {
            [unroll] for (int y = -1; y <= 1; ++y)
                [unroll] for (int x = -1; x <= 1; ++x)
                    edge = max(edge, sScore[int(tid.y) + 2 + y][int(tid.x) + 2 + x]);
        }
    }
    const float w = still * edge;
    if (inPlace != 0) {
        if (w > 0) {
            const float4 g = outputTex[id.xy];
            outputTex[id.xy] = float4(lerp(g.rgb, heldTex[id.xy].rgb, w), g.a);
        }
        return;
    }
    const float4 g = generatedTex[id.xy];
    outputTex[id.xy] = w > 0 ? float4(lerp(g.rgb, heldTex[id.xy].rgb, w), g.a) : g;
}
