// Three-step SR (2026-10-28): fits the RTX Video SR result into the smaller
// work extent. Each SR texel is first decoded (SDR, as ScaleBlit's decode) or
// HDR-restored (as HdrVideoSrRestore, base and proxy sampled at the texel's
// centre), then the linear values are averaged with NrDownsample's exact area
// weights. No full-size linear intermediate is written.
#include "HdrColor.hlsli"
Texture2D<float4> hdrBase:register(t0);
Texture2D<float4> proxyInput:register(t1);
Texture2D<float4> proxyOutput:register(t2);
RWTexture2D<float4> outputTex:register(u0);
cbuffer Dims:register(b0){uint sourceW;uint sourceH;uint outW;uint outH;float hdr;float3 unused;}
float3 Decode(float3 c){return select(c<=.04045,c/12.92,pow((c+.055)/1.055,2.4));}
float4 CubicWeights(float t){
    float t2=t*t,t3=t2*t;
    return float4(-0.5*t+t2-0.5*t3,1.0-2.5*t2+1.5*t3,0.5*t+2.0*t2-1.5*t3,-0.5*t2+0.5*t3);
}
float4 Texel(uint2 p){
    const float4 c=proxyOutput[p];
    if(hdr<.5)return float4(Decode(max(c.rgb,0.0)),c.a);
    uint bw,bh;hdrBase.GetDimensions(bw,bh);
    float3 base=0,proxy=0;
    if(bw==sourceW&&bh==sourceH){
        base=hdrBase[p].rgb;
        proxy=proxyInput[p].rgb;
    }else{
        float2 pos=(float2(p)+.5)*float2(bw,bh)/float2(sourceW,sourceH)-.5;
        int2 q=int2(floor(pos)),last=int2(bw,bh)-1;
        float2 t=pos-float2(q);
        float4 wx=CubicWeights(t.x),wy=CubicWeights(t.y);
        float3 baseLo=1e30,baseHi=-1e30,proxyLo=1e30,proxyHi=-1e30;
        [unroll]for(int y=0;y<4;++y){
            [unroll]for(int x=0;x<4;++x){
                int2 s=clamp(q+int2(x-1,y-1),int2(0,0),last);
                float weight=wx[x]*wy[y];
                float3 b=hdrBase[s].rgb;
                float3 pr=proxyInput[s].rgb;
                base+=b*weight;
                proxy+=pr*weight;
                if(x>=1&&x<=2&&y>=1&&y<=2){
                    baseLo=min(baseLo,b);baseHi=max(baseHi,b);
                    proxyLo=min(proxyLo,pr);proxyHi=max(proxyHi,pr);
                }
            }
        }
        base=clamp(base,baseLo,baseHi);
        proxy=clamp(proxy,proxyLo,proxyHi);
    }
    return float4(HdrRestore(base,Decode(proxy),Decode(c.rgb)),1);
}
[numthreads(16,16,1)]void main(uint3 id:SV_DispatchThreadID){
    if(id.x>=outW||id.y>=outH)return;
    float2 low=float2(id.xy)*float2(sourceW,sourceH)/float2(outW,outH);
    float2 high=float2(id.xy+1)*float2(sourceW,sourceH)/float2(outW,outH);
    float4 sum=0;float weights=0;
    for(uint y=(uint)floor(low.y);y<(uint)ceil(high.y);++y)for(uint x=(uint)floor(low.x);x<(uint)ceil(high.x);++x){
        float weight=max(0,min(high.x,x+1.0)-max(low.x,(float)x))*max(0,min(high.y,y+1.0)-max(low.y,(float)y));
        sum+=Texel(uint2(min(x,sourceW-1),min(y,sourceH-1)))*weight;weights+=weight;
    }
    outputTex[id.xy]=sum/max(weights,1e-6);
}
