#include "HdrColor.hlsli"
Texture2D<float4> hdrBase:register(t0);
Texture2D<float4> proxyInput:register(t1);
Texture2D<float4> proxyOutput:register(t2);
RWTexture2D<float4> outputTex:register(u0);
cbuffer Constants:register(b0){
    uint outW,outH,pad0,pad1;
    float encodeMode; // <= 0: linear FP16 workRgba; > 1.5: direct HDR10 PQ + dither into videoFrame
    float ditherStep;
    float pad2,pad3;
};
float3 Decode(float3 c){return select(c<=.04045,c/12.92,pow((c+.055)/1.055,2.4));}
float4 CubicWeights(float t){
    float t2=t*t,t3=t2*t;
    return float4(-0.5*t+t2-0.5*t3,1.0-2.5*t2+1.5*t3,0.5*t+2.0*t2-1.5*t3,-0.5*t2+0.5*t3);
}
[numthreads(16,16,1)]void main(uint3 id:SV_DispatchThreadID){
    uint w,h,sw,sh;outputTex.GetDimensions(w,h);hdrBase.GetDimensions(sw,sh);
    if(id.x>=w||id.y>=h)return;
    float3 base=0,proxy=0;
    if(sw==w&&sh==h){
        base=hdrBase[id.xy].rgb;
        proxy=proxyInput[id.xy].rgb;
    }else{
        float2 pos=(float2(id.xy)+.5)*float2(sw,sh)/float2(w,h)-.5;
        int2 q=int2(floor(pos)),last=int2(sw,sh)-1;
        float2 t=pos-float2(q);
        float4 wx=CubicWeights(t.x),wy=CubicWeights(t.y);
        float3 baseLo=1e30,baseHi=-1e30,proxyLo=1e30,proxyHi=-1e30;
        [unroll]for(int y=0;y<4;++y){
            [unroll]for(int x=0;x<4;++x){
                int2 p=clamp(q+int2(x-1,y-1),int2(0,0),last);
                float weight=wx[x]*wy[y];
                float3 b=hdrBase[p].rgb;
                float3 pr=proxyInput[p].rgb;
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
    const float3 restored=HdrRestore(base,Decode(proxy),Decode(proxyOutput[id.xy].rgb));
    if(encodeMode>1.5){outputTex[id.xy]=float4(HdrEncodePq(restored)+OutputDither(id.xy,ditherStep),1.0);return;}
    outputTex[id.xy]=float4(restored,1.0);
}
