#version 330 core
#include common.glsl
uniform vec2      iResolution;   // 视口分辨率 (in pixels)
uniform float     iTime;         // 时间 (in seconds)
uniform int       iFrame;        // 当前帧

uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform sampler2D iChannel4;
uniform sampler2D iChannel5;
uniform sampler2D iChannel6;
uniform sampler2D iChannel7;




out vec4 fragColor;


void main() {
    vec2 uv = gl_FragCoord.xy/iResolution.xy;
    fragColor = vec4(uv,0. ,1.0); // 渲染绿色
}