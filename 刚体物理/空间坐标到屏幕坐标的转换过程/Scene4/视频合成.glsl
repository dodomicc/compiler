#version 330 core
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform vec2 iResolution;   // 视口分辨率 (in pixels)
out vec4 fragColor;
void main( )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = (gl_FragCoord.xy /iResolution.xy); 
    uv.y = 1. - uv.y; 
    vec3 col1 = texture(iChannel0,uv).xyz;
    vec3 col2 = texture(iChannel1,uv).xyz;
    float alpha = col2.x > 0.5 || col2.y > 0.5 || col2.z > 0.5 ? 1. : 0.;
    vec3 col = mix(col1, col2, alpha);
    // Output to screen
    fragColor = vec4(col,1.0);
}