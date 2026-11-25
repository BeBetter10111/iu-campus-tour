#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

uniform vec4 uRect;
out vec2 vUV;

void main()
{
    vec2 p = aPos.xy * 0.5 + 0.5;
    gl_Position = vec4(uRect.xy + p * uRect.zw, 0.0, 1.0);
    vUV = aUV;
}
