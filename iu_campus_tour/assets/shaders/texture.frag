#version 330 core
in vec3 vNormal;
in vec2 vUV;

uniform sampler2D uTexture;   // Gắn với texture unit 0
out vec4 FragColor;

void main() {
    vec3 albedo = texture(uTexture, vUV).rgb;

    vec3 n = normalize(vNormal);
    vec3 L = normalize(vec3(0.4, 1.0, 0.3));
    float diff = dot(n, L) * 0.5 + 0.5;

    FragColor = vec4(albedo * (0.35 + 0.65 * diff), 1.0);
}