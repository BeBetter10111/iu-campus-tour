#version 330 core
in vec3 vDir;
uniform samplerCube uSkybox;
uniform vec3 uSunDir;    
uniform vec3 uSunColor;
out vec4 FragColor;

void main() {
    vec3 dir = normalize(vDir);
    vec3 sky = texture(uSkybox, dir).rgb;

    // Day/night: darken the sky as the sun sinks, pink/orange glow near the horizon at dusk/dawn.
    float h = normalize(uSunDir).y;
    float day = smoothstep(-0.25, 0.2, h);
    float dusk = (1.0 - smoothstep(0.0, 0.35, abs(h))) * pow(max(1.0 - abs(dir.y), 0.0), 2.0);
    sky = sky * mix(vec3(0.03, 0.04, 0.10), vec3(1.0), day) + vec3(0.60, 0.30, 0.35) * dusk * 0.7;

    // c is the cosine between the view direction and sun direction.
    float c = max(dot(dir, normalize(uSunDir)), 0.0);
    float moon = smoothstep(0.9993, 0.9997, max(dot(dir, -normalize(uSunDir)), 0.0)) * (1.0 - day);
    float disc = smoothstep(0.9990, 0.9996, c);  
    float glow = pow(c, 48.0) * 0.35;             

    FragColor = vec4(sky + uSunColor * (disc + glow) + vec3(0.8, 0.85, 0.95) * moon, 1.0);
}