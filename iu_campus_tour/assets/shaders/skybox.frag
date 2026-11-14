#version 330 core
in vec3 vDir;
uniform samplerCube uSkybox;
uniform vec3 uSunDir;    
uniform vec3 uSunColor;
out vec4 FragColor;

void main() {
    vec3 dir = normalize(vDir);
    vec3 sky = texture(uSkybox, dir).rgb;

    // c = cos(goc giua huong nhin va huong mat troi); c -> 1 khi dang nhin thang vao mat troi.
    float c = max(dot(dir, normalize(uSunDir)), 0.0);
    float disc = smoothstep(0.9990, 0.9996, c);  
    float glow = pow(c, 48.0) * 0.35;             

    FragColor = vec4(sky + uSunColor * (disc + glow), 1.0);
}