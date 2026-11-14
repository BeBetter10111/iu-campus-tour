#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;

uniform sampler2D uTexture;      // unit 0: surface color (albedo)
uniform sampler2D uShadowMap;    // unit 1: depth map from the sun

uniform mat4  uLightSpace;
uniform vec3  uLightDir;         // L: direction TOWARD the sun (unit vector)
uniform vec3  uLightColor;
uniform vec3  uAmbient;
uniform vec3  uViewPos;          // Camera position
uniform float uSpecular;         // Material specular reflection intensity
uniform float uShininess;        // Glossiness (exponent n)
uniform int   uBlinn;            // 0 = Phong, 1 = Blinn-Phong
uniform int   uShadowsOn;
uniform float uShadowTexel;      // 1 / shadow map size
uniform float uNormalOffset;     // Normal offset amount
uniform float uBias;
uniform int   uDebug;            // 0 = normal, 1 = shadow view, 2 = normal view

out vec4 FragColor;

// Return the lighting factor: 1.0 = fully lit, 0.0 = in shadow.
float shadowFactor(vec3 worldPos, vec3 N, float NdotL, out float coverage) {
    // (1) NORMAL OFFSET BIAS: shift the point slightly along the normal before sampling
    //     the shadow map. The steeper the surface relative to the light ray (larger sinθ),
    //     the more offset is needed.
    float sinTheta = sqrt(max(1.0 - NdotL * NdotL, 0.0));
    vec3 pos = worldPos + N * (uNormalOffset * sinTheta);

    // (2) Transform into light space and convert to [0,1] for shadow texture lookup:
    //     clip = LightSpace · world; with ortho w = 1, so NDC = clip.xyz.
    //     NDC ∈ [-1,1] -> *0.5 + 0.5 -> [0,1] (UV of the depth map and depth value).
    vec4 lp = uLightSpace * vec4(pos, 1.0);
    vec3 p = (lp.xyz / lp.w) * 0.5 + 0.5;

    // Is the point within the shadow map bounds? (used for debug and soft edge fading)
    vec2 d = abs(p.xy - 0.5) * 2.0;
    float edge = max(d.x, d.y);               // 0 at center, 1 at edge
    coverage = edge < 1.0 ? 1.0 : 0.0;
    if (p.z > 1.0) return 1.0;                // Beyond the sun's far plane -> treat as lit

    // (3) 3x3 PCF: sample 9 points around the current fragment, compare each sample, then average.
    //     Do not interpolate depth (meaningless); instead interpolate the comparison result for soft shadow edges.
    //     A point is "in shadow" when its depth is greater than the nearest depth visible from the sun.
    float lit = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float closest = texture(uShadowMap, p.xy + vec2(x, y) * uShadowTexel).r;
            lit += (p.z - uBias <= closest) ? 1.0 : 0.0;
        }
    }
    lit /= 9.0;

    // (4) Fade out near the shadow map edges so the "lit / shadowed" boundary is not abrupt.
    return mix(lit, 1.0, smoothstep(0.85, 1.0, edge));
}

void main() {
    vec3 albedo = texture(uTexture, vUV).rgb;
    vec3 N = normalize(vNormal);              // Must be normalized again: interpolation can distort length
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uViewPos - vWorldPos); // Direction from the point to the eye

    // ---------------- Phong Model ----------------
    // I = Ambient + Diffuse + Specular
    //
    // Ambient:  I_a = k_a · albedo                         (background light, no direction)
    vec3 ambient = uAmbient * albedo;

    // Diffuse (Lambert):  I_d = max(N·L, 0) · albedo · light_color
    //   N·L = cosθ between the normal and the light ray: perpendicular surfaces are brightest, angled ones are darker.
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = uLightColor * NdotL * albedo;

    // Specular:  I_s = k_s · (R·V)^n          (Phong)
    //   R = reflect(-L, N) = 2(N·L)N - L is the reflected ray; larger n gives a smaller, sharper highlight.
    // Blinn-Phong:  I_s = k_s · (N·H)^n'      with H = normalize(L + V) (half-vector)
    //   Faster and avoids highlight cutoff at large angles; n' ≈ 4n to approximate Phong.
    float spec = 0.0;
    if (NdotL > 0.0) {                         // Surface facing away from the sun has no specular highlight
        if (uBlinn == 1) {
            vec3 H = normalize(L + V);
            spec = pow(max(dot(N, H), 0.0), uShininess * 4.0);
        } else {
            vec3 R = reflect(-L, N);
            spec = pow(max(dot(V, R), 0.0), uShininess);
        }
    }
    vec3 specular = uLightColor * uSpecular * spec;   // Specular is the light's color (not multiplied by albedo)

    // ---------------- Shadows ----------------
    // Ambient is not shadowed (it represents scattered light everywhere),
    // only the direct components (diffuse + specular) are multiplied by the shadow factor.
    float shadow = 1.0;
    float coverage = 1.0;
    if (uShadowsOn == 1 && NdotL > 0.0) {
        shadow = shadowFactor(vWorldPos, N, NdotL, coverage);
    }
    vec3 color = ambient + shadow * (diffuse + specular);

    if (uDebug == 1)      color = vec3(shadow) * mix(vec3(1.0, 0.2, 0.2), vec3(1.0), coverage); // red = outside the shadow map
    else if (uDebug == 2) color = N * 0.5 + 0.5;

    FragColor = vec4(color, 1.0);
}