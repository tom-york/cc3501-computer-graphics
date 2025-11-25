#version 330

in float v_hue;
in float v_saturation;

out vec4 outColor;

// funcion de conversion HSV a RGB copiada desde color
vec3 hsv_to_rgb(vec3 hsv_color) {
    float h = mod(hsv_color.x, 360.0);
    float s = hsv_color.y;
    float v = hsv_color.z;

    float c = v * s;
    float h_prima = h / 60.0;
    float x = c * (1.0 - abs(mod(h_prima, 2.0) - 1));
    float m = v - c;

    vec3 rgb;

    if(h_prima < 1.0) {
        rgb = vec3(c, x, 0.0);
    } else if(h_prima <= 2.0) {
        rgb = vec3(x, c, 0.0);
    } else if(h_prima <= 3.0) {
        rgb = vec3(0.0, c, x);
    } else if(h_prima < 4.0) {
        rgb = vec3(0.0, x, c);
    } else if(h_prima < 5.0) {
        rgb = vec3(x, 0.0, c);
    } else {
        rgb = vec3(c, 0.0, x);
    }

    return clamp(rgb + m, 0.0, 1.0);
}

void main() {
    // suavizar value en funcion de la densidad normalizada (saturation)
    float value = smoothstep(0.1, 0.2, v_saturation) * v_saturation;
    vec3 rgb = hsv_to_rgb(vec3(v_hue, v_saturation, value));
    outColor = vec4(rgb, 1.0);
}