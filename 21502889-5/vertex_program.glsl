#version 330

in vec2 position;
in float hue;
in float saturation;

out float v_hue;
out float v_saturation;


void main()
{
    gl_PointSize = 2.0;
    gl_Position = vec4(position, 0.0, 1.0);
    v_hue = hue;
    v_saturation = saturation;
}