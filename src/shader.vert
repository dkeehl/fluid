#version 450

layout(location=0) in vec3 a_position;
layout(location=1) in vec2 a_uv;

layout(location=0) out vec2 v_uv;

layout(set=1, binding=0) uniform Uniforms {
  mat4 u_view_proj;
};

void main() {
  v_uv = a_uv;
  gl_Position = u_view_proj * vec4(a_position, 1.0);
}

