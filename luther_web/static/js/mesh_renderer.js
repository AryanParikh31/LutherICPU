/**
 * Dedicated WebGL2 4K UV-Textured Solid Triangle Mesh Renderer
 * - Pure Hardware Triangle Rasterization (gl.TRIANGLES & gl.drawElements)
 * - Zero Gaussian point/quad shader math, zero circular discards
 * - Full element vertex + element face binary PLY parser with (u, v) texture mapping
 * - Directional Diffuse + Ambient double-sided illumination
 * - 4K Photographic UV Texture Sampler (u_diffuseMap)
 * - 100% Solid Opaque (gl.disable(gl.BLEND), gl.depthMask(true))
 */

class SolidMeshRenderer {
  constructor(gl, canvas) {
    this.gl = gl;
    this.canvas = canvas;
    this.program = null;
    this.vao = null;
    this.vertexBuffer = null;
    this.normalBuffer = null;
    this.colorBuffer = null;
    this.uvBuffer = null;
    this.indexBuffer = null;
    this.texture = null;
    this.hasTexture = false;
    this.indexCount = 0;
    this.vertexCount = 0;
    this.isLoaded = false;
    this.visible = false;

    this.uniformLocations = {};
    this.attribLocations = {};

    this.initShaders();
    this.loadTexture("/api/scene/texture");
  }

  initShaders() {
    const gl = this.gl;
    if (!gl) return;

    const vsSource = `#version 300 es
    layout(location = 0) in vec3 a_position;
    layout(location = 1) in vec3 a_normal;
    layout(location = 2) in vec3 a_color;
    layout(location = 3) in vec2 a_texCoord;

    uniform mat4 u_projection;
    uniform mat4 u_view;
    uniform mat4 u_model;

    out vec3 v_normal;
    out vec3 v_color;
    out vec2 v_texCoord;
    out vec3 v_worldPos;

    void main() {
      v_normal = mat3(u_model) * a_normal;
      v_color = a_color;
      v_texCoord = a_texCoord;
      vec4 worldPos = u_model * vec4(a_position, 1.0);
      v_worldPos = worldPos.xyz;
      gl_Position = u_projection * u_view * worldPos;
    }
    `;

    const fsSource = `#version 300 es
    precision highp float;

    in vec3 v_normal;
    in vec3 v_color;
    in vec2 v_texCoord;
    in vec3 v_worldPos;

    uniform sampler2D u_diffuseMap;
    uniform bool u_useTexture;

    out vec4 fragColor;

    void main() {
      // Directional diffuse shading with key & fill lights
      vec3 lightDir1 = normalize(vec3(0.4, 0.8, 0.5));
      vec3 lightDir2 = normalize(vec3(-0.6, -0.4, -0.5));
      vec3 norm = normalize(v_normal);

      // Double-sided lighting so interior vehicle cabin panels render solidly
      float diff1 = max(abs(dot(norm, lightDir1)), 0.0);
      float diff2 = max(abs(dot(norm, lightDir2)), 0.0) * 0.35;
      float ambient = 0.38;

      float totalLighting = clamp(diff1 + diff2 + ambient, 0.0, 1.25);

      vec4 albedo = vec4(v_color, 1.0);
      if (u_useTexture) {
        vec4 tex = texture(u_diffuseMap, v_texCoord);
        if (tex.a > 0.05) {
          albedo = tex;
        }
      }

      fragColor = vec4(albedo.rgb * totalLighting, 1.0); // 100% Solid Opaque
    }
    `;

    const vs = this.compileShader(gl.VERTEX_SHADER, vsSource);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, fsSource);

    this.program = gl.createProgram();
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error("[SOLID MESH] Program link error:", gl.getProgramInfoLog(this.program));
      return;
    }

    this.uniformLocations = {
      projection: gl.getUniformLocation(this.program, "u_projection"),
      view: gl.getUniformLocation(this.program, "u_view"),
      model: gl.getUniformLocation(this.program, "u_model"),
      diffuseMap: gl.getUniformLocation(this.program, "u_diffuseMap"),
      useTexture: gl.getUniformLocation(this.program, "u_useTexture"),
    };
  }

  compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error("[SOLID MESH] Shader compile error:", gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  async loadTexture(url = "/api/scene/texture") {
    const gl = this.gl;
    if (!gl) return;

    try {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = url;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });

      this.texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.bindTexture(gl.TEXTURE_2D, null);

      this.hasTexture = true;
      console.log(`[SOLID MESH] 4K Texture Atlas loaded successfully (${img.width}x${img.height})`);
    } catch (e) {
      console.warn("[SOLID MESH] Texture atlas load deferred or fallback to vertex colors:", e);
    }
  }

  async loadFromPlyUrl(url = "/api/scene/solid") {
    const gl = this.gl;
    if (!gl) return false;

    try {
      console.log(`[SOLID MESH] Ingesting binary triangle manifold -> ${url}...`);
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status} fetching solid mesh`);

      const buffer = await res.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      const headerText = new TextDecoder().decode(bytes.subarray(0, 2048));

      const vMatch = headerText.match(/element vertex (\d+)/);
      const fMatch = headerText.match(/element face (\d+)/);
      const endHeaderIdx = headerText.indexOf("end_header\n");

      if (!vMatch || !fMatch || endHeaderIdx === -1) {
        throw new Error("Invalid Stanford PLY solid mesh format");
      }

      this.vertexCount = parseInt(vMatch[1], 10);
      const numFaces = parseInt(fMatch[1], 10);
      this.indexCount = numFaces * 3;

      const hasUV = headerText.includes("property float u") && headerText.includes("property float v");
      const vertStride = hasUV ? 35 : 27; // 35 bytes with (u, v), 27 bytes without

      const dataOffset = endHeaderIdx + "end_header\n".length;
      const dataView = new DataView(buffer, dataOffset);

      const positions = new Float32Array(this.vertexCount * 3);
      const normals = new Float32Array(this.vertexCount * 3);
      const colors = new Float32Array(this.vertexCount * 3);
      const uvs = new Float32Array(this.vertexCount * 2);

      let sumX = 0, sumY = 0, sumZ = 0;
      for (let i = 0; i < this.vertexCount; i++) {
        const offset = i * vertStride;
        const x = dataView.getFloat32(offset, true);
        const y = dataView.getFloat32(offset + 4, true);
        const z = dataView.getFloat32(offset + 8, true);
        sumX += x; sumY += y; sumZ += z;
      }

      const cX = sumX / this.vertexCount;
      const cY = sumY / this.vertexCount;
      const cZ = sumZ / this.vertexCount;

      for (let i = 0; i < this.vertexCount; i++) {
        const offset = i * vertStride;
        positions[i * 3]     = dataView.getFloat32(offset, true) - cX;
        positions[i * 3 + 1] = -(dataView.getFloat32(offset + 4, true) - cY);
        positions[i * 3 + 2] = -(dataView.getFloat32(offset + 8, true) - cZ);

        normals[i * 3]     = dataView.getFloat32(offset + 12, true);
        normals[i * 3 + 1] = -dataView.getFloat32(offset + 16, true);
        normals[i * 3 + 2] = -dataView.getFloat32(offset + 20, true);

        colors[i * 3]     = dataView.getUint8(offset + 24) / 255.0;
        colors[i * 3 + 1] = dataView.getUint8(offset + 25) / 255.0;
        colors[i * 3 + 2] = dataView.getUint8(offset + 26) / 255.0;

        if (hasUV) {
          uvs[i * 2]     = dataView.getFloat32(offset + 27, true);
          uvs[i * 2 + 1] = dataView.getFloat32(offset + 31, true);
        } else {
          uvs[i * 2]     = 0.5;
          uvs[i * 2 + 1] = 0.5;
        }
      }

      // Parse triangle element face indices
      const facesOffset = dataOffset + this.vertexCount * vertStride;
      const faceDataView = new DataView(buffer, facesOffset);
      const indices = new Uint32Array(this.indexCount);
      const faceStride = 13;

      for (let i = 0; i < numFaces; i++) {
        const offset = i * faceStride;
        indices[i * 3]     = faceDataView.getInt32(offset + 1, true);
        indices[i * 3 + 1] = faceDataView.getInt32(offset + 5, true);
        indices[i * 3 + 2] = faceDataView.getInt32(offset + 9, true);
      }

      // Setup WebGL2 VAO & Vertex Buffers
      if (this.vao) gl.deleteVertexArray(this.vao);
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);

      // 1. Position Buffer (location = 0)
      this.vertexBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

      // 2. Normal Buffer (location = 1)
      this.normalBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

      // 3. Color Buffer (location = 2)
      this.colorBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);

      // 4. UV Texture Coordinate Buffer (location = 3)
      this.uvBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 0, 0);

      // 5. Index Buffer (gl.ELEMENT_ARRAY_BUFFER)
      this.indexBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

      gl.bindVertexArray(null);

      this.isLoaded = true;
      console.log(`[SOLID MESH] WebGL2 4K Textured Triangle Mesh Initialized: ${this.vertexCount.toLocaleString()} vertices, ${numFaces.toLocaleString()} solid triangles.`);
      return true;
    } catch (e) {
      console.error("[SOLID MESH] Failed to load solid mesh:", e);
      return false;
    }
  }

  render(projectionMatrix, viewMatrix, modelMatrix) {
    if (!this.isLoaded || !this.visible) return;

    const gl = this.gl;
    if (!gl) return;

    // WebGL State for Solid Opaque Triangle Mesh
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.disable(gl.BLEND); // Zero alpha blending - 100% Solid Opaque
    gl.disable(gl.CULL_FACE); // Render double-sided faces

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uniformLocations.projection, false, projectionMatrix);
    gl.uniformMatrix4fv(this.uniformLocations.view, false, viewMatrix);
    gl.uniformMatrix4fv(this.uniformLocations.model, false, modelMatrix || new Float32Array([
      1,0,0,0,
      0,1,0,0,
      0,0,1,0,
      0,0,0,1
    ]));

    if (this.texture && this.hasTexture) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.uniform1i(this.uniformLocations.diffuseMap, 0);
      gl.uniform1i(this.uniformLocations.useTexture, 1);
    } else {
      gl.uniform1i(this.uniformLocations.useTexture, 0);
    }

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);

    // THE CRITICAL HARDWARE TRIANGLE DRAW CALL:
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_INT, 0);

    gl.bindVertexArray(null);
  }
}

if (typeof window !== "undefined") {
  window.SolidMeshRenderer = SolidMeshRenderer;
}
