/**
 * SIBR-Native 4K Instanced Quad Micro-Surfel Engine & Real-Time Gaussian Splatter
 * - Instanced 4-Vertex Camera Quads (Zero gl.POINTS single-pixel sprites)
 * - Exact Projective 2D Covariance Matrix Sigma' = J * W * Sigma * W^T * J^T
 * - Low-Pass +0.3 Anti-Aliasing Filter
 * - 2D Gaussian Conic Quadratic Exponential Falloff with Premultiplied Alpha
 * - WebGL2 Premultiplied Blending (gl.ONE, gl.ONE_MINUS_SRC_ALPHA) & depthWrite: false
 * - Dynamic Subject Centroid Framing & Eye-Level Target Solver
 */

class SurfelViewer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.surfelMesh = null;
    this.instancedGeometry = null;
    this.surfelMaterial = null;
    this.cageLines = null;
    this.cageGeometry = null;
    this.clock = new THREE.Clock();

    // Camera & Spherical Orbit State
    this.targetCenter = new THREE.Vector3(0.0, 0.05, 0.0);
    this.spherical = new THREE.Spherical(3.8, Math.PI / 2.0 - 0.26, 0.35); // 15 deg elevation

    // 360 Continuous Cinematic Orbit
    this.isCinematicOrbit = false;
    this.orbitSpeed = 0.35; // rad/s

    // Mouse Interaction
    this.isDragging = false;
    this.dragButton = 0;
    this.previousMouse = { x: 0, y: 0 };

    // WASD First-Person Movement
    this.moveState = { forward: false, backward: false, left: false, right: false, up: false, down: false };
    this.walkSpeed = 3.5;

    // Shader Quality & Knobs
    this.exposure = 1.0;
    this.dotScale = 1.8;
    this.renderMode = "rgb"; // DEFAULT: Strictly RGB Photorealistic
    this.enableClamping = true;
    this.showCage = true;

    // Solid Forensic Mesh
    this.solidMesh = null;
    this.showSolidMesh = false;

    // Recording & Sorter
    this.mediaRecorder = null;
    this.recordedChunks = [];
    this.isRecording = false;
    this.sorter = typeof RadixDepthSorter !== "undefined" ? new RadixDepthSorter() : null;

    this.init();
  }

  init() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0e17);

    // Illumination for Solid Forensic 3D Mesh
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.75);
    dirLight1.position.set(6, 12, 8);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x90cdf4, 0.4);
    dirLight2.position.set(-6, -4, -8);
    this.scene.add(dirLight2);

    const width = this.canvas.parentElement ? this.canvas.parentElement.clientWidth : 1200;
    const height = this.canvas.parentElement ? this.canvas.parentElement.clientHeight : 650;

    this.camera = new THREE.PerspectiveCamera(54, width / height, 0.05, 100);
    this.updateCameraFromSpherical();

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(new THREE.Color(0x0a0e17), 1.0);

    const gl = this.renderer.getContext();
    if (gl) {
      gl.enable(gl.DEPTH_TEST);
      gl.depthMask(false);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    }

    const grid = new THREE.GridHelper(16, 32, 0x1e293b, 0x0f172a);
    grid.position.y = -1.15;
    this.scene.add(grid);

    this.bindEvents();

    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  updateCameraFromSpherical() {
    this.camera.position.setFromSpherical(this.spherical).add(this.targetCenter);
    this.camera.lookAt(this.targetCenter);
  }

  bindEvents() {
    window.addEventListener("resize", () => this.onResize());
    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1.08 : 0.92;
      this.spherical.radius = Math.max(0.4, Math.min(24.0, this.spherical.radius * zoomFactor));
      this.updateCameraFromSpherical();
    }, { passive: false });

    this.canvas.addEventListener("mousedown", (e) => {
      this.isDragging = true;
      this.dragButton = e.button;
      this.previousMouse = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mouseup", () => {
      this.isDragging = false;
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isDragging) return;

      const dx = e.clientX - this.previousMouse.x;
      const dy = e.clientY - this.previousMouse.y;
      this.previousMouse = { x: e.clientX, y: e.clientY };

      if (this.dragButton === 0 && !e.shiftKey) {
        this.spherical.theta -= dx * 0.0055;
        this.spherical.phi = Math.max(0.04, Math.min(Math.PI / 2.02, this.spherical.phi + dy * 0.0055));
        this.updateCameraFromSpherical();
      } else if (this.dragButton === 2 || this.dragButton === 1 || (this.dragButton === 0 && e.shiftKey)) {
        const panSpeed = 0.003 * this.spherical.radius;
        const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
        const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);

        this.targetCenter.addScaledVector(right, -dx * panSpeed);
        this.targetCenter.addScaledVector(up, dy * panSpeed);
        this.updateCameraFromSpherical();
      }
    });

    window.addEventListener("keydown", (e) => {
      if (["input", "textarea", "select"].includes(document.activeElement.tagName.toLowerCase())) return;
      switch (e.key.toLowerCase()) {
        case "w": case "arrowup": this.moveState.forward = true; break;
        case "s": case "arrowdown": this.moveState.backward = true; break;
        case "a": case "arrowleft": this.moveState.left = true; break;
        case "d": case "arrowright": this.moveState.right = true; break;
        case "q": this.moveState.down = true; break;
        case "e": this.moveState.up = true; break;
        case "r": this.resetView(); break;
      }
    });

    window.addEventListener("keyup", (e) => {
      switch (e.key.toLowerCase()) {
        case "w": case "arrowup": this.moveState.forward = false; break;
        case "s": case "arrowdown": this.moveState.backward = false; break;
        case "a": case "arrowleft": this.moveState.left = false; break;
        case "d": case "arrowright": this.moveState.right = false; break;
        case "q": this.moveState.down = false; break;
        case "e": this.moveState.up = false; break;
      }
    });
  }

  onResize() {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const width = parent.clientWidth;
    const height = parent.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  loadSurfelData(sceneData) {
    this.loadSurfels(sceneData);
  }

  loadSurfels(sceneData) {
    if (this.surfelMesh) {
      this.scene.remove(this.surfelMesh);
      this.surfelMesh.geometry.dispose();
      this.surfelMesh.material.dispose();
      this.surfelMesh = null;
    }

    const count = sceneData.count || (sceneData.positions.length / 3);
    const rawPos = sceneData.positions;
    const rawCols = sceneData.colors || [];
    const rawQuats = sceneData.quaternions || [];
    const rawScales = sceneData.scales || [];

    // 1. Calculate Primary Object Centroid C = (1/|P|) sum mu_i
    let sumX = 0, sumY = 0, sumZ = 0;
    for (let i = 0; i < count; i++) {
      sumX += rawPos[i * 3];
      sumY += rawPos[i * 3 + 1];
      sumZ += rawPos[i * 3 + 2];
    }
    const cX = sumX / count;
    const cY = sumY / count;
    const cZ = sumZ / count;
    this.originalCentroid = new THREE.Vector3(cX, cY, cZ);

    let maxDistSq = 0;
    for (let i = 0; i < count; i++) {
      const dx = rawPos[i * 3] - cX;
      const dy = rawPos[i * 3 + 1] - cY;
      const dz = rawPos[i * 3 + 2] - cZ;
      const dSq = dx * dx + dy * dy + dz * dz;
      if (dSq > maxDistSq) maxDistSq = dSq;
    }
    const boundingRadius = Math.min(Math.sqrt(maxDistSq), 4.5);

    // 2. Build Centered Upright Instanced Buffers
    const instPos = new Float32Array(count * 3);
    const instRot = new Float32Array(count * 4);
    const instScale = new Float32Array(count * 3);
    const instColor = new Float32Array(count * 4);
    const instNormal = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      instPos[i * 3] = rawPos[i * 3] - cX;
      instPos[i * 3 + 1] = -(rawPos[i * 3 + 1] - cY);
      instPos[i * 3 + 2] = -(rawPos[i * 3 + 2] - cZ);

      // Quaternion rotation
      if (rawQuats.length >= (i + 1) * 4) {
        instRot[i * 4] = rawQuats[i * 4];
        instRot[i * 4 + 1] = rawQuats[i * 4 + 1];
        instRot[i * 4 + 2] = rawQuats[i * 4 + 2];
        instRot[i * 4 + 3] = rawQuats[i * 4 + 3];

        const qw = rawQuats[i * 4], qx = rawQuats[i * 4 + 1], qy = rawQuats[i * 4 + 2], qz = rawQuats[i * 4 + 3];
        const nx = 2.0 * (qx * qz + qw * qy);
        const ny = -(2.0 * (qy * qz - qw * qx));
        const nz = -(1.0 - 2.0 * (qx * qx + qy * qy));
        instNormal[i * 3] = nx * 0.5 + 0.5;
        instNormal[i * 3 + 1] = ny * 0.5 + 0.5;
        instNormal[i * 3 + 2] = nz * 0.5 + 0.5;
      } else {
        instRot[i * 4] = 1.0; instRot[i * 4 + 1] = 0.0; instRot[i * 4 + 2] = 0.0; instRot[i * 4 + 3] = 0.0;
        instNormal[i * 3] = 0.5; instNormal[i * 3 + 1] = 0.8; instNormal[i * 3 + 2] = 0.5;
      }

      // Local 3D Scale (sx, sy, sz)
      if (rawScales.length >= (i + 1) * 3) {
        instScale[i * 3] = rawScales[i * 3];
        instScale[i * 3 + 1] = rawScales[i * 3 + 1];
        instScale[i * 3 + 2] = rawScales[i * 3 + 2];
      } else {
        instScale[i * 3] = 0.012;
        instScale[i * 3 + 1] = 0.012;
        instScale[i * 3 + 2] = 0.001;
      }

      // RGB Color + Opacity
      if (rawCols.length > 0) {
        instColor[i * 4] = rawCols[i * 3];
        instColor[i * 4 + 1] = rawCols[i * 3 + 1];
        instColor[i * 4 + 2] = rawCols[i * 3 + 2];
        instColor[i * 4 + 3] = 0.95;
      } else {
        instColor[i * 4] = 0.85; instColor[i * 4 + 1] = 0.85; instColor[i * 4 + 2] = 0.85; instColor[i * 4 + 3] = 0.95;
      }
    }

    // 3. Create Instanced 4-Vertex Quad Mesh Geometry
    const quadVertices = new Float32Array([
      -1.0, -1.0, 0.0,
       1.0, -1.0, 0.0,
      -1.0,  1.0, 0.0,
       1.0,  1.0, 0.0,
    ]);
    const quadIndices = new Uint16Array([0, 1, 2, 2, 1, 3]);

    const geo = new THREE.InstancedBufferGeometry();
    geo.instanceCount = count;
    geo.setAttribute("position", new THREE.BufferAttribute(quadVertices, 3));
    geo.setIndex(new THREE.BufferAttribute(quadIndices, 1));

    geo.setAttribute("a_position", new THREE.InstancedBufferAttribute(instPos, 3));
    geo.setAttribute("a_rotation", new THREE.InstancedBufferAttribute(instRot, 4));
    geo.setAttribute("a_scale", new THREE.InstancedBufferAttribute(instScale, 3));
    geo.setAttribute("a_color", new THREE.InstancedBufferAttribute(instColor, 4));
    geo.setAttribute("a_normalColor", new THREE.InstancedBufferAttribute(instNormal, 3));

    this.instancedGeometry = geo;

    // 4. SIBR-Native Projective Covariance Quad Shader
    const width = this.canvas.parentElement ? this.canvas.parentElement.clientWidth : 1200;
    const height = this.canvas.parentElement ? this.canvas.parentElement.clientHeight : 650;

    this.surfelMaterial = new THREE.ShaderMaterial({
      vertexShader: `
        attribute vec3 a_position;
        attribute vec4 a_rotation;
        attribute vec3 a_scale;
        attribute vec4 a_color;
        attribute vec3 a_normalColor;

        uniform vec2 u_focal;
        uniform vec2 u_viewport;
        uniform float u_dotScale;
        uniform float u_exposure;
        uniform int u_renderMode;

        varying vec2 v_quad_offset;
        varying vec3 v_conic;
        varying vec4 v_color;
        varying vec3 v_normalColor;
        varying float v_depth;

        mat3 quaternion_to_rotation(vec4 q) {
          float r = q.x, x = q.y, y = q.z, z = q.w;
          return mat3(
            1.0 - 2.0*(y*y + z*z), 2.0*(x*y - r*z), 2.0*(x*z + r*y),
            2.0*(x*y + r*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - r*x),
            2.0*(x*z - r*y), 2.0*(y*z + r*x), 1.0 - 2.0*(x*x + y*y)
          );
        }

        void main() {
          // 1. Transform world position to view/camera space
          vec4 cam_pos = modelViewMatrix * vec4(a_position, 1.0);
          float depth = -cam_pos.z;
          v_depth = depth;

          if (depth <= 0.1 || depth >= 150.0) {
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
          }

          // 2. Reconstruct 3D Covariance Matrix Sigma = R * S * S^T * R^T
          mat3 R = quaternion_to_rotation(a_rotation);
          vec3 eff_scale = a_scale * u_dotScale;
          mat3 S = mat3(
            eff_scale.x, 0.0, 0.0,
            0.0, eff_scale.y, 0.0,
            0.0, 0.0, eff_scale.z
          );
          mat3 M = R * S;
          mat3 Vrk = M * transpose(M);

          // 3. Project 3D Covariance to 2D Screen Space: Sigma' = J * W * Sigma * W^T * J^T
          mat3 W = mat3(modelViewMatrix);
          float z_inv = 1.0 / depth;
          float z_inv2 = z_inv * z_inv;

          // Correct GLSL column-major Jacobian constructor
          mat3 J = mat3(
            u_focal.x * z_inv, 0.0, 0.0,
            0.0, u_focal.y * z_inv, 0.0,
            -(u_focal.x * cam_pos.x) * z_inv2, -(u_focal.y * cam_pos.y) * z_inv2, 0.0
          );
          mat3 T = J * W;
          mat3 cov2d = T * Vrk * transpose(T);

          // Add 0.35 anti-aliasing filter
          cov2d[0][0] += 0.35;
          cov2d[1][1] += 0.35;

          // 4. Invert 2D Covariance Matrix to get Conic parameters (a, b, c)
          float det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[1][0];
          if (det <= 0.000001) {
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
          }
          float det_inv = 1.0 / det;
          v_conic = vec3(cov2d[1][1] * det_inv, -cov2d[0][1] * det_inv, cov2d[0][0] * det_inv);

          // 5. Compute Screen Radius: Enforce MINIMUM 4.0 pixel radius to eliminate sub-pixel holes
          float mid = 0.5 * (cov2d[0][0] + cov2d[1][1]);
          float lambda = mid + sqrt(max(0.1, mid * mid - det));
          float radius = max(ceil(3.2 * sqrt(max(0.1, lambda))), 4.0);
          radius = min(radius, 64.0); // Clamp to prevent near-camera mosaic explosions

          // 6. Project Quad Vertex in Screen Pixels
          vec2 screen_offset = position.xy * radius;
          vec4 proj_pos = projectionMatrix * cam_pos;
          proj_pos.xy += (screen_offset / u_viewport) * proj_pos.w * 2.0;

          v_quad_offset = screen_offset;
          v_color = vec4(a_color.rgb * u_exposure, a_color.a);
          v_normalColor = a_normalColor;
          gl_Position = proj_pos;
        }
      `,
      fragmentShader: `
        varying vec2 v_quad_offset;
        varying vec3 v_conic;
        varying vec4 v_color;
        varying vec3 v_normalColor;
        varying float v_depth;
        uniform int u_renderMode;

        void main() {
          // Exact 2D Gaussian Conic Quadratic Falloff
          float power = -0.5 * (v_conic.x * v_quad_offset.x * v_quad_offset.x + 
                                2.0 * v_conic.y * v_quad_offset.x * v_quad_offset.y + 
                                v_conic.z * v_quad_offset.y * v_quad_offset.y);

          // Soft power cutoff to -5.0 so Gaussian skirts overlap completely
          if (power < -5.0) discard;

          // Continuous exponential falloff
          float alpha = v_color.a * exp(power);
          // Lower discard threshold to 0.002 to prevent edge fringing
          if (alpha < 0.002) discard;

          vec3 baseColor = v_color.rgb;
          if (u_renderMode == 1) {
            baseColor = v_normalColor;
          } else if (u_renderMode == 2) {
            float normDepth = clamp(v_depth / 8.0, 0.0, 1.0);
            baseColor = vec3(normDepth, 1.0 - normDepth, 0.5 + 0.5 * sin(normDepth * 6.28));
          } else if (u_renderMode == 3) {
            float ringDist = length(v_quad_offset);
            alpha = smoothstep(2.0, 0.0, abs(ringDist - 12.0));
            baseColor = vec3(0.0, 0.94, 1.0);
          }

          // PREMULTIPLIED ALPHA: Melts adjacent Gaussian ellipses with zero dark outlines
          gl_FragColor = vec4(baseColor * alpha, alpha);
        }
      `,
      uniforms: {
        u_focal: { value: new THREE.Vector2(800.0, 800.0) },
        u_viewport: { value: new THREE.Vector2(width, height) },
        u_dotScale: { value: this.dotScale },
        u_exposure: { value: this.exposure },
        u_renderMode: { value: 0 }, // 0: RGB Photorealistic
      },
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      depthWrite: false,
      depthTest: true,
      transparent: true,
      side: THREE.DoubleSide,
    });

    this.surfelMesh = new THREE.Mesh(geo, this.surfelMaterial);
    this.surfelMesh.frustumCulled = false;
    this.scene.add(this.surfelMesh);

    // Object-Centric Orbit Camera Initialization (elevation 15 deg, distance = 3.9m)
    this.targetCenter.set(0.0, 0.05, 0.0);
    const initDist = 3.9;
    this.spherical.set(initDist, Math.PI / 2.0 - 0.26, 0.35); // 15 deg elevation
    this.updateCameraFromSpherical();

    const countElem = document.getElementById("hud-surfel-count");
    if (countElem) countElem.textContent = `${count.toLocaleString()} Micro-Surfels`;
  }

  loadBoundaryCage(cageData) {
    if (this.cageLines) this.scene.remove(this.cageLines);
    this.cageGeometry = cageData;
    if (!cageData || !cageData.vertices || cageData.vertices.length === 0) return;

    const vertices = cageData.vertices;
    const faces = cageData.faces;
    const c = this.originalCentroid || new THREE.Vector3(0, 0, 0);

    const linePositions = [];
    for (const face of faces) {
      for (let i = 0; i < face.length; i++) {
        const v1 = vertices[face[i]];
        const v2 = vertices[face[(i + 1) % face.length]];
        linePositions.push(v1[0] - c.x, -(v1[1] - c.y), -(v1[2] - c.z));
        linePositions.push(v2[0] - c.x, -(v2[1] - c.y), -(v2[2] - c.z));
      }
    }

    const cageGeo = new THREE.BufferGeometry();
    cageGeo.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    const cageMat = new THREE.LineBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.35 });
    this.cageLines = new THREE.LineSegments(cageGeo, cageMat);
    this.cageLines.visible = this.showCage;
    this.scene.add(this.cageLines);
  }

  async loadSolidForensicMesh(url = "/api/scene/solid") {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error("Could not fetch solid mesh");
      const buffer = await res.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      const headerStr = new TextDecoder().decode(bytes.subarray(0, 2048));
      
      const vMatch = headerStr.match(/element vertex (\d+)/);
      const fMatch = headerStr.match(/element face (\d+)/);
      const endHeaderIdx = headerStr.indexOf("end_header\n");

      if (!vMatch || !fMatch || endHeaderIdx === -1) {
        throw new Error("Invalid PLY solid mesh format");
      }

      const numVerts = parseInt(vMatch[1], 10);
      const numFaces = parseInt(fMatch[1], 10);
      const dataOffset = endHeaderIdx + "end_header\n".length;
      const dataView = new DataView(buffer, dataOffset);

      const positions = new Float32Array(numVerts * 3);
      const normals = new Float32Array(numVerts * 3);
      const colors = new Float32Array(numVerts * 3);

      const vertStride = 27; // 6 floats (24B) + 3 uchar (3B)
      let sumX = 0, sumY = 0, sumZ = 0;

      for (let i = 0; i < numVerts; i++) {
        const offset = i * vertStride;
        const x = dataView.getFloat32(offset, true);
        const y = dataView.getFloat32(offset + 4, true);
        const z = dataView.getFloat32(offset + 8, true);
        sumX += x; sumY += y; sumZ += z;
      }
      const cX = sumX / numVerts;
      const cY = sumY / numVerts;
      const cZ = sumZ / numVerts;
      this.originalCentroid = new THREE.Vector3(cX, cY, cZ);

      for (let i = 0; i < numVerts; i++) {
        const offset = i * vertStride;
        positions[i * 3] = dataView.getFloat32(offset, true) - cX;
        positions[i * 3 + 1] = -(dataView.getFloat32(offset + 4, true) - cY);
        positions[i * 3 + 2] = -(dataView.getFloat32(offset + 8, true) - cZ);

        normals[i * 3] = dataView.getFloat32(offset + 12, true);
        normals[i * 3 + 1] = -dataView.getFloat32(offset + 16, true);
        normals[i * 3 + 2] = -dataView.getFloat32(offset + 20, true);

        colors[i * 3] = dataView.getUint8(offset + 24) / 255.0;
        colors[i * 3 + 1] = dataView.getUint8(offset + 25) / 255.0;
        colors[i * 3 + 2] = dataView.getUint8(offset + 26) / 255.0;
      }

      const facesOffset = dataOffset + numVerts * vertStride;
      const faceDataView = new DataView(buffer, facesOffset);
      const indices = new Uint32Array(numFaces * 3);
      const faceStride = 13; // 1 uchar count + 3 int32 (12B)

      for (let i = 0; i < numFaces; i++) {
        const offset = i * faceStride;
        indices[i * 3] = faceDataView.getInt32(offset + 1, true);
        indices[i * 3 + 1] = faceDataView.getInt32(offset + 5, true);
        indices[i * 3 + 2] = faceDataView.getInt32(offset + 9, true);
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geo.setIndex(new THREE.BufferAttribute(indices, 1));
      geo.computeVertexNormals();

      const mat = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.35,
        metalness: 0.15,
        side: THREE.DoubleSide,
      });

      if (this.solidMesh) {
        this.scene.remove(this.solidMesh);
        this.solidMesh.geometry.dispose();
        this.solidMesh.material.dispose();
      }

      this.solidMesh = new THREE.Mesh(geo, mat);
      this.solidMesh.frustumCulled = false;
      this.scene.add(this.solidMesh);

      const gl = this.renderer.getContext();
      if (gl) gl.depthMask(true);

      const countElem = document.getElementById("hud-surfel-count");
      if (countElem) countElem.textContent = `${numFaces.toLocaleString()} Solid Triangles`;

      console.log(`[SOLID MESH] Loaded ${numVerts.toLocaleString()} vertices, ${numFaces.toLocaleString()} solid triangles.`);
      return true;
    } catch (e) {
      console.warn("[SOLID MESH] Failed to load solid mesh in viewer:", e);
      return false;
    }
  }

  resetView() {
    this.targetCenter.set(0.0, 0.05, 0.0);
    this.spherical.set(3.8, Math.PI / 2.0 - 0.26, 0.35);
    this.updateCameraFromSpherical();
  }

  async setRenderMode(mode) {
    this.renderMode = mode;
    if (mode === "solid") {
      this.showSolidMesh = true;
      if (!this.solidMesh) {
        await this.loadSolidForensicMesh("/api/scene/solid");
      } else {
        this.solidMesh.visible = true;
      }
      if (this.surfelMesh) this.surfelMesh.visible = false;
    } else {
      this.showSolidMesh = false;
      if (this.solidMesh) this.solidMesh.visible = false;
      if (this.surfelMesh) {
        this.surfelMesh.visible = true;
        const modeMap = { rgb: 0, normals: 1, depth: 2, wireframe: 3 };
        if (this.surfelMaterial) this.surfelMaterial.uniforms.u_renderMode.value = modeMap[mode] ?? 0;
      }
    }
  }

  setExposure(val) {
    this.exposure = parseFloat(val);
    if (this.surfelMaterial) this.surfelMaterial.uniforms.u_exposure.value = this.exposure;
  }

  setDotScale(val) {
    this.dotScale = parseFloat(val);
    if (this.surfelMaterial) this.surfelMaterial.uniforms.u_dotScale.value = this.dotScale;
  }

  setFOV(val) {
    this.camera.fov = parseFloat(val);
    this.camera.updateProjectionMatrix();
    if (this.surfelMaterial) {
      const focalY = (this.canvas.height / 2.0) / Math.tan((this.camera.fov * Math.PI / 180.0) / 2.0);
      this.surfelMaterial.uniforms.u_focal.value.set(focalY, focalY);
    }
  }

  toggleCinematicOrbit() {
    this.isCinematicOrbit = !this.isCinematicOrbit;
    return this.isCinematicOrbit;
  }

  takeSnapshot() {
    const dataUrl = this.canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `microsurfel_snapshot_4k_${Date.now()}.png`;
    a.click();
  }

  animate() {
    requestAnimationFrame(this.animate);
    const delta = this.clock.getDelta();

    if (this.isCinematicOrbit) {
      this.spherical.theta += this.orbitSpeed * delta;
      this.updateCameraFromSpherical();
    }

    if (this.moveState.forward || this.moveState.backward || this.moveState.left || this.moveState.right || this.moveState.up || this.moveState.down) {
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
      forward.y = 0;
      forward.normalize();

      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
      right.y = 0;
      right.normalize();

      const move = new THREE.Vector3();
      if (this.moveState.forward) move.add(forward);
      if (this.moveState.backward) move.sub(forward);
      if (this.moveState.right) move.add(right);
      if (this.moveState.left) move.sub(right);
      if (this.moveState.up) move.y += 1.0;
      if (this.moveState.down) move.y -= 1.0;

      if (move.lengthSq() > 0) {
        move.normalize().multiplyScalar(this.walkSpeed * delta);
        this.targetCenter.add(move);
        this.updateCameraFromSpherical();
      }
    }

    this.renderer.render(this.scene, this.camera);
  }
}

let surfelViewer = null;

function initSurfelViewer(canvasId = "viewer-canvas") {
  if (!surfelViewer) surfelViewer = new SurfelViewer(canvasId);
  return surfelViewer;
}
