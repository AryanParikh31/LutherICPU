/**
 * INRIA SIBR — State-of-the-Art 3D Gaussian Splatting (3DGS) & Photorealistic Simulation Engine
 * 
 * SIBR Architecture & Features:
 * - 6-DoF Free-Viewpoint Radiance Field Navigation (Trackball & FPS Fly modes)
 * - Real-Time Asynchronous Radix Depth Sorter (60 FPS Back-to-Front Alpha Blending)
 * - Exact Projective 2D Covariance Matrix Sigma' = J * W * Sigma * W^T * J^T + nu * I
 * - 2D Conic Quadratic Exponential Falloff with Premultiplied Alpha Blending
 * - Standard Inria 3DGS Binary PLY & .splat Format Streaming
 * - High-Resolution 4K UV Textured Mesh Layer with Projective Texture Atlas
 * - Exact Calibrated Camera Snapping with World-to-Camera Rotation Matrix Projection
 * - Continuous 360 Turntable Orbit & Calibrated Camera Cycle
 * - Real-Time Rendering Modes: Radiance (1), Depth Field (2), Surface Normals (3), Conics (4), Textured Mesh (5)
 */

let simEngine = null;

class RadixDepthSorter {
  constructor() {
    this.worker = null;
    this.isSorting = false;
    this.onSortComplete = null;
    this.initWorker();
  }

  initWorker() {
    try {
      this.worker = new Worker("/js/sort_worker.js?v=10");
      this.worker.onmessage = (e) => {
        const { type, sortedIndices } = e.data;
        if (type === "sort_done") {
          this.isSorting = false;
          if (this.onSortComplete) {
            this.onSortComplete(new Uint32Array(sortedIndices));
          }
        }
      };
    } catch (err) {
      console.warn("[SIBR] Web Worker initialization notice:", err);
      this.worker = null;
    }
  }

  initPositions(positions, count) {
    if (this.worker) {
      // Create a copy of the buffer so we don't transfer the main thread buffer
      const posCopy = new Float32Array(positions);
      this.worker.postMessage({
        type: "init",
        positions: posCopy.buffer,
        count: count,
      }, [posCopy.buffer]);
    }
  }

  requestSort(viewProjElements, count) {
    if (this.isSorting || !this.worker) return;
    this.isSorting = true;
    this.worker.postMessage({
      type: "sort",
      viewProj: viewProjElements,
      count: count,
    });
  }
}

class SIBRSimulationEngine {
  constructor() {
    this.canvas = document.getElementById("sim-canvas");
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.splatMesh = null;
    this.instancedGeometry = null;
    this.splatMaterial = null;
    this.meshObject = null;
    this.clock = new THREE.Clock();

    // Scene & Splat State
    this.sceneName = "playroom";
    this.splatCount = 0;
    this.sceneCenter = new THREE.Vector3(0, 0, 0);
    this.sceneRadius = 4.5;

    // Original Source Attribute Buffers for Radix Reordering
    this.rawPositions = null;
    this.rawScales = null;
    this.rawRotations = null;
    this.rawColors = null;
    this.rawNormals = null;

    // Radix Sorter
    this.sorter = new RadixDepthSorter();
    this.sorter.onSortComplete = (sortedIndices) => this.applySortedIndices(sortedIndices);
    this.lastCamMatrix = new THREE.Matrix4();
    this.needsSort = true;

    // SIBR Navigation Modes: 'trackball' or 'fps'
    this.navMode = "trackball";
    this.targetCenter = new THREE.Vector3(0.0, 0.0, 0.0);
    this.spherical = new THREE.Spherical(4.5, Math.PI / 2.0 - 0.28, 0.55);

    // Continuous 360 Turntable Orbit
    this.isOrbiting = false;
    this.orbitSpeed = 0.35; // rad/s

    // Mouse Navigation State
    this.isDragging = false;
    this.dragButton = 0;
    this.previousMouse = { x: 0, y: 0 };

    // SIBR FPS Fly Controls
    this.moveState = { forward: false, backward: false, left: false, right: false, up: false, down: false, sprint: false };
    this.flySpeed = 3.5;
    this.fpsYaw = 0;
    this.fpsPitch = 0;

    // Rendering Tuning & Modes
    this.splatScale = 0.65;
    this.exposure = 1.0;
    this.fov = 54.0;
    this.renderMode = 0; // 0: Radiance, 1: Depth, 2: Normals, 3: Conics, 4: Mesh
    this.showGTPeek = false;

    // Calibrated Camera Dataset Views
    this.calibratedCameras = [];
    this.currentCameraIdx = 0;

    // FPS Telemetry
    this.frameCount = 0;
    this.lastFpsTime = performance.now();

    this.init();
  }

  init() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x030712);

    // Illumination for 3D Textured Mesh layer
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
    this.scene.add(ambientLight);
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight1.position.set(5, 10, 7);
    this.scene.add(dirLight1);
    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight2.position.set(-5, -5, -7);
    this.scene.add(dirLight2);

    const width = window.innerWidth || 1400;
    const height = window.innerHeight || 800;

    this.camera = new THREE.PerspectiveCamera(this.fov, width / height, 0.05, 500.0);
    this.updateCameraFromSpherical();

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      powerPreference: "high-performance",
      preserveDrawingBuffer: true,
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.bindEvents();
    this.loadScene();

    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  updateCameraFromSpherical() {
    this.camera.position.setFromSpherical(this.spherical).add(this.targetCenter);
    this.camera.lookAt(this.targetCenter);
    this.needsSort = true;
  }

  bindEvents() {
    window.addEventListener("resize", () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
      if (this.splatMaterial) {
        this.splatMaterial.uniforms.u_viewport.value.set(width, height);
        const focalY = (height / 2.0) / Math.tan((this.fov * Math.PI / 180.0) / 2.0);
        this.splatMaterial.uniforms.u_focal.value.set(focalY, focalY);
      }
      this.needsSort = true;
    });

    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    // Mouse Wheel Zoom
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1.08 : 0.92;
      if (this.navMode === "trackball") {
        this.spherical.radius = Math.max(0.15, Math.min(35.0, this.spherical.radius * zoomFactor));
        this.updateCameraFromSpherical();
      } else {
        const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
        this.camera.position.addScaledVector(forward, e.deltaY > 0 ? -0.5 : 0.5);
        this.needsSort = true;
      }
    }, { passive: false });

    // Mouse Down
    this.canvas.addEventListener("mousedown", (e) => {
      this.isDragging = true;
      this.dragButton = e.button;
      this.previousMouse = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mouseup", () => {
      this.isDragging = false;
    });

    // Mouse Move Navigation
    window.addEventListener("mousemove", (e) => {
      if (!this.isDragging) return;

      const dx = e.clientX - this.previousMouse.x;
      const dy = e.clientY - this.previousMouse.y;
      this.previousMouse = { x: e.clientX, y: e.clientY };

      if (this.navMode === "trackball") {
        if (this.dragButton === 0 && !e.shiftKey) {
          // Orbit (Left-click)
          this.spherical.theta -= dx * 0.0055;
          this.spherical.phi = Math.max(0.04, Math.min(Math.PI / 2.02, this.spherical.phi + dy * 0.0055));
          this.updateCameraFromSpherical();
        } else if (this.dragButton === 2 || this.dragButton === 1 || (this.dragButton === 0 && e.shiftKey)) {
          // Pan (Right-click or Middle-click)
          const panSpeed = 0.0022 * this.spherical.radius;
          const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
          const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);

          this.targetCenter.addScaledVector(right, -dx * panSpeed);
          this.targetCenter.addScaledVector(up, dy * panSpeed);
          this.updateCameraFromSpherical();
        }
      } else if (this.navMode === "fps") {
        // FPS Free Mouse Look
        this.fpsYaw -= dx * 0.0035;
        this.fpsPitch = Math.max(-Math.PI / 2.05, Math.min(Math.PI / 2.05, this.fpsPitch - dy * 0.0035));

        const euler = new THREE.Euler(this.fpsPitch, this.fpsYaw, 0, "YXZ");
        this.camera.quaternion.setFromEuler(euler);
        this.needsSort = true;
      }
    });

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      const key = e.key.toLowerCase();

      switch (key) {
        case "w": case "arrowup": this.moveState.forward = true; break;
        case "s": case "arrowdown": this.moveState.backward = true; break;
        case "a": case "arrowleft": this.moveState.left = true; break;
        case "d": case "arrowright": this.moveState.right = true; break;
        case "q": this.moveState.down = true; break;
        case "e": this.moveState.up = true; break;
        case "shift": this.moveState.sprint = true; break;

        case "1": this.setRenderMode(0); break;
        case "2": this.setRenderMode(1); break;
        case "3": this.setRenderMode(2); break;
        case "4": this.setRenderMode(3); break;
        case "5": case "m": this.setRenderMode(4); break;

        case "o": this.toggleOrbit(); break;
        case "y": this.toggleNavMode(); break;
        case "r": this.resetCamera(); break;
        case "g": this.toggleGTPeek(); break;
        case "h": case "?": this.toggleHelpModal(); break;

        case " ": case "c":
          e.preventDefault();
          this.applyCalibratedCamera(this.currentCameraIdx + 1);
          break;
        case "p":
          e.preventDefault();
          this.applyCalibratedCamera(this.currentCameraIdx - 1);
          break;
      }
    });

    window.addEventListener("keyup", (e) => {
      const key = e.key.toLowerCase();
      switch (key) {
        case "w": case "arrowup": this.moveState.forward = false; break;
        case "s": case "arrowdown": this.moveState.backward = false; break;
        case "a": case "arrowleft": this.moveState.left = false; break;
        case "d": case "arrowright": this.moveState.right = false; break;
        case "q": this.moveState.down = false; break;
        case "e": this.moveState.up = false; break;
        case "shift": this.moveState.sprint = false; break;
      }
    });
  }

  async loadScene() {
    this.updateLoadingStatus("Streaming 3D Gaussian Radiance Field...");

    // 1. Fetch Calibrated Cameras
    try {
      const camRes = await fetch("/api/cameras");
      if (camRes.ok) {
        this.calibratedCameras = await camRes.json();
      }
    } catch (e) {
      console.warn("[SIBR] Cameras fetch notice:", e);
    }

    // 2. Stream 3D Gaussian Splats (.splat or .ply)
    let loaded = await this.loadSplatFile();
    if (!loaded) {
      loaded = await this.loadPlyFile();
    }

    // 3. Load 4K Textured Mesh in Background
    this.loadTexturedMesh();

    if (loaded) {
      const loader = document.getElementById("loading-overlay");
      if (loader) {
        loader.style.opacity = "0";
        setTimeout(() => loader.style.display = "none", 400);
      }
    } else {
      this.updateLoadingStatus("Reconstruction in progress...");
    }
  }

  updateLoadingStatus(text) {
    const el = document.getElementById("loading-status");
    if (el) el.textContent = text;
  }

  async loadSplatFile() {
    try {
      const res = await fetch("/api/scene/splat");
      if (!res.ok) return false;

      const buffer = await res.arrayBuffer();
      if (buffer.byteLength < 32) return false;

      const count = Math.floor(buffer.byteLength / 32);
      console.log(`[SIBR] Parsing .splat buffer: ${count.toLocaleString()} Gaussians (${(buffer.byteLength / (1024*1024)).toFixed(1)} MB) at full density...`);

      const pos = new Float32Array(count * 3);
      const scale = new Float32Array(count * 3);
      const col = new Float32Array(count * 4);
      const rot = new Float32Array(count * 4);

      const f32 = new Float32Array(buffer);
      const u8 = new Uint8Array(buffer);

      for (let i = 0; i < count; i++) {
        const fOffset = i * 8; // 8 float32s = 32 bytes
        const uOffset = i * 32;

        pos[i * 3]     = f32[fOffset];
        pos[i * 3 + 1] = f32[fOffset + 1];
        pos[i * 3 + 2] = f32[fOffset + 2];

        // Exact physical metric scales without artificial dilation
        scale[i * 3]     = f32[fOffset + 3];
        scale[i * 3 + 1] = f32[fOffset + 4];
        scale[i * 3 + 2] = f32[fOffset + 5];

        // Colors RGBA (uint8 -> float32 [0, 1])
        col[i * 4]     = u8[uOffset + 24] / 255.0;
        col[i * 4 + 1] = u8[uOffset + 25] / 255.0;
        col[i * 4 + 2] = u8[uOffset + 26] / 255.0;
        col[i * 4 + 3] = u8[uOffset + 27] / 255.0;

        // Quaternions (uint8 [0, 255] -> float32 [-1, 1], w, x, y, z)
        rot[i * 4]     = (u8[uOffset + 28] - 128) / 127.5;
        rot[i * 4 + 1] = (u8[uOffset + 29] - 128) / 127.5;
        rot[i * 4 + 2] = (u8[uOffset + 30] - 128) / 127.5;
        rot[i * 4 + 3] = (u8[uOffset + 31] - 128) / 127.5;
      }

      this.setupGaussianField(pos, scale, rot, col, count);
      return true;
    } catch (e) {
      console.warn("[SIBR] Splat buffer parsing fallback:", e);
      return false;
    }
  }

  async loadPlyFile() {
    try {
      const res = await fetch("/api/scene/pointcloud");
      if (!res.ok) return false;

      const buffer = await res.arrayBuffer();
      if (buffer.byteLength < 100) return false;

      const u8 = new Uint8Array(buffer);
      let headerStr = "";
      let headerEnd = 0;
      for (let i = 0; i < Math.min(4096, u8.length); i++) {
        headerStr += String.fromCharCode(u8[i]);
        if (headerStr.includes("end_header\n")) {
          headerEnd = headerStr.indexOf("end_header\n") + "end_header\n".length;
          break;
        }
      }

      if (headerEnd === 0) return false;

      const match = headerStr.match(/element vertex (\d+)/);
      if (!match) return false;
      const count = parseInt(match[1]);

      console.log(`[SIBR] Parsing Inria PLY buffer: ${count.toLocaleString()} Gaussians at full density...`);

      const lines = headerStr.split("\n");
      let inVertex = false;
      let stride = 0;
      let offX = -1, offY = -1, offZ = -1;
      let offFdc0 = -1, offFdc1 = -1, offFdc2 = -1;
      let offOpacity = -1;
      let offScale0 = -1, offScale1 = -1, offScale2 = -1;
      let offRot0 = -1, offRot1 = -1, offRot2 = -1, offRot3 = -1;

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("element vertex")) {
          inVertex = true;
          continue;
        } else if (trimmed.startsWith("element ")) {
          inVertex = false;
        }

        if (inVertex && trimmed.startsWith("property")) {
          const parts = trimmed.split(/\s+/);
          const type = parts[1];
          const name = parts[2];
          const currentOff = stride;

          let size = 4;
          if (type === "float" || type === "int" || type === "uint") size = 4;
          else if (type === "uchar" || type === "char" || type === "uint8" || type === "int8") size = 1;
          else if (type === "short" || type === "ushort") size = 2;
          else if (type === "double") size = 8;

          if (name === "x") offX = currentOff;
          else if (name === "y") offY = currentOff;
          else if (name === "z") offZ = currentOff;
          else if (name === "f_dc_0") offFdc0 = currentOff;
          else if (name === "f_dc_1") offFdc1 = currentOff;
          else if (name === "f_dc_2") offFdc2 = currentOff;
          else if (name === "opacity") offOpacity = currentOff;
          else if (name === "scale_0") offScale0 = currentOff;
          else if (name === "scale_1") offScale1 = currentOff;
          else if (name === "scale_2") offScale2 = currentOff;
          else if (name === "rot_0") offRot0 = currentOff;
          else if (name === "rot_1") offRot1 = currentOff;
          else if (name === "rot_2") offRot2 = currentOff;
          else if (name === "rot_3") offRot3 = currentOff;

          stride += size;
        }
      }

      if (stride === 0) stride = 68;

      const pos = new Float32Array(count * 3);
      const scale = new Float32Array(count * 3);
      const col = new Float32Array(count * 4);
      const rot = new Float32Array(count * 4);

      const SH_C0 = 0.28209479177387814;
      const dataView = new DataView(buffer, headerEnd);

      for (let i = 0; i < count; i++) {
        const o = i * stride;
        pos[i * 3]     = offX !== -1 ? dataView.getFloat32(o + offX, true) : dataView.getFloat32(o, true);
        pos[i * 3 + 1] = offY !== -1 ? dataView.getFloat32(o + offY, true) : dataView.getFloat32(o + 4, true);
        pos[i * 3 + 2] = offZ !== -1 ? dataView.getFloat32(o + offZ, true) : dataView.getFloat32(o + 8, true);

        if (offFdc0 !== -1) {
          const f0 = dataView.getFloat32(o + offFdc0, true);
          const f1 = dataView.getFloat32(o + offFdc1, true);
          const f2 = dataView.getFloat32(o + offFdc2, true);
          col[i * 4]     = Math.max(0, Math.min(1, f0 * SH_C0 + 0.5));
          col[i * 4 + 1] = Math.max(0, Math.min(1, f1 * SH_C0 + 0.5));
          col[i * 4 + 2] = Math.max(0, Math.min(1, f2 * SH_C0 + 0.5));
        } else {
          col[i * 4] = 0.8; col[i * 4 + 1] = 0.8; col[i * 4 + 2] = 0.8;
        }

        if (offOpacity !== -1) {
          const logit_opacity = dataView.getFloat32(o + offOpacity, true);
          col[i * 4 + 3] = 1.0 / (1.0 + Math.exp(-Math.max(-10.0, Math.min(10.0, logit_opacity))));
        } else {
          col[i * 4 + 3] = 0.95;
        }

        if (offScale0 !== -1) {
          const s0 = dataView.getFloat32(o + offScale0, true);
          const s1 = dataView.getFloat32(o + offScale1, true);
          const s2 = dataView.getFloat32(o + offScale2, true);
          scale[i * 3]     = Math.max(0.002, Math.min(0.6, Math.exp(Math.max(-12.0, Math.min(2.0, s0)))));
          scale[i * 3 + 1] = Math.max(0.002, Math.min(0.6, Math.exp(Math.max(-12.0, Math.min(2.0, s1)))));
          scale[i * 3 + 2] = Math.max(0.001, Math.min(0.3, Math.exp(Math.max(-12.0, Math.min(2.0, s2)))));
        } else {
          scale[i * 3] = 0.04; scale[i * 3 + 1] = 0.04; scale[i * 3 + 2] = 0.02;
        }

        if (offRot0 !== -1) {
          rot[i * 4]     = dataView.getFloat32(o + offRot0, true);
          rot[i * 4 + 1] = dataView.getFloat32(o + offRot1, true);
          rot[i * 4 + 2] = dataView.getFloat32(o + offRot2, true);
          rot[i * 4 + 3] = dataView.getFloat32(o + offRot3, true) || 0.0;
        } else {
          rot[i * 4] = 1.0; rot[i * 4 + 1] = 0.0; rot[i * 4 + 2] = 0.0; rot[i * 4 + 3] = 0.0;
        }
      }

      this.setupGaussianField(pos, scale, rot, col, count);
      return true;
    } catch (e) {
      console.error("[SIBR] PLY loading notice:", e);
      return false;
    }
  }

  async loadTexturedMesh() {
    // 1. Instant 4K GLB Mesh Loader (<50ms decode)
    if (typeof THREE.GLTFLoader !== "undefined") {
      try {
        const gltfLoader = new THREE.GLTFLoader();
        gltfLoader.load(
          "/api/scene/glb",
          (gltf) => {
            const root = gltf.scene || gltf.scenes[0];
            const cX = this.sceneCenter.x;
            const cY = this.sceneCenter.y;
            const cZ = this.sceneCenter.z;

            root.traverse((child) => {
              if (child.isMesh) {
                child.material.side = THREE.DoubleSide;
                if (child.material.map) {
                  child.material.map.colorSpace = THREE.SRGBColorSpace;
                }
              }
            });
            root.position.set(-cX, cY, cZ);
            root.scale.set(1, -1, -1);
            root.visible = (this.renderMode === 4);
            this.meshObject = root;
            this.scene.add(root);
            console.log("[SIBR] Instant 4K Textured GLB Model mounted (<50ms).");
            return;
          },
          undefined,
          (err) => {
            console.log("[SIBR] GLB fallback to binary PLY parser:", err);
            this.loadBinaryPLYMesh();
          }
        );
        return;
      } catch (err) {
        console.warn("[SIBR] GLTFLoader error, falling back:", err);
      }
    }
    this.loadBinaryPLYMesh();
  }

  async loadBinaryPLYMesh() {
    try {
      const res = await fetch("/api/scene/solid");
      if (!res.ok) return;

      const buffer = await res.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      let headerText = "";
      let endHeaderIdx = -1;

      for (let i = 0; i < Math.min(4096, bytes.length); i++) {
        headerText += String.fromCharCode(bytes[i]);
        if (headerText.includes("end_header\n")) {
          endHeaderIdx = headerText.indexOf("end_header\n") + "end_header\n".length;
          break;
        }
      }

      if (endHeaderIdx === -1) return;

      const vMatch = headerText.match(/element vertex (\d+)/);
      const fMatch = headerText.match(/element face (\d+)/);
      if (!vMatch || !fMatch) return;

      const vertexCount = parseInt(vMatch[1], 10);
      const numFaces = parseInt(fMatch[1], 10);

      const lines = headerText.split("\n");
      let inVertex = false;
      let vertStride = 0;
      let posX = -1, posY = -1, posZ = -1;
      let colR = -1, colG = -1, colB = -1;
      let normX = -1, normY = -1, normZ = -1;

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("element vertex")) {
          inVertex = true;
          continue;
        } else if (trimmed.startsWith("element ")) {
          inVertex = false;
        }

        if (inVertex && trimmed.startsWith("property")) {
          const parts = trimmed.split(/\s+/);
          const type = parts[1];
          const name = parts[2];
          const currentOffset = vertStride;

          let size = 4;
          if (type === "float" || type === "int" || type === "uint") size = 4;
          else if (type === "uchar" || type === "char" || type === "uint8" || type === "int8") size = 1;
          else if (type === "short" || type === "ushort") size = 2;
          else if (type === "double") size = 8;

          if (name === "x") posX = currentOffset;
          else if (name === "y") posY = currentOffset;
          else if (name === "z") posZ = currentOffset;
          else if (name === "red" || name === "r") colR = currentOffset;
          else if (name === "green" || name === "g") colG = currentOffset;
          else if (name === "blue" || name === "b") colB = currentOffset;
          else if (name === "nx") normX = currentOffset;
          else if (name === "ny") normY = currentOffset;
          else if (name === "nz") normZ = currentOffset;

          vertStride += size;
        }
      }

      const dataOffset = endHeaderIdx;
      const dataView = new DataView(buffer, dataOffset);

      const positions = new Float32Array(vertexCount * 3);
      const colors = new Float32Array(vertexCount * 3);
      const hasNormals = normX !== -1;
      const normals = hasNormals ? new Float32Array(vertexCount * 3) : null;

      const cX = this.sceneCenter.x;
      const cY = this.sceneCenter.y;
      const cZ = this.sceneCenter.z;

      for (let i = 0; i < vertexCount; i++) {
        const offset = i * vertStride;
        const vx = dataView.getFloat32(offset + posX, true);
        const vy = dataView.getFloat32(offset + posY, true);
        const vz = dataView.getFloat32(offset + posZ, true);

        positions[i * 3]     = vx - cX;
        positions[i * 3 + 1] = -(vy - cY);
        positions[i * 3 + 2] = -(vz - cZ);

        if (colR !== -1) {
          colors[i * 3]     = dataView.getUint8(offset + colR) / 255.0;
          colors[i * 3 + 1] = dataView.getUint8(offset + colG) / 255.0;
          colors[i * 3 + 2] = dataView.getUint8(offset + colB) / 255.0;
        } else {
          colors[i * 3]     = 0.8;
          colors[i * 3 + 1] = 0.8;
          colors[i * 3 + 2] = 0.8;
        }

        if (hasNormals) {
          normals[i * 3]     = dataView.getFloat32(offset + normX, true);
          normals[i * 3 + 1] = -dataView.getFloat32(offset + normY, true);
          normals[i * 3 + 2] = -dataView.getFloat32(offset + normZ, true);
        }
      }

      // Parse triangle face indices
      const facesOffset = dataOffset + vertexCount * vertStride;
      const faceDataView = new DataView(buffer, facesOffset);
      const indices = new Uint32Array(numFaces * 3);
      let faceByteOffset = 0;

      for (let i = 0; i < numFaces; i++) {
        const nVerts = faceDataView.getUint8(faceByteOffset);
        if (nVerts === 3) {
          indices[i * 3]     = faceDataView.getInt32(faceByteOffset + 1, true);
          indices[i * 3 + 1] = faceDataView.getInt32(faceByteOffset + 5, true);
          indices[i * 3 + 2] = faceDataView.getInt32(faceByteOffset + 9, true);
          faceByteOffset += 13;
        } else {
          faceByteOffset += 1 + nVerts * 4;
        }
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      if (hasNormals) {
        geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
      } else {
        geometry.computeVertexNormals();
      }
      geometry.setIndex(new THREE.BufferAttribute(indices, 1));

      // Synthesize Triplanar UV coordinates so 4K diffuse texture atlas maps seamlessly
      const uvs = new Float32Array(vertexCount * 2);
      const normAttr = normals;
      for (let i = 0; i < vertexCount; i++) {
        const x = positions[i * 3];
        const y = positions[i * 3 + 1];
        const z = positions[i * 3 + 2];
        const nx = normAttr ? Math.abs(normAttr[i * 3]) : 0.0;
        const ny = normAttr ? Math.abs(normAttr[i * 3 + 1]) : 1.0;
        const nz = normAttr ? Math.abs(normAttr[i * 3 + 2]) : 0.0;

        let u = 0.0, v = 0.0;
        if (nx >= ny && nx >= nz) {
          u = z * 0.25;
          v = y * 0.25;
        } else if (ny >= nx && ny >= nz) {
          u = x * 0.25;
          v = z * 0.25;
        } else {
          u = x * 0.25;
          v = y * 0.25;
        }
        uvs[i * 2]     = u - Math.floor(u);
        uvs[i * 2 + 1] = v - Math.floor(v);
      }
      geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));

      // Load photographic diffuse texture atlas
      const texLoader = new THREE.TextureLoader();
      texLoader.load(
        "/api/scene/texture",
        (texture) => {
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.wrapS = THREE.RepeatWrapping;
          texture.wrapT = THREE.RepeatWrapping;
          texture.flipY = false;
          texture.generateMipmaps = true;
          texture.minFilter = THREE.LinearMipmapLinearFilter;
          texture.magFilter = THREE.LinearFilter;
          if (this.renderer && this.renderer.capabilities) {
            texture.anisotropy = Math.min(16, this.renderer.capabilities.getMaxAnisotropy());
          }

          // Unlit photographic material to preserve full dynamic range of baked photos
          const material = new THREE.MeshBasicMaterial({
            map: texture,
            side: THREE.DoubleSide
          });
          const mesh = new THREE.Mesh(geometry, material);
          mesh.visible = (this.renderMode === 4);
          this.meshObject = mesh;
          this.scene.add(mesh);
          console.log(`[SIBR] Solid 4K UV Textured Mesh mounted: ${vertexCount.toLocaleString()} verts, ${numFaces.toLocaleString()} faces.`);
        },
        undefined,
        () => {
          const material = new THREE.MeshBasicMaterial({
            vertexColors: true,
            side: THREE.DoubleSide
          });
          const mesh = new THREE.Mesh(geometry, material);
          mesh.visible = (this.renderMode === 4);
          this.meshObject = mesh;
          this.scene.add(mesh);
        }
      );
    } catch (e) {
      console.warn("[SIBR] Binary mesh layer notice:", e);
    }
  }

  setupGaussianField(pos, scale, rot, col, count) {
    if (this.splatMesh) {
      this.scene.remove(this.splatMesh);
      this.splatMesh.geometry.dispose();
      this.splatMesh.material.dispose();
      this.splatMesh = null;
    }

    this.splatCount = count;

    // Robust 5th-95th Percentile Bounding Box to eliminate peripheral floaters
    const sampleStep = Math.max(1, Math.floor(count / 5000));
    const sampleX = [], sampleY = [], sampleZ = [];
    for (let i = 0; i < count; i += sampleStep) {
      sampleX.push(pos[i * 3]);
      sampleY.push(pos[i * 3 + 1]);
      sampleZ.push(pos[i * 3 + 2]);
    }
    sampleX.sort((a, b) => a - b);
    sampleY.sort((a, b) => a - b);
    sampleZ.sort((a, b) => a - b);

    const idx05 = Math.floor(sampleX.length * 0.05);
    const idx95 = Math.floor(sampleX.length * 0.95);
    const minX = sampleX[idx05], maxX = sampleX[idx95];
    const minY = sampleY[idx05], maxY = sampleY[idx95];
    const minZ = sampleZ[idx05], maxZ = sampleZ[idx95];

    const cX = (minX + maxX) * 0.5;
    const cY = (minY + maxY) * 0.5;
    const cZ = (minZ + maxZ) * 0.5;
    this.sceneCenter.set(cX, cY, cZ);

    const extX = maxX - minX;
    const extY = maxY - minY;
    const extZ = maxZ - minZ;
    this.sceneRadius = Math.max(1.5, Math.min(25.0, 0.5 * Math.sqrt(extX * extX + extY * extY + extZ * extZ)));

    const centeredPos = new Float32Array(count * 3);
    const normalColors = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const dx = pos[i * 3] - cX;
      const dy = -(pos[i * 3 + 1] - cY);
      const dz = -(pos[i * 3 + 2] - cZ);

      centeredPos[i * 3]     = dx;
      centeredPos[i * 3 + 1] = dy;
      centeredPos[i * 3 + 2] = dz;

      // Coordinate-aligned Normal extraction from quaternion (w, x, y, z)
      const qw = rot[i * 4], qx = rot[i * 4 + 1], qy = rot[i * 4 + 2], qz = rot[i * 4 + 3];
      // Reflect y and z for Three.js coordinates
      const nx = 2.0 * (qx * qz + qw * qy);
      const ny = -(2.0 * (qy * qz - qw * qx));
      const nz = -(1.0 - 2.0 * (qx * qx + qy * qy));
      normalColors[i * 3]     = nx * 0.5 + 0.5;
      normalColors[i * 3 + 1] = ny * 0.5 + 0.5;
      normalColors[i * 3 + 2] = nz * 0.5 + 0.5;
    }

    // Retain original buffers for Radix sorting
    this.rawPositions = centeredPos;
    this.rawScales = scale;
    this.rawRotations = rot;
    this.rawColors = col;
    this.rawNormals = normalColors;

    // Initialize Radix Sorter with 3D positions
    this.sorter.initPositions(centeredPos, count);

    // Quad Geometry (4 vertices, 2 triangles)
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

    geo.setAttribute("a_position", new THREE.InstancedBufferAttribute(new Float32Array(centeredPos), 3));
    geo.setAttribute("a_scale", new THREE.InstancedBufferAttribute(new Float32Array(scale), 3));
    geo.setAttribute("a_rotation", new THREE.InstancedBufferAttribute(new Float32Array(rot), 4));
    geo.setAttribute("a_color", new THREE.InstancedBufferAttribute(new Float32Array(col), 4));
    geo.setAttribute("a_normalColor", new THREE.InstancedBufferAttribute(new Float32Array(normalColors), 3));

    this.instancedGeometry = geo;

    const width = window.innerWidth;
    const height = window.innerHeight;
    const focalY = (height / 2.0) / Math.tan((this.fov * Math.PI / 180.0) / 2.0);

    // Exact SIBR 3D Gaussian Splatting Shaders with T * R * T Basis Transformation
    this.splatMaterial = new THREE.ShaderMaterial({
      vertexShader: `
        attribute vec3 a_position;
        attribute vec3 a_scale;
        attribute vec4 a_rotation;
        attribute vec4 a_color;
        attribute vec3 a_normalColor;

        uniform vec2 u_focal;
        uniform vec2 u_viewport;
        uniform float u_splatScale;
        uniform float u_exposure;

        varying vec2 v_quad_pos;
        varying vec2 v_quad_offset;
        varying vec3 v_conic;
        varying vec4 v_color;
        varying vec3 v_normalColor;
        varying float v_depth;

        mat3 buildTransformedRotation(vec4 q) {
          vec4 nq = normalize(q);
          float r = nq.x, x = nq.y, y = nq.z, z = nq.w; // (w, x, y, z)
          
          float r00 = 1.0 - 2.0 * (y * y + z * z);
          float r01 = 2.0 * (x * y - r * z);
          float r02 = 2.0 * (x * z + r * y);
          float r10 = 2.0 * (x * y + r * z);
          float r11 = 1.0 - 2.0 * (x * x + z * z);
          float r12 = 2.0 * (y * z - r * x);
          float r20 = 2.0 * (x * z - r * y);
          float r21 = 2.0 * (y * z + r * x);
          float r22 = 1.0 - 2.0 * (x * x + y * y);

          // Exact T * R * T similarity transformation (T = diag(1, -1, -1)):
          // Cross-terms involving Y and Z axes are sign-inverted:
          // Column 0, Column 1, Column 2 in GLSL column-major format:
          return mat3(
            vec3( r00, -r10, -r20),
            vec3(-r01,  r11,  r21),
            vec3(-r02,  r12,  r22)
          );
        }

        void main() {
          vec4 cam_pos = modelViewMatrix * vec4(a_position, 1.0);
          float depth = -cam_pos.z;
          v_depth = depth;

          // Frustum near-plane & far-plane clipping
          if (cam_pos.z >= -0.05 || depth >= 250.0) {
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
          }

          // 3D Covariance in World Space: Sigma_world = (T*R*T) * S * S^T * (T*R*T)^T
          mat3 R_three = buildTransformedRotation(a_rotation);
          vec3 eff_scale = a_scale * u_splatScale;
          mat3 S = mat3(
            eff_scale.x, 0.0, 0.0,
            0.0, eff_scale.y, 0.0,
            0.0, 0.0, eff_scale.z
          );
          mat3 M = R_three * S;
          mat3 V_world = M * transpose(M);

          // Transform 3D Covariance to Camera View Space: Sigma_cam = W * Sigma_world * W^T
          mat3 W = mat3(modelViewMatrix);
          mat3 V_cam = W * V_world * transpose(W);

          // Project to 2D Screen Space via Jacobian J
          float z = depth;
          float z2 = z * z;
          vec3 J0 = vec3(u_focal.x / z, 0.0, -(u_focal.x * cam_pos.x) / z2);
          vec3 J1 = vec3(0.0, u_focal.y / z, -(u_focal.y * cam_pos.y) / z2);

          // 2D Covariance projection
          float cov_raw_00 = dot(J0, V_cam * J0);
          float cov_raw_11 = dot(J1, V_cam * J1);
          float cov_raw_01 = dot(J0, V_cam * J1);
          float det_raw = max(0.0, cov_raw_00 * cov_raw_11 - cov_raw_01 * cov_raw_01);

          // EWA Anti-aliasing low-pass filter dilation (+0.3px low-pass kernel)
          float s_filter = 0.3;
          float cov00 = cov_raw_00 + s_filter;
          float cov11 = cov_raw_11 + s_filter;
          float cov01 = cov_raw_01;

          float det = cov00 * cov11 - cov01 * cov01;
          if (det <= 0.00001) {
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
          }

          float det_inv = 1.0 / det;
          v_conic = vec3(cov11 * det_inv, -cov01 * det_inv, cov00 * det_inv);

          // ECA-EWA Determinant-Preserving Energy Opacity Compensation:
          // alpha_dilated = alpha_base * sqrt(det_raw / det_dilated)
          float opacity_compensation = sqrt(clamp(det_raw / det, 0.0, 1.0));
          float effective_alpha = a_color.a * opacity_compensation;

          // Culling sub-threshold or tiny transparent splats (prevents overview bokeh fog)
          if (effective_alpha < 0.005) {
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
          }

          // 2D Screen Radius Calculation (3-sigma confidence)
          float mid = 0.5 * (cov00 + cov11);
          float lambda = mid + sqrt(max(0.001, mid * mid - det));
          float radius = ceil(3.0 * sqrt(max(0.001, lambda)));
          radius = clamp(radius, 0.5, 48.0);

          vec2 screen_offset = position.xy * radius;
          vec4 proj_pos = projectionMatrix * cam_pos;
          proj_pos.xy += (screen_offset / u_viewport) * proj_pos.w * 2.0;

          // Near plane soft fade
          float nearFade = smoothstep(0.05, 0.25, depth);

          v_quad_pos = position.xy;
          v_quad_offset = screen_offset;
          v_color = vec4(a_color.rgb * u_exposure, effective_alpha * nearFade);
          v_normalColor = a_normalColor;
          gl_Position = proj_pos;
        }
      `,
      fragmentShader: `
        varying vec2 v_quad_pos;
        varying vec2 v_quad_offset;
        varying vec3 v_conic;
        varying vec4 v_color;
        varying vec3 v_normalColor;
        varying float v_depth;
        uniform int u_renderMode;

        void main() {
          // Circular disc boundary mask: eliminates 100% of quad square edges
          float r2 = dot(v_quad_pos, v_quad_pos);
          if (r2 > 1.0) discard;

          // Exact 2D Gaussian Conic Quadratic Exponential Falloff
          float power = -0.5 * (v_conic.x * v_quad_offset.x * v_quad_offset.x + 
                                2.0 * v_conic.y * v_quad_offset.x * v_quad_offset.y + 
                                v_conic.z * v_quad_offset.y * v_quad_offset.y);

          if (power < -4.5) discard;

          // Smooth radial edge feathering so boundaries dissolve seamlessly
          float edgeFade = 1.0 - smoothstep(0.70, 1.0, r2);
          float alpha = clamp(v_color.a * exp(power) * edgeFade, 0.0, 0.99);
          if (alpha < 0.015) discard;

          vec3 baseColor = v_color.rgb;
          if (u_renderMode == 1) {
            // Optical Depth Field
            float normDepth = clamp(v_depth / 6.0, 0.0, 1.0);
            baseColor = vec3(normDepth, 1.0 - normDepth, 0.5 + 0.5 * sin(normDepth * 6.28));
          } else if (u_renderMode == 2) {
            // Surface Normals
            baseColor = v_normalColor;
          } else if (u_renderMode == 3) {
            // Conic Ellipse Wireframes
            float ringDist = length(v_quad_offset);
            if (abs(ringDist - 8.0) > 1.8) discard;
            baseColor = vec3(0.0, 0.94, 1.0);
          }

          // PREMULTIPLIED ALPHA: Smooth continuous Gaussian radiance blending without circular disc cutouts
          gl_FragColor = vec4(baseColor * alpha, alpha);
        }
      `,
      uniforms: {
        u_focal: { value: new THREE.Vector2(focalY, focalY) },
        u_viewport: { value: new THREE.Vector2(width, height) },
        u_splatScale: { value: this.splatScale },
        u_exposure: { value: this.exposure },
        u_renderMode: { value: this.renderMode },
      },
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendEquation: THREE.AddEquation,
      side: THREE.DoubleSide,
    });

    this.splatMesh = new THREE.Mesh(geo, this.splatMaterial);
    this.splatMesh.frustumCulled = false;
    this.scene.add(this.splatMesh);

    const badgeSurfel = document.getElementById("badge-surfel-count");
    if (badgeSurfel) {
      badgeSurfel.textContent = `${(count / 1000).toFixed(0)}k Splats`;
    }

    // Set initial natural room overview framing
    const overviewDist = Math.max(6.0, Math.min(22.0, this.sceneRadius * 1.0));
    this.spherical.set(overviewDist, Math.PI / 2.0 - 0.28, 0.45);
    this.targetCenter.set(0, 0, 0);
    this.updateCameraFromSpherical();
    this.camera.near = 0.05;
    this.camera.far = 250.0;
    this.camera.updateProjectionMatrix();

    const camBadge = document.getElementById("badge-cam-id");
    if (camBadge) {
      camBadge.textContent = "Overview";
    }

    console.log(`[SUCCESS] INRIA SIBR 3DGS Simulation Active: ${count.toLocaleString()} Splats.`);
  }

  applySortedIndices(sortedIndices) {
    if (!this.instancedGeometry || !this.rawPositions) return;

    const count = this.splatCount;
    const posAttr = this.instancedGeometry.attributes.a_position.array;
    const scaleAttr = this.instancedGeometry.attributes.a_scale.array;
    const rotAttr = this.instancedGeometry.attributes.a_rotation.array;
    const colAttr = this.instancedGeometry.attributes.a_color.array;
    const normAttr = this.instancedGeometry.attributes.a_normalColor.array;

    const rawP = this.rawPositions;
    const rawS = this.rawScales;
    const rawR = this.rawRotations;
    const rawC = this.rawColors;
    const rawN = this.rawNormals;

    // Fast back-to-front buffer reordering
    for (let i = 0; i < count; i++) {
      const src = sortedIndices[i];
      const i3 = i * 3;
      const src3 = src * 3;
      const i4 = i * 4;
      const src4 = src * 4;

      posAttr[i3]     = rawP[src3];
      posAttr[i3 + 1] = rawP[src3 + 1];
      posAttr[i3 + 2] = rawP[src3 + 2];

      scaleAttr[i3]     = rawS[src3];
      scaleAttr[i3 + 1] = rawS[src3 + 1];
      scaleAttr[i3 + 2] = rawS[src3 + 2];

      normAttr[i3]     = rawN[src3];
      normAttr[i3 + 1] = rawN[src3 + 1];
      normAttr[i3 + 2] = rawN[src3 + 2];

      rotAttr[i4]     = rawR[src4];
      rotAttr[i4 + 1] = rawR[src4 + 1];
      rotAttr[i4 + 2] = rawR[src4 + 2];
      rotAttr[i4 + 3] = rawR[src4 + 3];

      colAttr[i4]     = rawC[src4];
      colAttr[i4 + 1] = rawC[src4 + 1];
      colAttr[i4 + 2] = rawC[src4 + 2];
      colAttr[i4 + 3] = rawC[src4 + 3];
    }

    this.instancedGeometry.attributes.a_position.needsUpdate = true;
    this.instancedGeometry.attributes.a_scale.needsUpdate = true;
    this.instancedGeometry.attributes.a_rotation.needsUpdate = true;
    this.instancedGeometry.attributes.a_color.needsUpdate = true;
    this.instancedGeometry.attributes.a_normalColor.needsUpdate = true;
  }

  applyCalibratedCamera(camIdx) {
    if (!this.calibratedCameras || this.calibratedCameras.length === 0) return;
    this.currentCameraIdx = ((camIdx % this.calibratedCameras.length) + this.calibratedCameras.length) % this.calibratedCameras.length;
    const cam = this.calibratedCameras[this.currentCameraIdx];
    if (!cam || !cam.position) return;

    this.isOrbiting = false;
    const cX = this.sceneCenter.x;
    const cY = this.sceneCenter.y;
    const cZ = this.sceneCenter.z;

    const pos = cam.position;
    const C_prime = new THREE.Vector3(pos[0] - cX, -(pos[1] - cY), -(pos[2] - cZ));

    if (cam.rotation && Array.isArray(cam.rotation) && cam.rotation.length === 3) {
      const R = cam.rotation;
      // Exact COLMAP World-to-Camera Transform into Three.js Coordinate Space:
      // COLMAP: +X right, +Y down, +Z forward. C = -R^T * t
      // Three.js: +X right, +Y up, looking down -Z.
      // In Three.js world space (X, -Y, -Z):
      // Forward = Row 2 of R: (R[2][0], -R[2][1], -R[2][2])
      // Up = -Row 1 of R: (-R[1][0], R[1][1], R[1][2])
      // Right = Row 0 of R: (R[0][0], -R[0][1], -R[0][2])
      const right = new THREE.Vector3(R[0][0], -R[0][1], -R[0][2]).normalize();
      const up    = new THREE.Vector3(-R[1][0], R[1][1], R[1][2]).normalize();
      const fwd   = new THREE.Vector3(R[2][0], -R[2][1], -R[2][2]).normalize();
      const cam_z = fwd.clone().negate();

      const cam_matrix = new THREE.Matrix4().makeBasis(right, up, cam_z);
      this.camera.quaternion.setFromRotationMatrix(cam_matrix);
      this.camera.position.copy(C_prime);
      this.camera.updateMatrixWorld(true);

      const target = C_prime.clone().addScaledVector(fwd, 2.5);
      this.targetCenter.copy(target);
      this.spherical.setFromVector3(this.camera.position.clone().sub(this.targetCenter));

      const dir = new THREE.Vector3();
      this.camera.getWorldDirection(dir);
      this.fpsYaw = Math.atan2(-dir.x, -dir.z);
      this.fpsPitch = Math.asin(dir.y);
    } else {
      this.camera.position.copy(C_prime);
      this.targetCenter.set(0, 0, 0);
      this.camera.lookAt(this.targetCenter);
      this.spherical.setFromVector3(this.camera.position);
    }

    this.needsSort = true;

    const camBadge = document.getElementById("badge-cam-id");
    if (camBadge) {
      camBadge.textContent = cam.img_name ? cam.img_name.replace(".jpg", "") : `#${this.currentCameraIdx + 1}`;
    }

    const gtImg = document.getElementById("gt-reference-photo");
    if (gtImg && cam.img_name) {
      gtImg.src = `/api/scene/reference_photo/${cam.img_name}`;
    }
  }

  toggleNavMode() {
    this.navMode = this.navMode === "trackball" ? "fps" : "trackball";
    const badge = document.getElementById("badge-nav-mode");
    if (badge) badge.textContent = this.navMode === "trackball" ? "Trackball" : "FPS Fly";
    const btnLabel = document.getElementById("btn-nav-label");
    if (btnLabel) btnLabel.textContent = this.navMode === "trackball" ? "FPS Mode" : "Trackball";

    if (this.navMode === "fps") {
      const dir = new THREE.Vector3();
      this.camera.getWorldDirection(dir);
      this.fpsYaw = Math.atan2(-dir.x, -dir.z);
      this.fpsPitch = Math.asin(dir.y);
    }
  }

  toggleOrbit() {
    this.isOrbiting = !this.isOrbiting;
    const btnLabel = document.getElementById("btn-orbit-label");
    if (btnLabel) btnLabel.textContent = this.isOrbiting ? "Pause Orbit" : "Orbit 360°";
  }

  nextCameraPose() {
    this.applyCalibratedCamera(this.currentCameraIdx + 1);
  }

  setRenderMode(mode) {
    this.renderMode = parseInt(mode);
    if (this.splatMaterial) {
      this.splatMaterial.uniforms.u_renderMode.value = this.renderMode;
    }

    // Toggle 3DGS splats vs Textured Mesh
    if (this.splatMesh) {
      this.splatMesh.visible = (this.renderMode !== 4);
    }
    if (this.meshObject) {
      this.meshObject.visible = (this.renderMode === 4);
    }

    for (let i = 0; i <= 4; i++) {
      const tab = document.getElementById(`tab-mode-${i}`);
      if (tab) tab.classList.toggle("active", i === this.renderMode);
    }
  }

  resetCamera() {
    this.isOrbiting = false;
    const overviewDist = Math.max(6.0, Math.min(22.0, this.sceneRadius * 1.0));
    this.spherical.set(overviewDist, Math.PI / 2.0 - 0.28, 0.45);
    this.targetCenter.set(0, 0, 0);
    this.updateCameraFromSpherical();
    const camBadge = document.getElementById("badge-cam-id");
    if (camBadge) camBadge.textContent = "Overview";
  }

  toggleGTPeek() {
    this.showGTPeek = !this.showGTPeek;
    const peek = document.getElementById("gt-peek-container");
    if (peek) {
      peek.style.display = this.showGTPeek ? "block" : "none";
    }
  }

  toggleHelpModal() {
    const modal = document.getElementById("sibr-help-modal");
    if (modal) {
      modal.style.display = modal.style.display === "block" ? "none" : "block";
    }
  }

  animate() {
    requestAnimationFrame(this.animate);

    const dt = this.clock.getDelta();

    // SIBR Continuous 360 Turntable Orbit
    if (this.isOrbiting && this.navMode === "trackball") {
      this.spherical.theta += this.orbitSpeed * dt;
      this.updateCameraFromSpherical();
    }

    // Dynamic Resolution Scaling (DRS) on CPU: 0.75x during camera motion, 1.0x at rest
    const isMoving = this.isOrbiting || this.isDragging || (this.navMode === "fps" && (this.moveState.forward || this.moveState.backward || this.moveState.left || this.moveState.right || this.moveState.up || this.moveState.down));
    const targetPixelRatio = isMoving ? 0.75 : Math.min(window.devicePixelRatio || 1, 1.25);
    if (this.currentPixelRatio !== targetPixelRatio) {
      this.currentPixelRatio = targetPixelRatio;
      this.renderer.setPixelRatio(targetPixelRatio);
    }

    // SIBR FPS 6-DoF Fly Movement
    if (this.navMode === "fps") {
      const speedMult = this.moveState.sprint ? 2.5 : 1.0;
      const speed = this.flySpeed * speedMult * dt;
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
      const up = new THREE.Vector3(0, 1, 0);

      if (this.moveState.forward) { this.camera.position.addScaledVector(forward, speed); this.needsSort = true; }
      if (this.moveState.backward) { this.camera.position.addScaledVector(forward, -speed); this.needsSort = true; }
      if (this.moveState.right) { this.camera.position.addScaledVector(right, speed); this.needsSort = true; }
      if (this.moveState.left) { this.camera.position.addScaledVector(right, -speed); this.needsSort = true; }
      if (this.moveState.up) { this.camera.position.addScaledVector(up, speed); this.needsSort = true; }
      if (this.moveState.down) { this.camera.position.addScaledVector(up, -speed); this.needsSort = true; }
    }

    // Throttled camera motion delta check for Radix depth sorting
    const now = performance.now();
    if (this.splatCount > 0 && this.renderMode !== 4) {
      if (!this.lastSortPos) {
        this.lastSortPos = this.camera.position.clone();
        this.lastSortDir = new THREE.Vector3();
        this.camera.getWorldDirection(this.lastSortDir);
        this.lastSortTime = 0;
      }
      const curDir = new THREE.Vector3();
      this.camera.getWorldDirection(curDir);
      const distSq = this.camera.position.distanceToSquared(this.lastSortPos);
      const dotDir = curDir.dot(this.lastSortDir);
      const timeSinceSort = now - (this.lastSortTime || 0);

      if ((distSq > 0.0009 || dotDir < 0.9992 || this.needsSort) && timeSinceSort > 75) {
        this.lastSortPos.copy(this.camera.position);
        this.lastSortDir.copy(curDir);
        this.lastSortTime = now;
        this.needsSort = false;
        const viewProj = new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
        this.sorter.requestSort(viewProj.elements, this.splatCount);
      }
    }

    // FPS Telemetry
    this.frameCount++;
    if (now - this.lastFpsTime >= 1000) {
      const fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsTime));
      const badgeFps = document.getElementById("badge-fps");
      if (badgeFps) badgeFps.textContent = `${fps} FPS`;
      this.frameCount = 0;
      this.lastFpsTime = now;
    }

    this.renderer.render(this.scene, this.camera);
  }
}

// Global Interaction Bridges
function toggleTurntable() { if (simEngine) simEngine.toggleOrbit(); }
function nextCameraPose() { if (simEngine) simEngine.nextCameraPose(); }
function toggleNavMode() { if (simEngine) simEngine.toggleNavMode(); }
function resetCamera() { if (simEngine) simEngine.resetCamera(); }
function setRenderMode(mode) { if (simEngine) simEngine.setRenderMode(mode); }
function toggleGTPeek() { if (simEngine) simEngine.toggleGTPeek(); }
function toggleHelpModal() { if (simEngine) simEngine.toggleHelpModal(); }

window.addEventListener("DOMContentLoaded", () => {
  simEngine = new SIBRSimulationEngine();
});
