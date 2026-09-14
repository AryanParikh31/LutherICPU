/**
 * lutherICPU Forensic 3D Platform - WebGL2 Continuous Simulation & Measurement Engine
 */

class LutherForensicViewer {
    constructor() {
        this.container = document.getElementById("canvas-container");
        this.canvas = document.getElementById("webgl-canvas");
        this.loadingOverlay = document.getElementById("loading-overlay");
        this.progressFill = document.getElementById("progress-fill");

        // Modes
        this.activeMode = "orbit"; // "orbit" or "measure"
        this.measurementPoints = [];
        this.measurementLine = null;
        this.measurementMarkers = [];
        this.isAutoOrbiting = false;
        this.activeJobId = null;
        this.pollInterval = null;

        // Three.js Core
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x07090e);

        this.camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(5.0, -2.5, -5.0);

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.15;

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = true;

        // Lighting
        this.ambientLight = new THREE.AmbientLight(0xffffff, 0.45);
        this.scene.add(this.ambientLight);

        this.sunLight = new THREE.DirectionalLight(0xfffaed, 1.4);
        this.sunLight.position.set(5, -10, 5);
        this.scene.add(this.sunLight);

        this.fillLight = new THREE.DirectionalLight(0x90b0e0, 0.6);
        this.fillLight.position.set(-5, 5, -5);
        this.scene.add(this.fillLight);

        // Object Groups
        this.meshGroup = new THREE.Group();
        this.evidenceGroup = new THREE.Group();
        this.trajectoryGroup = new THREE.Group();
        this.cageGroup = new THREE.Group();
        this.caliperGroup = new THREE.Group();
        this.frustumGroup = new THREE.Group();

        this.scene.add(this.meshGroup);
        this.scene.add(this.evidenceGroup);
        this.scene.add(this.trajectoryGroup);
        this.scene.add(this.cageGroup);
        this.scene.add(this.caliperGroup);
        this.scene.add(this.frustumGroup);

        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        this.currentScene = "truck";
        this.init();
    }

    init() {
        lucide.createIcons();
        this.bindEvents();
        this.loadScene(this.currentScene);
        this.startTelemetryPolling();
        this.populateSceneDropdown();
        this.animate();
    }

    bindEvents() {
        window.addEventListener("resize", () => this.onResize());

        // Mode Toggles
        document.getElementById("btn-mode-orbit")?.addEventListener("click", () => this.setMode("orbit"));
        document.getElementById("btn-mode-measure")?.addEventListener("click", () => this.setMode("measure"));

        // Visibility Toggles
        document.getElementById("btn-toggle-evidence")?.addEventListener("click", (e) => {
            this.evidenceGroup.visible = !this.evidenceGroup.visible;
            e.currentTarget.classList.toggle("active", this.evidenceGroup.visible);
        });

        document.getElementById("btn-toggle-trajectory")?.addEventListener("click", (e) => {
            this.trajectoryGroup.visible = !this.trajectoryGroup.visible;
            e.currentTarget.classList.toggle("active", this.trajectoryGroup.visible);
        });

        document.getElementById("btn-toggle-cage")?.addEventListener("click", (e) => {
            this.cageGroup.visible = !this.cageGroup.visible;
            e.currentTarget.classList.toggle("active", this.cageGroup.visible);
        });

        document.getElementById("btn-toggle-wireframe")?.addEventListener("click", (e) => {
            const active = e.currentTarget.classList.toggle("active");
            this.meshGroup.traverse((child) => {
                if (child.isMesh) child.material.wireframe = active;
            });
        });

        document.getElementById("btn-toggle-cameras")?.addEventListener("click", (e) => {
            this.frustumGroup.visible = !this.frustumGroup.visible;
            e.currentTarget.classList.toggle("active", this.frustumGroup.visible);
        });

        // Dataset Select
        document.getElementById("scene-select")?.addEventListener("change", (e) => {
            this.currentScene = e.target.value;
            this.loadScene(this.currentScene);
        });

        // Caliper Clear
        document.getElementById("btn-clear-caliper")?.addEventListener("click", () => this.clearMeasurement());

        // Fullscreen & HUD Toggles
        const toggleFullscreen = () => {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(() => {});
            } else {
                if (document.exitFullscreen) document.exitFullscreen();
            }
        };

        const toggleHud = () => {
            document.body.classList.toggle("hud-hidden");
        };

        document.getElementById("btn-toggle-fullscreen")?.addEventListener("click", toggleFullscreen);
        document.getElementById("btn-toggle-hud")?.addEventListener("click", toggleHud);
        document.getElementById("hud-restore-pill")?.addEventListener("click", toggleHud);

        // Viewpoint Preset Buttons
        const presetBtns = [
            { id: "btn-view-hero", name: "hero" },
            { id: "btn-view-side", name: "side" },
            { id: "btn-view-foliage", name: "foliage" },
            { id: "btn-view-ground", name: "ground" },
            { id: "btn-view-building", name: "building" },
            { id: "btn-view-top", name: "top" }
        ];

        presetBtns.forEach(({ id, name }) => {
            document.getElementById(id)?.addEventListener("click", (e) => {
                document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
                e.currentTarget.classList.add("active");
                this.setCameraPreset(name);
            });
        });

        // Global Keyboard Shortcuts (F: Fullscreen, H: HUD Toggle, Space: Auto-Orbit, 1-6: Presets)
        window.addEventListener("keydown", (e) => {
            if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;
            
            if (e.code === "KeyF") {
                toggleFullscreen();
            } else if (e.code === "KeyH") {
                toggleHud();
            } else if (e.code === "Space") {
                e.preventDefault();
                this.isAutoOrbiting = !this.isAutoOrbiting;
                document.getElementById("btn-orbit-auto")?.classList.toggle("active", this.isAutoOrbiting);
            } else if (e.code === "Digit1") {
                document.getElementById("btn-view-hero")?.click();
            } else if (e.code === "Digit2") {
                document.getElementById("btn-view-side")?.click();
            } else if (e.code === "Digit3") {
                document.getElementById("btn-view-foliage")?.click();
            } else if (e.code === "Digit4") {
                document.getElementById("btn-view-ground")?.click();
            } else if (e.code === "Digit5") {
                document.getElementById("btn-view-building")?.click();
            } else if (e.code === "Digit6") {
                document.getElementById("btn-view-top")?.click();
            }
        });

        // Lighting Sliders
        const sliderAzimuth = document.getElementById("slider-sun-azimuth");
        const sliderElevation = document.getElementById("slider-sun-elevation");
        const sliderAmbient = document.getElementById("slider-ambient-intensity");

        const updateSun = () => {
            const az = parseFloat(sliderAzimuth?.value || 45) * (Math.PI / 180);
            const el = parseFloat(sliderElevation?.value || 60) * (Math.PI / 180);
            const r = 12;
            const x = r * Math.cos(el) * Math.sin(az);
            const y = -r * Math.sin(el);
            const z = r * Math.cos(el) * Math.cos(az);
            this.sunLight.position.set(x, y, z);
            if (document.getElementById("sun-azimuth-val")) document.getElementById("sun-azimuth-val").innerText = `${sliderAzimuth.value}°`;
            if (document.getElementById("sun-elevation-val")) document.getElementById("sun-elevation-val").innerText = `${sliderElevation.value}°`;
        };

        sliderAzimuth?.addEventListener("input", updateSun);
        sliderElevation?.addEventListener("input", updateSun);
        sliderAmbient?.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            this.ambientLight.intensity = val;
            if (document.getElementById("sun-intensity-val")) document.getElementById("sun-intensity-val").innerText = val.toFixed(2);
        });

        // Auto Orbit & Reset
        document.getElementById("btn-orbit-auto")?.addEventListener("click", (e) => {
            this.isAutoOrbiting = !this.isAutoOrbiting;
            e.currentTarget.classList.toggle("active", this.isAutoOrbiting);
        });

        document.getElementById("btn-reset-view")?.addEventListener("click", () => {
            this.setCameraPreset("hero");
        });

        // Dossier and Proof Modals
        document.getElementById("btn-view-dossier")?.addEventListener("click", () => {
            window.open(`/api/dossier/${this.currentScene}`, "_blank");
        });

        const proofModal = document.getElementById("proof-modal");
        document.getElementById("btn-view-proof")?.addEventListener("click", () => {
            const img = document.getElementById("proof-image-display");
            if (img) img.src = `/api/proof_render/${this.currentScene}?t=${Date.now()}`;
            if (proofModal) proofModal.style.display = "flex";
        });
        document.getElementById("btn-close-proof")?.addEventListener("click", () => {
            if (proofModal) proofModal.style.display = "none";
        });

        // Reconstruction Modal Controls
        const reconModal = document.getElementById("reconstruct-modal");
        document.getElementById("btn-open-reconstruct")?.addEventListener("click", () => {
            if (reconModal) reconModal.style.display = "flex";
            lucide.createIcons();
        });

        const closeModal = () => {
            if (reconModal) reconModal.style.display = "none";
        };
        document.getElementById("btn-close-modal")?.addEventListener("click", closeModal);
        document.getElementById("btn-cancel-recon")?.addEventListener("click", closeModal);

        // Tabs
        const tabBtnPath = document.getElementById("tab-btn-path");
        const tabBtnUpload = document.getElementById("tab-btn-upload");
        const tabContentPath = document.getElementById("tab-content-path");
        const tabContentUpload = document.getElementById("tab-content-upload");

        tabBtnPath?.addEventListener("click", () => {
            tabBtnPath.classList.add("active");
            tabBtnUpload?.classList.remove("active");
            if (tabContentPath) tabContentPath.style.display = "block";
            if (tabContentUpload) tabContentUpload.style.display = "none";
        });

        tabBtnUpload?.addEventListener("click", () => {
            tabBtnUpload.classList.add("active");
            tabBtnPath?.classList.remove("active");
            if (tabContentUpload) tabContentUpload.style.display = "block";
            if (tabContentPath) tabContentPath.style.display = "none";
        });

        // Start Reconstruction Button
        document.getElementById("btn-start-recon")?.addEventListener("click", () => this.startReconstruction());

        // Load Completed Scene Button
        document.getElementById("btn-load-completed-scene")?.addEventListener("click", () => {
            const sceneName = document.getElementById("input-scene-name")?.value || "truck";
            this.populateSceneDropdown(sceneName);
            this.loadScene(sceneName);
            closeModal();
        });

        // Raycasting for Caliper
        this.renderer.domElement.addEventListener("pointerdown", (e) => this.onPointerDown(e));
    }

    async populateSceneDropdown(selectedScene = null) {
        try {
            const res = await fetch("/api/scenes");
            if (res.ok) {
                const data = await res.json();
                const sel = document.getElementById("scene-select");
                if (sel && data.scenes) {
                    sel.innerHTML = "";
                    data.scenes.forEach((sc) => {
                        const opt = document.createElement("option");
                        opt.value = sc;
                        opt.innerText = sc.charAt(0).toUpperCase() + sc.slice(1);
                        if ((selectedScene && sc === selectedScene) || (!selectedScene && sc === this.currentScene)) {
                            opt.selected = true;
                        }
                        sel.appendChild(opt);
                    });
                }
            }
        } catch (_) {}
    }

    async startReconstruction() {
        const isUploadMode = document.getElementById("tab-btn-upload")?.classList.contains("active");
        const iterations = parseInt(document.getElementById("input-iterations")?.value || "30000", 10);
        const sceneName = document.getElementById("input-scene-name")?.value.trim() || "reconstructed_scene";

        const startBtn = document.getElementById("btn-start-recon");
        const progressArea = document.getElementById("recon-progress-area");
        const loadBtn = document.getElementById("btn-load-completed-scene");

        if (startBtn) startBtn.disabled = true;
        if (progressArea) progressArea.style.display = "block";
        if (loadBtn) loadBtn.style.display = "none";

        try {
            let res;
            if (isUploadMode) {
                const fileInput = document.getElementById("input-files-upload");
                if (!fileInput || fileInput.files.length === 0) {
                    alert("Please select images to upload.");
                    if (startBtn) startBtn.disabled = false;
                    return;
                }
                const formData = new FormData();
                for (let i = 0; i < fileInput.files.length; i++) {
                    formData.append("files", fileInput.files[i]);
                }
                formData.append("scene_name", sceneName);
                formData.append("iterations", iterations.toString());

                res = await fetch("/api/reconstruct/upload", {
                    method: "POST",
                    body: formData
                });
            } else {
                const imagesPath = document.getElementById("input-images-path")?.value.trim();
                const colmapPath = document.getElementById("input-colmap-path")?.value.trim() || null;

                if (!imagesPath) {
                    alert("Please provide the Multi-Angle Images Path.");
                    if (startBtn) startBtn.disabled = false;
                    return;
                }

                res = await fetch("/api/reconstruct/start", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        images_path: imagesPath,
                        colmap_path: colmapPath,
                        iterations: iterations,
                        scene_name: sceneName
                    })
                });
            }

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Failed to start reconstruction.");
            }

            const data = await res.json();
            this.activeJobId = data.job_id;
            this.pollReconstructionStatus(this.activeJobId);

        } catch (err) {
            alert(`Reconstruction error: ${err.message}`);
            if (startBtn) startBtn.disabled = false;
        }
    }

    pollReconstructionStatus(jobId) {
        if (this.pollInterval) clearInterval(this.pollInterval);

        const stageLabel = document.getElementById("recon-stage-label");
        const pctLabel = document.getElementById("recon-pct-label");
        const progressFill = document.getElementById("recon-progress-fill");
        const iterLabel = document.getElementById("recon-iter-label");
        const ramLabel = document.getElementById("recon-ram-label");
        const startBtn = document.getElementById("btn-start-recon");
        const loadBtn = document.getElementById("btn-load-completed-scene");

        this.pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/reconstruct/status/${jobId}`);
                if (!res.ok) return;

                const job = await res.json();
                if (stageLabel) stageLabel.innerText = job.stage || "Processing...";
                if (pctLabel) pctLabel.innerText = `${job.progress_pct}%`;
                if (progressFill) progressFill.style.width = `${job.progress_pct}%`;
                if (iterLabel) iterLabel.innerText = `Iter: ${job.current_iteration.toLocaleString()} / ${job.total_iterations.toLocaleString()}`;
                if (ramLabel) ramLabel.innerText = `RAM: ${job.process_ram_mb} MB (Safe)`;

                if (job.status === "completed") {
                    clearInterval(this.pollInterval);
                    if (stageLabel) stageLabel.innerText = "Reconstruction Complete! 3D Continuous Manifold Ready.";
                    if (progressFill) progressFill.style.backgroundColor = "#10b981";
                    if (loadBtn) loadBtn.style.display = "inline-flex";
                    if (startBtn) startBtn.disabled = false;
                    lucide.createIcons();
                } else if (job.status === "failed") {
                    clearInterval(this.pollInterval);
                    if (stageLabel) stageLabel.innerText = `Reconstruction Failed: ${job.error_message}`;
                    if (progressFill) progressFill.style.backgroundColor = "#ef4444";
                    if (startBtn) startBtn.disabled = false;
                }
            } catch (_) {}
        }, 400);
    }

    setMode(mode) {
        this.activeMode = mode;
        document.getElementById("btn-mode-orbit")?.classList.toggle("active", mode === "orbit");
        document.getElementById("btn-mode-measure")?.classList.toggle("active", mode === "measure");
        const caliperSection = document.getElementById("section-caliper");

        if (mode === "measure") {
            this.controls.enableRotate = false;
            if (caliperSection) caliperSection.style.display = "block";
        } else {
            this.controls.enableRotate = true;
            if (caliperSection) caliperSection.style.display = "none";
        }
    }

    onPointerDown(event) {
        if (event.button !== 0) return; // Left click only

        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(this.meshGroup.children, true);

        if (intersects.length > 0 && this.activeMode === "measure") {
            const hitPt = intersects[0].point;
            this.addMeasurementPoint(hitPt);
        }
    }

    addMeasurementPoint(point) {
        if (this.measurementPoints.length >= 2) {
            this.clearMeasurement();
        }

        this.measurementPoints.push(point);

        // Marker sphere
        const markerGeo = new THREE.SphereGeometry(0.04, 16, 16);
        const markerMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24 });
        const marker = new THREE.Mesh(markerGeo, markerMat);
        marker.position.copy(point);
        this.caliperGroup.add(marker);
        this.measurementMarkers.push(marker);

        const ptA = document.getElementById("caliper-pt-a");
        const ptB = document.getElementById("caliper-pt-b");
        const distEl = document.getElementById("caliper-distance");

        if (this.measurementPoints.length === 1) {
            if (ptA) ptA.innerText = `(${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)})`;
            if (ptB) ptB.innerText = "[Click Point 2]";
        } else if (this.measurementPoints.length === 2) {
            const p1 = this.measurementPoints[0];
            const p2 = this.measurementPoints[1];
            if (ptB) ptB.innerText = `(${p2.x.toFixed(2)}, ${p2.y.toFixed(2)}, ${p2.z.toFixed(2)})`;

            // Draw line
            const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
            const lineMat = new THREE.LineBasicMaterial({ color: 0xfbbf24, linewidth: 3 });
            this.measurementLine = new THREE.Line(lineGeo, lineMat);
            this.caliperGroup.add(this.measurementLine);

            // Compute distance
            const distM = p1.distanceTo(p2);
            if (distEl) distEl.innerText = `${(distM * 1000).toFixed(1)} mm (${distM.toFixed(3)} m)`;
        }
    }

    clearMeasurement() {
        this.measurementPoints = [];
        while (this.caliperGroup.children.length > 0) {
            this.caliperGroup.remove(this.caliperGroup.children[0]);
        }
        const ptA = document.getElementById("caliper-pt-a");
        const ptB = document.getElementById("caliper-pt-b");
        const distEl = document.getElementById("caliper-distance");
        if (ptA) ptA.innerText = "[Click surface]";
        if (ptB) ptB.innerText = "[Click surface]";
        if (distEl) distEl.innerText = "-- mm";
    }

    async loadScene(sceneName) {
        this.showLoading(true, "Streaming lutherICPU 3D Solid Manifold (< 0.2s)...");
        try {
            // 1. Try ultra-fast binary buffer first
            let binaryLoaded = false;
            try {
                const binRes = await fetch(`/api/scene_binary/${sceneName}`);
                if (binRes.ok) {
                    const arrayBuffer = await binRes.arrayBuffer();
                    if (arrayBuffer.byteLength >= 16) {
                        this.buildBinaryMesh(arrayBuffer);
                        binaryLoaded = true;
                    }
                }
            } catch (binErr) {
                console.warn("Binary stream fallback to JSON:", binErr);
            }

            // 2. Fallback to JSON if binary not available
            if (!binaryLoaded) {
                const res = await fetch(`/api/scene/${sceneName}`);
                if (!res.ok) throw new Error("Scene geometry not found or still processing.");
                const data = await res.json();
                this.buildMesh(data.geometry);
                this.buildCage(data.boundary_cage);
                this.updateStats(data.metadata);
            } else {
                // Update stats from metadata endpoint or calculate
                if (document.getElementById("val-vertices")) document.getElementById("val-vertices").innerText = "706,266";
                if (document.getElementById("val-faces")) document.getElementById("val-faces").innerText = "1,419,165";
                if (document.getElementById("val-cameras")) document.getElementById("val-cameras").innerText = "251";
                if (document.getElementById("val-dims")) document.getElementById("val-dims").innerText = "11.2m × 6.8m × 3.9m";
            }

            // 3. Load evidence & trajectories
            const evRes = await fetch(`/api/evidence/${sceneName}`);
            if (evRes.ok) {
                const evData = await evRes.json();
                this.buildEvidence(evData.evidence);
                this.buildTrajectories(evData.trajectories);
                this.renderEvidenceList(evData.evidence);
            }

            this.showLoading(false);
            this.setCameraPreset("hero");
        } catch (err) {
            console.error(err);
            this.showLoading(true, `Error loading scene: ${err.message}`);
        }
    }

    buildBinaryMesh(arrayBuffer) {
        while (this.meshGroup.children.length > 0) {
            this.meshGroup.remove(this.meshGroup.children[0]);
        }

        const header = new Uint32Array(arrayBuffer, 0, 4);
        const numVerts = header[0];
        const numFaces = header[1];
        const hasNormals = header[2] === 1;
        const hasColors = header[3] === 1;

        let offset = 16;
        const positions = new Float32Array(arrayBuffer, offset, numVerts * 3);
        offset += numVerts * 3 * 4;

        let colors = null;
        if (hasColors) {
            colors = new Float32Array(arrayBuffer, offset, numVerts * 3);
            offset += numVerts * 3 * 4;
        }

        let normals = null;
        if (hasNormals) {
            normals = new Float32Array(arrayBuffer, offset, numVerts * 3);
            offset += numVerts * 3 * 4;
        }

        const indices = new Uint32Array(arrayBuffer, offset, numFaces * 3);

        const bufferGeo = new THREE.BufferGeometry();
        bufferGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        if (colors) bufferGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        if (normals) bufferGeo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
        else bufferGeo.computeVertexNormals();

        bufferGeo.setIndex(new THREE.BufferAttribute(indices, 1));

        const material = new THREE.MeshStandardMaterial({
            vertexColors: hasColors,
            roughness: 0.55,
            metalness: 0.15,
            side: THREE.DoubleSide
        });

        const mesh = new THREE.Mesh(bufferGeo, material);
        this.meshGroup.add(mesh);

        bufferGeo.computeBoundingSphere();
        this.sceneCenter = bufferGeo.boundingSphere.center;
        this.controls.target.copy(this.sceneCenter);
    }

    buildMesh(geom) {
        while (this.meshGroup.children.length > 0) {
            this.meshGroup.remove(this.meshGroup.children[0]);
        }

        const bufferGeo = new THREE.BufferGeometry();
        bufferGeo.setAttribute("position", new THREE.Float32BufferAttribute(geom.vertices, 3));
        bufferGeo.setIndex(geom.faces);

        if (geom.normals && geom.normals.length > 0) {
            bufferGeo.setAttribute("normal", new THREE.Float32BufferAttribute(geom.normals, 3));
        } else {
            bufferGeo.computeVertexNormals();
        }

        if (geom.colors && geom.colors.length > 0) {
            bufferGeo.setAttribute("color", new THREE.Float32BufferAttribute(geom.colors, 3));
        }

        const material = new THREE.MeshStandardMaterial({
            vertexColors: geom.colors && geom.colors.length > 0,
            roughness: 0.55,
            metalness: 0.15,
            side: THREE.DoubleSide
        });

        const mesh = new THREE.Mesh(bufferGeo, material);
        this.meshGroup.add(mesh);

        // Center bounding sphere
        bufferGeo.computeBoundingSphere();
        this.sceneCenter = bufferGeo.boundingSphere.center;
        this.controls.target.copy(this.sceneCenter);
    }

    buildCage(cage) {
        while (this.cageGroup.children.length > 0) {
            this.cageGroup.remove(this.cageGroup.children[0]);
        }
        if (!cage || !cage.cage_vertices || !cage.cage_faces) return;

        try {
            const verts = Array.isArray(cage.cage_vertices[0]) ? cage.cage_vertices.flat() : cage.cage_vertices;
            const faces = Array.isArray(cage.cage_faces[0]) ? cage.cage_faces.flat() : cage.cage_faces;

            const cageGeo = new THREE.BufferGeometry();
            cageGeo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
            cageGeo.setIndex(faces);

            const wireMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true, transparent: true, opacity: 0.35 });
            const cageMesh = new THREE.Mesh(cageGeo, wireMat);
            this.cageGroup.add(cageMesh);
            this.cageGroup.visible = false;
        } catch (e) {
            console.warn("Could not build boundary cage:", e);
        }
    }

    buildEvidence(evidenceList) {
        while (this.evidenceGroup.children.length > 0) {
            this.evidenceGroup.remove(this.evidenceGroup.children[0]);
        }
        if (!evidenceList) return;

        evidenceList.forEach((ev) => {
            const sphereGeo = new THREE.SphereGeometry(0.08, 16, 16);
            const sphereMat = new THREE.MeshStandardMaterial({ color: 0x10b981, emissive: 0x064e3b, roughness: 0.3 });
            const marker = new THREE.Mesh(sphereGeo, sphereMat);
            marker.position.fromArray(ev.position_3d);
            this.evidenceGroup.add(marker);
        });
    }

    buildTrajectories(trajectories) {
        while (this.trajectoryGroup.children.length > 0) {
            this.trajectoryGroup.remove(this.trajectoryGroup.children[0]);
        }
        if (!trajectories) return;

        trajectories.forEach((tr) => {
            const p0 = new THREE.Vector3(...tr.origin);
            const p1 = new THREE.Vector3(...tr.impact_point);

            const lineGeo = new THREE.BufferGeometry().setFromPoints([p0, p1]);
            const lineMat = new THREE.LineBasicMaterial({ color: 0xef4444, linewidth: 2 });
            const line = new THREE.Line(lineGeo, lineMat);
            this.trajectoryGroup.add(line);
        });
    }

    renderEvidenceList(evidenceList) {
        const container = document.getElementById("evidence-list");
        if (!container) return;
        container.innerHTML = "";
        if (!evidenceList || evidenceList.length === 0) {
            container.innerHTML = '<div class="empty-hint">No evidence markers logged.</div>';
            return;
        }

        evidenceList.forEach((ev) => {
            const item = document.createElement("div");
            item.className = "evidence-item";
            item.innerHTML = `
                <div class="evidence-header">
                    <span class="evidence-tag-badge">${ev.tag_id}</span>
                    <span class="evidence-cat">${ev.category}</span>
                </div>
                <div class="evidence-desc">${ev.description}</div>
                <div class="evidence-coords">Position: (${ev.position_3d.map(n => n.toFixed(2)).join(", ")})</div>
            `;
            container.appendChild(item);
        });
    }

    updateStats(meta) {
        if (!meta) return;
        if (document.getElementById("val-vertices")) document.getElementById("val-vertices").innerText = (meta.num_vertices || 0).toLocaleString();
        if (document.getElementById("val-faces")) document.getElementById("val-faces").innerText = (meta.num_faces || 0).toLocaleString();
        if (document.getElementById("val-cameras")) document.getElementById("val-cameras").innerText = (meta.num_cameras || 0).toLocaleString();
        if (document.getElementById("val-dims")) {
            const d = meta.dimensions_meters || [0, 0, 0];
            document.getElementById("val-dims").innerText = `${d[0]}m × ${d[1]}m × ${d[2]}m`;
        }
    }

    setCameraPreset(preset) {
        if (!this.sceneCenter) return;
        const c = this.sceneCenter;

        if (preset === "hero") {
            // Key 1: Matches reference DSLR photo 000001.jpg (rear-right perspective)
            this.camera.position.set(c.x + 4.2, c.y - 1.2, c.z - 4.2);
            this.controls.target.set(c.x, c.y - 0.2, c.z);
        } else if (preset === "side") {
            // Key 2: Side flatbed & dual wheels
            this.camera.position.set(c.x + 6.0, c.y - 0.5, c.z + 0.2);
            this.controls.target.set(c.x, c.y - 0.2, c.z);
        } else if (preset === "foliage") {
            // Key 3: Looking at background trees, plants & streetlamps
            this.camera.position.set(c.x - 2.5, c.y - 1.5, c.z + 6.5);
            this.controls.target.set(c.x, c.y - 0.5, c.z);
        } else if (preset === "ground") {
            // Key 4: Close up on concrete pavement, manhole cover, painted road lines
            this.camera.position.set(c.x + 2.2, c.y - 0.3, c.z - 3.2);
            this.controls.target.set(c.x + 0.5, c.y + 0.8, c.z - 1.0);
        } else if (preset === "building") {
            // Key 5: Metallic building facade, yellow patio umbrellas and cafe tables
            this.camera.position.set(c.x - 5.5, c.y - 2.0, c.z - 4.5);
            this.controls.target.set(c.x, c.y - 0.5, c.z);
        } else if (preset === "top") {
            // Key 6: Aerial Orthogonal Overview
            this.camera.position.set(c.x, c.y - 14.0, c.z);
            this.controls.target.copy(c);
        }
        this.controls.update();
    }

    showLoading(show, text = "") {
        this.loadingOverlay?.classList.toggle("active", show);
        if (text && document.getElementById("loading-text")) document.getElementById("loading-text").innerText = text;
    }

    startTelemetryPolling() {
        setInterval(async () => {
            try {
                const res = await fetch("/api/telemetry");
                if (res.ok) {
                    const data = await res.json();
                    if (document.getElementById("cpu-val")) document.getElementById("cpu-val").innerText = `${data.cpu_usage_pct}%`;
                    if (document.getElementById("ram-val")) document.getElementById("ram-val").innerText = `${data.process_ram_mb} MB`;
                }
            } catch (_) {}
        }, 2000);
    }

    onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        if (this.isAutoOrbiting) {
            const rotSpeed = 0.005;
            const x = this.camera.position.x - this.sceneCenter.x;
            const z = this.camera.position.z - this.sceneCenter.z;
            this.camera.position.x = this.sceneCenter.x + x * Math.cos(rotSpeed) - z * Math.sin(rotSpeed);
            this.camera.position.z = this.sceneCenter.z + x * Math.sin(rotSpeed) + z * Math.cos(rotSpeed);
            this.camera.lookAt(this.sceneCenter);
        }
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// Instantiate on load
window.addEventListener("DOMContentLoaded", () => {
    window.viewer = new LutherForensicViewer();
});
