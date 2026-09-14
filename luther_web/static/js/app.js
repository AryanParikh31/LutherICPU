/**
 * MicroSurfel 3D Dashboard Application Controller & Telemetry Handler
 */

let currentTab = 'setup';
let telemetrySocket = null;
let lossHistory = [];
let psnrHistory = [];

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));

  const targetPane = document.getElementById(`tab-${tabId}`);
  const targetBtn = document.getElementById(`tab-btn-${tabId}`);
  if (targetPane) targetPane.classList.add('active');
  if (targetBtn) targetBtn.classList.add('active');

  currentTab = tabId;

  if (tabId === 'viewport' && surfelViewer) {
    setTimeout(() => {
      surfelViewer.onResize();
      loadActiveScene();
    }, 50);
  } else if (tabId === 'gallery') {
    refreshProofs();
  }
}

// Knob value label sync
function updateKnobVal(type, val) {
  if (type === 'tau') {
    document.getElementById('val-tau').textContent = parseFloat(val).toFixed(5);
  } else if (type === 'sh') {
    const names = ['Degree 0 (Ambient DC)', 'Degree 1 (12 coeffs)', 'Degree 2 (27 coeffs)', 'Degree 3 (48 coeffs)'];
    document.getElementById('val-sh').textContent = names[parseInt(val)] || `Degree ${val}`;
  } else if (type === 'aa') {
    document.getElementById('val-aa').textContent = `${parseFloat(val).toFixed(2)} px`;
  }
}

// Viewport slider & mode proxies
function updateExposure(val) {
  document.getElementById('val-exposure').textContent = parseFloat(val).toFixed(2);
  if (surfelViewer) surfelViewer.setExposure(val);
}
function updateFOV(val) {
  document.getElementById('val-fov').textContent = `${val}°`;
  if (surfelViewer) surfelViewer.setFOV(val);
}
function updateDotScale(val) {
  document.getElementById('val-scale').textContent = `${parseFloat(val).toFixed(1)}x`;
  if (surfelViewer) surfelViewer.setDotScale(val);
}
function changeRenderMode(mode) {
  if (surfelViewer) surfelViewer.setRenderMode(mode);
}
function toggleCinematicOrbit() {
  if (surfelViewer) surfelViewer.toggleCinematicOrbit();
}
function toggleBoundaryCage(show) {
  if (surfelViewer) surfelViewer.toggleCage(show);
}
function toggleCameraClamping(active) {
  if (surfelViewer) surfelViewer.toggleClamping(active);
}
function resetCameraView() {
  if (surfelViewer) surfelViewer.resetView();
}
function takeSnapshot() {
  if (surfelViewer) surfelViewer.takeSnapshot();
}
function toggleRecordVideo() {
  if (!surfelViewer) return;
  const btn = document.getElementById('btn-record-video');
  if (!surfelViewer.isRecording) {
    surfelViewer.startRecording();
    btn.textContent = '⏹️ Stop Recording';
    btn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
  } else {
    surfelViewer.stopRecording();
    btn.textContent = '🔴 Record Video';
    btn.style.background = '';
  }
}
function toggleHud() {
  const body = document.getElementById('hud-body');
  body.style.display = body.style.display === 'none' ? 'block' : 'none';
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  fetchHardwareProfile();
  setupDropzone();
  checkExistingImages();
  connectWebSocket();
});

// 1. Hardware Profiler Telemetry Fetch
async function fetchHardwareProfile() {
  try {
    const res = await fetch('/api/profile');
    if (!res.ok) return;
    const p = await res.json();

    document.getElementById('spec-cpu').textContent = `${p.cpu_physical_cores} Cores @ ${p.cpu_freq_mhz} MHz`;
    document.getElementById('spec-cpu-threads').textContent = `${p.cpu_total_cores} Logical Threads`;
    document.getElementById('spec-ram').textContent = `${p.total_ram_gb} GB Total`;
    document.getElementById('spec-ram-avail').textContent = `${p.available_ram_gb} GB Available`;
    document.getElementById('spec-ceiling').textContent = `${p.ram_ceiling_gb} GB Hard Limit`;
    document.getElementById('spec-gpu').textContent = p.gpu_name || 'CPU Engine (Safe Mode)';
    document.getElementById('spec-vram').textContent = p.gpu_name ? `${p.gpu_vram_gb} GB VRAM` : 'NumPy / SIMD Vectorized';

    const modeBadge = document.getElementById('hardware-mode-badge');
    const badgeText = document.getElementById('badge-text');
    const banner = document.getElementById('notification-banner');
    const bannerText = document.getElementById('banner-text');

    if (p.mode === 'PROFILE_A_GPU') {
      modeBadge.textContent = 'GPU HIGH PERFORMANCE';
      badgeText.textContent = 'GPU CUDA Ready';
      banner.className = 'notification-banner banner-gpu';
      bannerText.textContent = p.notification_banner;
    } else {
      modeBadge.textContent = 'CPU SAFE MODE';
      badgeText.textContent = 'Safe CPU Ready';
      banner.className = 'notification-banner banner-safe';
      bannerText.textContent = p.notification_banner;
    }

    document.getElementById('gauge-max-surfels').textContent = `Cap: ${p.max_surfels.toLocaleString()}`;
    document.getElementById('gauge-ram-ceiling').textContent = `Hard Ceiling: ${Math.round(p.ram_ceiling_gb * 1024)} MB`;

  } catch (err) {
    console.error('Failed to fetch hardware profile:', err);
  }
}

// 2. Dropzone & File Uploads
function setupDropzone() {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadFiles(files);
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) uploadFiles(e.target.files);
  });
}

async function uploadFiles(files) {
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Upload failed');
    const data = await res.json();
    checkExistingImages();
  } catch (err) {
    alert(`Upload error: ${err.message}`);
  }
}

async function checkExistingImages() {
  try {
    const res = await fetch('/api/input-images');
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById('image-count-badge').textContent = `${data.count} photos loaded`;
    const gallery = document.getElementById('thumbnail-gallery');
    gallery.innerHTML = '';

    data.images.forEach(imgName => {
      const item = document.createElement('div');
      item.className = 'thumbnail-item';
      const img = document.createElement('img');
      img.src = `/input/${imgName}?t=${Date.now()}`;
      img.alt = imgName;
      img.onerror = () => {
        item.style.background = '#1e293b';
        item.innerHTML = `<div style="font-size:0.65rem;padding:0.5rem;color:#94a3b8;word-break:break-all;">${imgName}</div>`;
      };
      item.appendChild(img);
      gallery.appendChild(item);
    });
  } catch (err) {
    console.error('Error fetching images:', err);
  }
}

// 3. Start Reconstruction Simulation
async function startReconstruction() {
  const tau = parseFloat(document.getElementById('slider-tau').value);
  const sh = parseInt(document.getElementById('slider-sh').value);
  const aa = parseFloat(document.getElementById('slider-aa').value);

  const btn = document.getElementById('btn-start-training');
  btn.disabled = true;
  btn.textContent = '⏳ Initializing...';

  try {
    const res = await fetch('/api/start-training', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        densify_grad_threshold: tau,
        sh_degree: sh,
        anti_aliasing_filter: aa,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start');
    }

    switchTab('monitor');
    document.getElementById('live-pill').style.display = 'inline-block';
    appendLog('[START] Reconstruction job launched. Streaming live telemetry.');
  } catch (err) {
    alert(`Could not start reconstruction: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = '🚀 Reconstruct & Simulate';
  }
}

// 4. Dispatch 1-Click Cloud Training
async function dispatchCloudTraining() {
  try {
    appendLog('[CLOUD] Packaging dataset payload for 1-Click Google Colab GPU training...');
    const res = await fetch('/api/cloud-train', { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Cloud dispatch failed');
    }
    const data = await res.json();
    alert(`✨ 1-Click Cloud GPU Payload Ready!\n\nPackage: ${data.package_path} (${data.package_size_mb} MB)\nTarget: ${data.recommended_gpu}\n\nOpening train_cloud.ipynb instructions.`);
    appendLog(`[CLOUD] Dataset packaged (${data.package_size_mb} MB). Direct upload to Google Colab ready.`);
  } catch (err) {
    alert(`Cloud Training Dispatch: ${err.message}`);
  }
}

// 5. WebSocket Progress Telemetry
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/progress`;

  telemetrySocket = new WebSocket(wsUrl);

  telemetrySocket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'SCENE_RELOAD') {
        appendLog(`[HOT-RELOAD] ${data.message || 'New 30,000-step model detected! Reloading 3D viewport...'}`);
        loadActiveScene();
      } else {
        updateTelemetryUI(data);
      }
    } catch (e) {
      console.error('WebSocket parse error:', e);
    }
  };

  telemetrySocket.onclose = () => {
    setTimeout(connectWebSocket, 2000);
  };
}

function updateTelemetryUI(data) {
  const badge = document.getElementById('monitor-phase-badge');
  badge.textContent = data.status;
  document.getElementById('monitor-subtext').textContent = data.notification;

  const pct = data.percentage || 0;
  document.getElementById('progress-fill').style.width = `${pct}%`;
  document.getElementById('progress-pct-text').textContent = `${pct.toFixed(1)}%`;
  document.getElementById('progress-step-text').textContent = `Step ${data.step} / ${data.total_steps}`;

  document.getElementById('gauge-loss').textContent = data.loss.toFixed(4);
  document.getElementById('gauge-psnr').textContent = `${data.psnr.toFixed(2)} dB`;
  document.getElementById('gauge-surfels').textContent = data.surfel_count.toLocaleString();
  document.getElementById('gauge-ram').textContent = `${Math.round(data.memory_used_mb)} MB`;
  document.getElementById('gauge-eta').textContent = data.eta_seconds > 0 ? `${Math.round(data.eta_seconds)}s` : '--';

  if (data.history_loss && data.history_loss.length > 0) {
    lossHistory = data.history_loss;
    psnrHistory = data.history_psnr;
    renderCharts();
  }

  if (data.status === 'COMPLETED') {
    document.getElementById('live-pill').style.display = 'none';
    badge.textContent = 'COMPLETED';
    badge.style.background = 'rgba(16, 185, 129, 0.2)';
    badge.style.color = '#10b981';
    badge.style.borderColor = 'rgba(16, 185, 129, 0.5)';
  }
}

// 6. Canvas Chart Rendering
function renderCharts() {
  drawCurve('loss-canvas', lossHistory, '#00f0ff', true);
  drawCurve('psnr-canvas', psnrHistory, '#a855f7', false);
}

function drawCurve(canvasId, values, color, isMin) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || values.length < 2) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let y = 30; y < h; y += 35) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const range = (maxVal - minVal) || 1e-4;

  const padding = 20;
  const plotH = h - padding * 2;

  ctx.beginPath();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = color;

  for (let i = 0; i < values.length; i++) {
    const x = (i / (values.length - 1)) * (w - 40) + 20;
    const normalized = (values[i] - minVal) / range;
    const y = isMin ? (h - padding - normalized * plotH) : (h - padding - normalized * plotH);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  const lastVal = values[values.length - 1];
  ctx.fillStyle = color;
  ctx.font = '11px JetBrains Mono';
  ctx.fillText(lastVal.toFixed(3), w - 55, padding);
}

function appendLog(text) {
  const box = document.getElementById('terminal-log');
  if (!box) return;
  const line = document.createElement('div');
  line.className = 'terminal-line';
  const time = new Date().toLocaleTimeString();
  line.innerHTML = `<span style="color:#64748b;">[${time}]</span> ${text}`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

// 7. Load Active Scene into 3D Viewport
async function loadActiveScene() {
  if (!surfelViewer) return;
  try {
    let loaded = false;
    // Try fast binary PLY first
    try {
      const plyRes = await fetch('/api/scene/default/ply');
      if (plyRes.ok) {
        const buffer = await plyRes.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        const headerStr = new TextDecoder().decode(bytes.subarray(0, 1024));
        const match = headerStr.match(/element vertex (\d+)/);
        const endHeaderIdx = headerStr.indexOf("end_header\n");

        if (match && endHeaderIdx !== -1) {
          const count = parseInt(match[1], 10);
          const dataOffset = endHeaderIdx + "end_header\n".length;
          const dataView = new DataView(buffer, dataOffset);

          const rawPos = new Float32Array(count * 3);
          const rawCols = new Float32Array(count * 3);
          const rawQuats = [];
          const rawScales = [];

          const isEnhanced = headerStr.includes("scale_x") && headerStr.includes("rot_w");
          const stride = isEnhanced ? 55 : 15;

          for (let i = 0; i < count; i++) {
            const byteOffset = i * stride;
            rawPos[i * 3] = dataView.getFloat32(byteOffset, true);
            rawPos[i * 3 + 1] = dataView.getFloat32(byteOffset + 4, true);
            rawPos[i * 3 + 2] = dataView.getFloat32(byteOffset + 8, true);

            if (isEnhanced) {
              rawScales.push(
                dataView.getFloat32(byteOffset + 24, true),
                dataView.getFloat32(byteOffset + 28, true),
                dataView.getFloat32(byteOffset + 32, true)
              );
              rawQuats.push(
                dataView.getFloat32(byteOffset + 36, true),
                dataView.getFloat32(byteOffset + 40, true),
                dataView.getFloat32(byteOffset + 44, true),
                dataView.getFloat32(byteOffset + 48, true)
              );
              rawCols[i * 3] = dataView.getUint8(byteOffset + 52) / 255.0;
              rawCols[i * 3 + 1] = dataView.getUint8(byteOffset + 53) / 255.0;
              rawCols[i * 3 + 2] = dataView.getUint8(byteOffset + 54) / 255.0;
            } else {
              rawCols[i * 3] = dataView.getUint8(byteOffset + 12) / 255.0;
              rawCols[i * 3 + 1] = dataView.getUint8(byteOffset + 13) / 255.0;
              rawCols[i * 3 + 2] = dataView.getUint8(byteOffset + 14) / 255.0;
            }
          }

          surfelViewer.loadSurfelData({
            count,
            positions: rawPos,
            colors: rawCols,
            quaternions: rawQuats,
            scales: rawScales,
          });
          loaded = true;
        }
      }
    } catch (e) {
      console.warn("PLY stream failed, trying JSON fallback:", e);
    }

    if (!loaded) {
      const res = await fetch('/api/scene/default/json');
      if (!res.ok) return;
      const sceneData = await res.json();
      surfelViewer.loadSurfelData(sceneData);
    }

    const cageRes = await fetch('/api/scene/default/cage');
    if (cageRes.ok) {
      const cageData = await cageRes.json();
      surfelViewer.loadBoundaryCage(cageData);
    }
  } catch (err) {
    console.error('Failed to load scene into 3D viewer:', err);
  }
}

// 8. Refresh Simulation Proofs Gallery
async function refreshProofs() {
  try {
    const res = await fetch('/api/renders');
    if (!res.ok) return;
    const data = await res.json();
    console.log(`[GALLERY] Found ${data.count} proof renders.`);
  } catch (e) {
    console.warn('Proof refresh:', e);
  }
}

// 9. Load Sample Dataset
async function loadSampleDataset() {
  appendLog('[DATASET] Generating multi-view room dataset in input/...');
  const btn = document.querySelector('.btn-secondary');
  if (btn) btn.textContent = '⏳ Preparing Images...';

  try {
    const res = await fetch('/api/generate-sample-dataset', { method: 'POST' });
    if (res.ok) {
      checkExistingImages();
      appendLog('[DATASET] Multi-angle room images loaded successfully.');
    }
  } catch (e) {
    console.warn('Dataset route fallback:', e);
  } finally {
    if (btn) btn.textContent = '✨ Load Sample Dataset';
  }
}
