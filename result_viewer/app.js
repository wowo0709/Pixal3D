import './vendor/model-viewer.min.js';

const DEFAULT_ORBIT = '-25deg 70deg 2.1m';
const DEFAULT_TARGET = '0m 0m 0m';
const DEFAULT_FOV = '35deg';
const results = document.querySelector('#results');
const sync = document.querySelector('#sync-camera');
const dialog = document.querySelector('#lightbox');
const lightboxImage = document.querySelector('#lightbox-image');
let viewers = [];
let activeViewer = null;
let mode = '3d';
let catalog;

function node(tag, className, text) {
  const element = document.createElement(tag);
  element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function showImage(url, title) {
  document.querySelector('#lightbox-title').textContent = title;
  lightboxImage.src = url;
  lightboxImage.alt = title;
  lightboxImage.classList.remove('actual-size');
  dialog.showModal();
}

document.querySelector('#lightbox-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target === dialog) dialog.close();
});
lightboxImage.addEventListener('click', () => lightboxImage.classList.toggle('actual-size'));

function copyCamera(source) {
  const orbit = source.getCameraOrbit();
  const target = source.getCameraTarget();
  const fov = source.getFieldOfView();
  for (const viewer of viewers) {
    if (viewer === source) continue;
    viewer.cameraOrbit = `${orbit.theta}rad ${orbit.phi}rad ${orbit.radius}m`;
    viewer.cameraTarget = `${target.x}m ${target.y}m ${target.z}m`;
    viewer.fieldOfView = `${fov}deg`;
    viewer.jumpCameraToGoal();
  }
}

function updateLoadCount() {
  const ready = results.querySelectorAll('[data-state="ready"]').length;
  const failed = results.querySelectorAll('[data-state="error"]').length;
  document.querySelector('#load-count').textContent = `${ready} / ${viewers.length} models loaded${failed ? ` · ${failed} failed` : ''}`;
}

function setMode(next) {
  mode = next;
  for (const button of document.querySelectorAll('[data-mode]')) {
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  }
  for (const card of results.querySelectorAll('.result-card')) {
    card.querySelector('.interactive-pane').hidden = mode !== '3d';
    const pane = card.querySelector('.render-pane');
    pane.hidden = mode === '3d';
    if (mode !== '3d') {
      const image = pane.querySelector('img');
      image.src = card.result.files[mode];
      image.alt = `${card.result.label} ${mode === 'normal' ? 'normal' : 'base-color'} renders, four directions`;
    }
  }
}

for (const button of document.querySelectorAll('[data-mode]')) {
  button.addEventListener('click', () => setMode(button.dataset.mode));
}
sync.addEventListener('change', () => {
  if (sync.checked && activeViewer?.loaded) copyCamera(activeViewer);
});
document.querySelector('#reset-camera').addEventListener('click', () => {
  for (const viewer of viewers) {
    viewer.cameraOrbit = DEFAULT_ORBIT;
    viewer.cameraTarget = DEFAULT_TARGET;
    viewer.fieldOfView = DEFAULT_FOV;
    viewer.jumpCameraToGoal();
  }
});

const stageNames = { ss64: 'SS64', shape512: 'Shape512', shape1024: 'Shape1024', pbr1024: 'PBR1024' };

function createCard(result, index) {
  const card = node('article', 'result-card');
  card.dataset.state = 'loading';
  card.dataset.resultId = result.id;
  card.result = result;
  card.innerHTML = `
    <header class="card-heading"><span class="result-number"></span><div><h2></h2><p class="weight-caption"></p></div><span class="view-badge"></span></header>
    <div class="input-section"><div class="section-caption"><span>ACTUAL INPUTS</span><span>Click to enlarge</span></div><div class="input-strip"></div></div>
    <div class="output-section">
      <div class="interactive-pane"><model-viewer camera-controls disable-tap interaction-prompt="none" touch-action="pan-y" environment-image="neutral" exposure="1.3" shadow-intensity="0" min-camera-orbit="auto 5deg 0.3m" max-camera-orbit="auto 175deg 5m" min-field-of-view="20deg" max-field-of-view="60deg"><span slot="pan-target" class="pan-target-marker" aria-hidden="true"></span></model-viewer><span class="load-status" aria-live="polite">Loading 3D…</span><div class="viewport-label">EXPORTED GLB <span>Drag to explore</span></div></div>
      <div class="render-pane" hidden><button type="button"><img alt=""><span>Enlarge four-direction render ↗</span></button></div>
    </div>
    <div class="result-stats"></div>
    <details><summary>Run details</summary><div class="run-details"></div></details>
    <footer class="card-footer"><a class="manifest-download" download>Manifest ↗</a><a class="glb-download" download>Download GLB ↓</a></footer>`;
  card.querySelector('.result-number').textContent = String(index + 1).padStart(2, '0');
  card.querySelector('h2').textContent = result.label;
  card.querySelector('.weight-caption').textContent = result.training_steps ?
    `SV initialization · ${result.training_steps}-step multi-view fine-tune` : 'Official single-view pretrained weights';
  card.querySelector('.view-badge').textContent = `${result.inputs.length} ${result.inputs.length === 1 ? 'view' : 'views'}`;
  const inputs = card.querySelector('.input-strip');
  for (const input of result.inputs) {
    const button = node('button', 'input-thumb');
    button.type = 'button';
    button.setAttribute('aria-label', `${result.label}: enlarge input view ${input.view_index}`);
    const image = document.createElement('img');
    image.src = input.url;
    image.alt = `Input view ${input.view_index}`;
    button.append(image, node('span', '', `V${String(input.view_index).padStart(2, '0')}`));
    button.addEventListener('click', () => showImage(input.url, `${result.label} · Input view ${input.view_index}`));
    inputs.append(button);
  }
  const viewer = card.querySelector('model-viewer');
  // o_voxel converts Z-up → Y-up, but these view-aligned meshes are already
  // Y-up. Restore their native anchor frame without changing the saved GLB.
  viewer.orientation = '0deg 90deg 0deg';
  viewer.alt = `${result.label} generated textured 3D model. Drag to rotate, scroll to zoom.`;
  viewer.cameraOrbit = DEFAULT_ORBIT;
  viewer.cameraTarget = DEFAULT_TARGET;
  viewer.fieldOfView = DEFAULT_FOV;
  viewer.addEventListener('progress', event => {
    if (card.dataset.state === 'loading') card.querySelector('.load-status').textContent = `Loading 3D · ${Math.round(event.detail.totalProgress * 100)}%`;
  });
  viewer.addEventListener('load', () => {
    card.dataset.state = 'ready';
    card.querySelector('.load-status').textContent = '3D ready';
    if (sync.checked && activeViewer?.loaded) copyCamera(activeViewer);
    updateLoadCount();
  });
  viewer.addEventListener('error', event => {
    card.dataset.state = 'error';
    card.querySelector('.load-status').textContent = `3D loading failed (${event.detail.type}). Saved renders are available in the tabs.`;
    updateLoadCount();
  });
  viewer.addEventListener('camera-change', event => {
    if (event.detail.source !== 'user-interaction') return;
    activeViewer = viewer;
    if (sync.checked) copyCamera(viewer);
  });
  viewer.src = result.files.model;
  viewers.push(viewer);
  card.querySelector('.render-pane button').addEventListener('click', () =>
    showImage(result.files[mode], `${result.label} · ${mode === 'normal' ? 'Normals' : 'Base color'} · Four directions`));
  for (const [value, label] of [[result.training_steps, 'FINE-TUNE STEPS'], [result.resolution, 'RESOLUTION'], [result.seed, 'SEED']]) {
    const stat = node('div', 'stat');
    stat.append(node('span', '', label), node('strong', '', String(value)));
    card.querySelector('.result-stats').append(stat);
  }
  const details = card.querySelector('.run-details');
  details.append(node('p', '', `Pipeline: ${result.pipeline} · ${result.sampler_steps} sampling steps per stage`));
  details.append(node('p', '', `Mesh scale: ${result.mesh_scale.toFixed(6)} · Anchor: input view ${result.anchor_view}`));
  details.append(node('p', '', `Scale source: ${result.mesh_scale_source}`));
  for (const [stage, checkpoint] of Object.entries(result.checkpoints)) {
    const row = node('div', 'checkpoint-row');
    // Pretrained inference records checkpoint descriptors; the PoC records paths.
    const path = typeof checkpoint === 'string' ? checkpoint : checkpoint.path;
    row.append(node('span', '', stageNames[stage] || stage), node('code', '', path));
    details.append(row);
  }
  const download = card.querySelector('.glb-download');
  download.href = result.files.model;
  download.download = result.download_name;
  card.querySelector('.manifest-download').href = result.files.manifest;
  return card;
}

function showCase(id) {
  const selected = catalog.cases.find(item => item.id === id);
  viewers = [];
  activeViewer = null;
  results.replaceChildren();
  for (const [index, result] of selected.results.entries()) results.append(createCard(result, index));
  document.querySelector('#asset-label').textContent = `${selected.title} · ${selected.asset.slice(0, 12)}…`;
  const first = selected.results[0];
  document.querySelector('#common-settings').textContent = `${first.resolution} cascade · Seed ${first.seed} · Anchor view ${first.anchor_view}`;
  updateLoadCount();
  setMode(mode);
}

async function initialize() {
  const response = await fetch('/api/cases');
  if (!response.ok) throw new Error(`Could not load experiments (HTTP ${response.status})`);
  catalog = await response.json();
  const select = document.querySelector('#case-select');
  select.replaceChildren();
  for (const item of catalog.cases) {
    const option = node('option', '', item.title);
    option.value = item.id;
    select.append(option);
  }
  select.addEventListener('change', () => showCase(select.value));
  showCase(select.value);
}

initialize().catch(error => {
  const banner = document.querySelector('#page-error');
  banner.textContent = error.message;
  banner.hidden = false;
  console.error(error);
});
