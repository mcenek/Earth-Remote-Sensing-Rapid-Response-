/* Read-only comparison of the original team's archived RGBA overlay and saved Model6 evidence. */
(() => {
  'use strict';

  const ARCHIVE_ID = 'GTM_Model0_capstone_archive_20260327';
  const PORTABLE_ARCHIVE_ID = 'GTM_Model0_capstone_archive_portable';
  const DEFAULT_MODEL_ID = 'GTM_Model6_mapper_mars_pilot_20260922_r5_mean';
  const PORTABLE_MODEL_ID = 'GTM_Model6_mapper_r5_portable';
  const initialSelection = new URLSearchParams(window.location.search);
  const byId = (id) => document.getElementById(id);
  const state = {
    registry: null,
    archive: null,
    models: [],
    currentExperiment: null,
    currentScene: null,
    renderId: 0,
    maps: {},
    layers: { archive: [], model: [] },
    contextLayers: [],
  };

  function setStatus(id, message, visible = true) {
    const node = byId(id);
    node.textContent = message;
    node.hidden = !visible;
  }

  function assetUrl(path) {
    if (typeof path !== 'string' || !path) throw new Error('Required local asset path is missing.');
    return new URL(path.replace(/^\//, ''), window.location.href).href;
  }

  function showError(panel, error) {
    console.error(error);
    const message = error instanceof Error ? error.message : String(error);
    setStatus(`${panel}-state`, `Could not load this saved evidence: ${message}`);
  }

  function removeLayers(panel) {
    const map = state.maps[panel];
    state.layers[panel].forEach((layer) => {
      if (map.hasLayer(layer)) map.removeLayer(layer);
    });
    state.layers[panel] = [];
  }

  function addImage(panel, path, bounds, opacity = 1) {
    if (!Array.isArray(bounds) || bounds.length !== 2) throw new Error('Scene has no geographic image bounds.');
    const layer = L.imageOverlay(assetUrl(path), bounds, { opacity, interactive: false, crossOrigin: false });
    layer.addTo(state.maps[panel]);
    state.layers[panel].push(layer);
    return layer;
  }

  function setContextEnabled(enabled) {
    state.contextLayers.forEach((layer) => {
      state.maps.archive.removeLayer(layer);
      state.maps.model.removeLayer(layer);
    });
    state.contextLayers = [];
    if (!enabled) return;
    const tiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors',
    });
    const secondTiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors',
    });
    tiles.addTo(state.maps.archive);
    secondTiles.addTo(state.maps.model);
    state.contextLayers = [tiles, secondTiles];
    if (state.archive?.scenes?.[0]?.bounds) state.maps.archive.fitBounds(state.archive.scenes[0].bounds, { padding: [18, 18] });
    if (state.currentScene?.bounds) state.maps.model.fitBounds(state.currentScene.bounds, { padding: [18, 18] });
  }

  function safeText(value, fallback = 'Not recorded') {
    return value === undefined || value === null || value === '' ? fallback : String(value);
  }

  function addArchiveProvenance(experiment, scene) {
    const body = byId('archive-provenance');
    const meta = experiment.archiveProvenance || experiment.provenance || {};
    const rows = [
      ['Archive date', '2026-03-27 saved commit date; image acquisition date is not recorded here.'],
      ['Saved output', `${safeText(scene?.predictionImage, 'Archived prediction image is not registered.')}${scene?.predictionHash ? ` · SHA-256 ${scene.predictionHash}` : ''}`],
      ['Source artifact', safeText(scene?.source, 'Original prediction source artifact is not recorded.')],
      ['Source revision', safeText(meta.sourceCommit || experiment.sourceCommit, 'ffbd96f41f9fae454bf21bc8536a9c3e05711321')],
      ['Checkpoint receipt', safeText(scene?.checkpointHash, 'Saved run checkpoint unrecorded')],
      ['Checkpoint context', safeText(meta.checkpointAttribution || experiment.checkpointAttribution, 'The archived source revision points to checkpoint.weights.h5. This is source-level attribution; the output has no per-run checkpoint record.')],
      ['Rendering', safeText(meta.rendering || experiment.rendering, 'Stored RGBA red overlay: clipped mask, red channel scaled by 2, alpha only above 0.3. Raw probabilities cannot be recovered from this rendering.')],
      ['Evidence missing', safeText(meta.limits || experiment.limits, 'No source RGB, raw probabilities, matched truth, or shared native grid is included in the archive.')],
    ];
    body.replaceChildren(...rows.map(([label, value]) => {
      const p = document.createElement('p');
      const strong = document.createElement('strong');
      strong.textContent = `${label}: `;
      const text = document.createTextNode(value);
      p.append(strong, text);
      return p;
    }));
  }

  function clearMapForRender(panel, message) {
    state.renderId += 1;
    removeLayers(panel);
    setStatus(`${panel}-state`, message);
  }

  async function loadArchive() {
    const experiment = state.registry.experiments.find((item) => item.id === ARCHIVE_ID) ||
      state.registry.experiments.find((item) => item.id === PORTABLE_ARCHIVE_ID);
    if (!experiment) {
      byId('archive-title').textContent = 'Archived prediction overlay unavailable';
      byId('archive-subtitle').textContent = 'The archive bundle is not present in the local experiment registry.';
      addArchiveProvenance({}, null);
      setStatus('archive-state', 'Add the GTM_Model0_capstone_archive_20260327 bundle to the viewer’s local bundle directory.');
      return;
    }
    state.archive = experiment;
    const scene = (experiment.scenes || []).find((item) => item.predictionImage && item.bounds);
    byId('archive-title').textContent = experiment.name || 'Archived capstone overlay';
    byId('archive-subtitle').textContent = scene?.name || 'Archive commit dated 2026-03-27 · acquisition date not recorded';
    addArchiveProvenance(experiment, scene);
    byId('archive-viewer-link').href = `index.html?experiment=${encodeURIComponent(experiment.id)}&scene=${encodeURIComponent(scene?.id || '')}#compare`;
    if (!scene) {
      setStatus('archive-state', 'The archive bundle has no georeferenced prediction image and bounds.');
      return;
    }
    clearMapForRender('archive', 'Loading the saved fixed-color overlay…');
    try {
      const image = await new Promise((resolve, reject) => {
        const node = new Image();
        node.onload = () => resolve(node);
        node.onerror = () => reject(new Error('Archive PNG could not be read from the local bundle.'));
        node.src = assetUrl(scene.predictionImage);
      });
      if (!image.naturalWidth || !image.naturalHeight) throw new Error('Archive PNG has no image dimensions.');
      addImage('archive', scene.predictionImage, scene.bounds);
      state.maps.archive.fitBounds(scene.bounds, { padding: [20, 20] });
      setStatus('archive-state', '', false);
    } catch (error) {
      showError('archive', error);
    }
  }

  function modelExperiments(experiments) {
    return experiments.filter((experiment) => {
      if (!experiment.id?.startsWith('GTM_Model6') || experiment.lifecycle === 'archived') return false;
      return (experiment.scenes || []).some((scene) => scene.bounds && scene.rgb && scene.grid && scene.task === 'plume_segmentation');
    });
  }

  function scenesFor(experiment) {
    return (experiment?.scenes || []).filter((scene) => scene.bounds && scene.rgb && scene.grid && scene.task === 'plume_segmentation');
  }

  function populateExperiments() {
    const select = byId('experiment-select');
    select.replaceChildren();
    state.models.forEach((experiment) => {
      const option = document.createElement('option');
      option.value = experiment.id;
      option.textContent = `${experiment.name || experiment.id}${experiment.status ? ` · ${experiment.status}` : ''}`;
      select.append(option);
    });
    select.disabled = state.models.length === 0;
    if (!state.models.length) throw new Error('No non-archived, georeferenced Model6 plume scenes are available.');
    const defaultExperiment = state.models.find((experiment) => experiment.id === initialSelection.get('experiment')) ||
      state.models.find((experiment) => experiment.id === DEFAULT_MODEL_ID) ||
      state.models.find((experiment) => experiment.id === PORTABLE_MODEL_ID) || state.models[0];
    select.value = defaultExperiment.id;
    populateScenes(defaultExperiment);
  }

  function populateScenes(experiment) {
    const previousSceneId = state.currentScene?.id || initialSelection.get('scene');
    clearMapForRender('model', 'Loading selected Model6 scene…');
    state.currentExperiment = experiment;
    const scenes = scenesFor(experiment);
    const select = byId('scene-select');
    select.replaceChildren();
    scenes.forEach((scene) => {
      const option = document.createElement('option');
      option.value = scene.id;
      option.textContent = scene.name || scene.id;
      select.append(option);
    });
    select.disabled = scenes.length === 0;
    if (!scenes.length) throw new Error(`No georeferenced Model6 plume scenes found in ${experiment.name || experiment.id}.`);
    const scene = scenes.find((item) => item.id === previousSceneId) || scenes[0];
    select.value = scene.id;
    updateRunStatus(experiment);
    renderModelScene(scene);
  }

  function updateRunStatus(experiment) {
    const badge = byId('run-status');
    const status = safeText(experiment.status, safeText(experiment.lifecycle, 'status not recorded'));
    badge.textContent = `RUN STATUS · ${status}`;
    badge.className = `badge ${/fail|reject|not promoted/i.test(status) ? 'failed' : /pass|promot/i.test(status) ? 'other' : ''}`;
  }

  function validateGrid(grid, scene) {
    if (!Number.isInteger(grid.width) || !Number.isInteger(grid.height) || grid.width < 1 || grid.height < 1) throw new Error('Saved grid has invalid dimensions.');
    const count = grid.width * grid.height;
    if (!Number.isSafeInteger(count) || count > 1048576) throw new Error('Saved grid exceeds the 1,048,576-cell viewer limit.');
    for (const key of ['probability', 'truth', 'valid']) {
      if (!Array.isArray(grid[key]) || grid[key].length !== count) throw new Error(`Saved grid ${key} array does not match its dimensions.`);
    }
    if (grid.crs !== 'EPSG:3857') throw new Error(`Unsupported saved display grid: ${safeText(grid.crs)}.`);
    if (!scene.bounds) throw new Error('Scene geographic bounds are unavailable.');
    for (let index = 0; index < count; index += 1) {
      const probability = grid.probability[index];
      const truth = grid.truth[index];
      const valid = grid.valid[index];
      if (typeof probability !== 'number' || !Number.isFinite(probability) || probability < 0 || probability > 1) {
        throw new Error(`Saved probability value at cell ${index} is invalid.`);
      }
      if (truth !== null && truth !== 0 && truth !== 1 && truth !== false && truth !== true) {
        throw new Error(`Saved truth value at cell ${index} is invalid.`);
      }
      if (valid !== 0 && valid !== 1 && valid !== false && valid !== true) {
        throw new Error(`Saved support value at cell ${index} is invalid.`);
      }
    }
  }

  function buildOverlay(grid, threshold) {
    const canvas = document.createElement('canvas');
    canvas.width = grid.width;
    canvas.height = grid.height;
    const context = canvas.getContext('2d', { willReadFrequently: false });
    const data = context.createImageData(grid.width, grid.height);
    const width = grid.width;
    const height = grid.height;
    const supported = (index) => (grid.valid[index] === 1 || grid.valid[index] === true) && Number.isFinite(grid.probability[index]);
    const positive = (index) => grid.truth[index] === 1 || grid.truth[index] === true;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        if (!supported(index)) continue;
        const offset = index * 4;
        if (grid.probability[index] >= threshold) {
          data.data[offset] = 245;
          data.data[offset + 1] = 132;
          data.data[offset + 2] = 42;
          data.data[offset + 3] = 148;
        }
        if (positive(index)) {
          const edge = x === 0 || y === 0 || x === width - 1 || y === height - 1 ||
            !positive(index - 1) || !positive(index + 1) ||
            !positive(index - width) || !positive(index + width);
          if (edge) {
            data.data[offset] = 46;
            data.data[offset + 1] = 235;
            data.data[offset + 2] = 231;
            data.data[offset + 3] = 244;
          }
        }
      }
    }
    context.putImageData(data, 0, 0);
    return canvas.toDataURL('image/png');
  }

  function showMetrics(scene, experiment) {
    const root = byId('scene-metrics');
    root.replaceChildren();
    const metrics = scene.nativeMetrics;
    if (!metrics || typeof metrics !== 'object') {
      const note = document.createElement('p');
      note.className = 'scene-notes';
      note.textContent = 'No native scene metrics are attached to this saved record.';
      root.append(note);
      return;
    }
    const values = [
      ['IoU', metrics.iou],
      ['Recall', metrics.recall],
      ['Precision', metrics.precision],
      ['Pixels', metrics.evaluated_pixels],
    ].filter(([, value]) => Number.isFinite(value));
    values.forEach(([label, value]) => {
      const card = document.createElement('div');
      card.className = 'metric';
      const caption = document.createElement('span');
      caption.textContent = label;
      const number = document.createElement('strong');
      number.textContent = label === 'Pixels' ? Number(value).toLocaleString() : Number(value).toFixed(3);
      card.append(caption, number);
      root.append(card);
    });
    const note = document.createElement('p');
    note.className = 'scene-notes';
    note.style.flexBasis = '100%';
    note.textContent = `${experiment.name || 'Model6'} native saved-scene metrics. These were measured on Model6’s scene and do not compare against the capstone archive.`;
    root.append(note);
  }

  function updateViewerLink(experiment, scene) {
    byId('model-viewer-link').href = `index.html?experiment=${encodeURIComponent(experiment.id)}&scene=${encodeURIComponent(scene.id)}#compare`;
    const url = new URL(window.location.href);
    url.searchParams.set('experiment', experiment.id);
    url.searchParams.set('scene', scene.id);
    window.history.replaceState(null, '', url);
  }

  async function renderModelScene(scene) {
    state.currentScene = scene;
    clearMapForRender('model', 'Changing scene…');
    const renderId = state.renderId;
    const experiment = state.currentExperiment;
    const threshold = Number(scene.decisionThreshold ?? experiment.decisionThreshold ?? 0.5);
    const location = safeText(scene.location, 'Geographic location not recorded');
    byId('scene-heading').textContent = scene.name || scene.id;
    const referenceNote = scene.positiveOnlyReference
      ? `${safeText(scene.truthLabel, 'Positive-only reference')} · ${safeText(scene.truthNote, 'exterior is unknown')}`
      : safeText(scene.truthLabel, 'saved reference');
    byId('scene-notes').textContent = `${location} · prediction thresholded at saved cutoff ${threshold.toFixed(2)} · ${referenceNote}`;
    showMetrics(scene, experiment);
    updateViewerLink(experiment, scene);
    try {
      const [gridResponse, rgbImage] = await Promise.all([
        fetch(assetUrl(scene.grid), { cache: 'no-store' }),
        new Promise((resolve, reject) => {
          const image = new Image();
          image.onload = () => resolve(image);
          image.onerror = () => reject(new Error('Model6 source RGB image could not be loaded.'));
          image.src = assetUrl(scene.rgb);
        }),
      ]);
      if (!gridResponse.ok) throw new Error(`Model6 grid request returned ${gridResponse.status}.`);
      const grid = await gridResponse.json();
      validateGrid(grid, scene);
      if (renderId !== state.renderId) return;
      const overlayUrl = buildOverlay(grid, threshold);
      if (!rgbImage.naturalWidth || !rgbImage.naturalHeight) throw new Error('Model6 source RGB image has no dimensions.');
      addImage('model', scene.rgb, scene.bounds);
      const overlay = L.imageOverlay(overlayUrl, scene.bounds, { opacity: 1, interactive: false, crossOrigin: false });
      overlay.addTo(state.maps.model);
      state.layers.model.push(overlay);
      state.maps.model.fitBounds(scene.bounds, { padding: [20, 20] });
      setStatus('model-state', '', false);
    } catch (error) {
      if (renderId === state.renderId) showError('model', error);
    }
  }

  async function start() {
    if (!window.L) throw new Error('Local Leaflet files are missing.');
    state.maps.archive = L.map('archive-map', { zoomControl: true, attributionControl: true, minZoom: 2, zoomSnap: 0.25 }).setView([39, -98], 4);
    state.maps.model = L.map('model-map', { zoomControl: true, attributionControl: true, minZoom: 2, zoomSnap: 0.25 }).setView([39, -98], 4);
    L.control.scale({ metric: true, imperial: false, position: 'bottomleft', maxWidth: 110 }).addTo(state.maps.archive);
    L.control.scale({ metric: true, imperial: false, position: 'bottomleft', maxWidth: 110 }).addTo(state.maps.model);
    byId('map-context').addEventListener('change', (event) => setContextEnabled(event.target.checked));
    byId('experiment-select').addEventListener('change', (event) => {
      const experiment = state.models.find((item) => item.id === event.target.value);
      if (experiment) {
        try { populateScenes(experiment); }
        catch (error) { showError('model', error); }
      }
    });
    byId('scene-select').addEventListener('change', (event) => {
      const scene = scenesFor(state.currentExperiment).find((item) => item.id === event.target.value);
      if (scene) renderModelScene(scene);
    });

    const response = await fetch('api/experiments', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Experiment registry returned ${response.status}.`);
    state.registry = await response.json();
    state.models = modelExperiments(state.registry.experiments || []);
    await loadArchive();
    populateExperiments();
  }

  start().catch((error) => {
    showError('model', error);
    if (!state.archive) setStatus('archive-state', 'The local experiment registry could not be loaded. Start the GTM local viewer launcher first.');
  });
})();
