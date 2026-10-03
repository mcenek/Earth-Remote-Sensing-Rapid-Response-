'use strict';
const $=id=>document.getElementById(id), esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const registry=window.GTM_DATA;let experiment,scene,grid=null,view='map',loadToken=0,syncing=false,profileRow=null,mapMode='geo';const maps={},layers={},markers=[],basemaps={},homeZoom={};
let gridRequest, gridLoading = false, gridError = '', renderGeneration = 0;
let assetStates = new Map();
let showArchive = new URLSearchParams(location.search).get('catalog') === 'all';
function presentationCase(entry) {
  return registry.presentation?.experiments?.find(item => item.id === entry?.id);
}
function experimentLabel(entry) { return presentationCase(entry)?.label || entry.name; }
function catalogExperiments() {
  if (showArchive) return registry.experiments;
  const selected = (registry.presentation?.experiments || []).map(item =>
    registry.experiments.find(entry => entry.id === item.id)).filter(entry => entry?.scenes.some(hasPred));
  return selected.length ? selected : registry.experiments.filter(entry => entry.scenes.some(hasPred));
}
function updateCatalogSummary() {
  const entries = catalogExperiments();
  $('catalog-mode').value = showArchive ? 'all' : 'presentation';
  text('catalog-title', showArchive ? 'Full research archive' : 'Presentation examples');
  text('catalog-note', showArchive ? entries.length + ' runs, including diagnostics and input-only records' :
    entries.reduce((n, entry) => n + entry.scenes.length, 0) + ' saved prediction scenes. Selected overlaps, misses and false alarms; models remain experimental.');
}
function startPresentation() {
  showArchive = false;
  for (const id of ['atlas-search', 'atlas-from', 'atlas-to']) $(id).value = '';
  $('atlas-filter').value = $('atlas-experiment').value = 'all';
  for (const id of ['pred-toggle', 'truth-toggle', 'atlas-prediction', 'atlas-reference']) $(id).checked = true;
  $('error-toggle').checked = false;
  const entry = catalogExperiments().find(item => item.id === registry.activeExperimentId) || catalogExperiments()[0];
  populateExperiments();
  selectExperiment(entry.id, presentationCase(entry)?.sceneId);
  atlas.refresh();
  if (atlas.map) { atlas.filter(); atlas.needsInitialFit = true; }
  switchView('compare');
  $('layout-overlay').click();
}
const cursors = {};
function updateCursors(latlng) {
  for (const id of ['original-map', 'prediction-map', 'truth-map']) {
    if (!cursors[id]) cursors[id] = L.marker(latlng, {interactive: false, keyboard: false,
      icon: L.divIcon({className: 'shared-cursor', iconSize: [24, 24], iconAnchor: [12, 12]})}).addTo(maps[id]);
    else cursors[id].setLatLng(latlng);
  }
}
function clearCursors() {
  for (const id of Object.keys(cursors)) { cursors[id].remove(); delete cursors[id]; }
}

function allowedAsset(value) {
  if (!value) return null;
  if (typeof value !== 'string') throw Error('Asset must be a string');
  if (/^data:image\/(png|jpeg|webp);base64,/.test(value)) return value;
  const decoded = decodeURIComponent(value);
  if (!decoded.startsWith('/') && !/[\\:#?\x00-\x1f]/.test(decoded) &&
      !decoded.split('/').some(part => !part || part === '.' || part === '..')) return value;
  throw Error('Assets must be local relative paths or embedded PNG/JPEG/WebP images');
}

function assetUrl(value) {
  const path = allowedAsset(value);
  if (!path || path.startsWith('data:')) return path;
  const url = new URL(path, document.baseURI);
  if (experiment?.live?.revision) url.searchParams.set('v', experiment.live.revision);
  return url.href;
}

function validateExperiment(e) {
  if (!e || typeof e.id !== 'string' || typeof e.name !== 'string' || !Array.isArray(e.scenes))
    throw Error('Need experiment id, name and scenes');
  if (e.reportUrl) allowedAsset(e.reportUrl);
  if (e.scenes.length > 1000) throw Error('Bundle exceeds 1,000 scenes');
  const ids = new Set();
  for (const s of e.scenes) {
    if (!s || typeof s.id !== 'string' || typeof s.name !== 'string') throw Error('Invalid scene');
    if (ids.has(s.id)) throw Error('Duplicate scene IDs');
    ids.add(s.id);
    if (s.coordinateSystem === 'pixel') {
      const size = s.imageSize || [32, 32];
      if (!Array.isArray(size) || size.length !== 2 || size.some(n => !Number.isInteger(n) || n < 1))
        throw Error('Invalid image dimensions');
      s.bounds = [[0, 0], size];
    }
    if (!Array.isArray(s.bounds) || s.bounds.length !== 2 ||
        s.bounds.some(b => !Array.isArray(b) || b.length !== 2 || b.some(v => !Number.isFinite(v))) ||
        s.bounds[0][0] >= s.bounds[1][0] || s.bounds[0][1] >= s.bounds[1][1] ||
        (s.coordinateSystem !== 'pixel' && s.bounds.some(b => Math.abs(b[0]) > 85 || Math.abs(b[1]) > 180)))
      throw Error('Invalid scene bounds');
    for (const key of ['rgb', 'emit', 'truthImage', 'predictionImage', 'probabilityImage',
      'disagreementImage', 'observedImage', 'coarseImage', 'residualImage', 'reference90Rgb', 'reference365Rgb'])
      allowedAsset(s[key]);
    if (typeof s.grid === 'string') {
      allowedAsset(s.grid);
      if (!s.grid.endsWith('.json')) throw Error('Invalid grid path');
    } else if (s.grid) validateGrid(s.grid);
  }
  return e;
}
function validateGrid(g){const n=g.width*g.height;if(!Number.isInteger(g.width)||!Number.isInteger(g.height)||g.width<1||g.height<1||n<1||n>1048576)throw Error('Invalid display grid size');const reconstruction=g.task==='methane_reconstruction';const keys=reconstruction?['observed','coarse','prediction','residual','valid']:['probability','truth','valid'];for(const key of keys){if(g[key]&&(g[key].length!==n||g[key].some(v=>v!==null&&!Number.isFinite(v))))throw Error('Invalid '+key+' grid');}if(reconstruction){if(typeof g.units!=='string'||!g.units.trim())throw Error('Reconstruction grid requires explicit units');if(!g.observed&&!g.coarse&&!g.prediction&&!g.residual)throw Error('Reconstruction grid has no continuous layers');}if(g.probability?.some(v=>v!==null&&(v<0||v>1)))throw Error('Probability outside [0,1]');if(g.truth?.some(v=>v!==null&&v!==0&&v!==1))throw Error('Truth mask must be 0/1/null');if(g.valid?.some(v=>v!==0&&v!==1))throw Error('Validity mask must be 0/1');return g;}
function hasPred(s){return !!(s.task==='methane_reconstruction'?(s.predictionAvailable||s.predictionImage||typeof s.grid==='object'&&s.grid.prediction):(s.predictionImage||s.predictionAvailable||typeof s.grid==='object'&&s.grid.probability));}function isReconstruction(){return scene?.task==='methane_reconstruction'||grid?.task==='methane_reconstruction';}function text(id,v){$(id).textContent=v??'—';}
function initMap(id){const pixel=mapMode==='pixel';const m=L.map(id,{crs:pixel?L.CRS.Simple:L.CRS.EPSG3857,zoomControl:false,zoomSnap:0.25,zoomDelta:0.5,maxZoom:pixel?8:20,minZoom:pixel?-6:3}).setView(pixel?[16,16]:[39,-98],pixel?2:4);L.control.zoom({position:'topright'}).addTo(m);if(!pixel){L.control.scale({position:'bottomleft',imperial:false}).addTo(m);basemaps[id]=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Geographic context © Esri and contributors'});if($('online-basemap').checked)basemaps[id].addTo(m);}layers[id]=[];maps[id]=m;m.on('mousemove',e=>inspectPixel(e.latlng,false));m.on('click',e=>inspectPixel(e.latlng,true));m.on('mouseout',clearCursors);if(id!=='main-map')m.on('moveend',()=>{if(syncing||!$('linked').checked)return;syncing=true;for(const other of ['original-map','prediction-map','truth-map'])if(other!==id&&maps[other])maps[other].setView(m.getCenter(),m.getZoom()+(homeZoom[other]??m.getZoom())-(homeZoom[id]??m.getZoom()),{animate:false});syncing=false;});return m;}
function ensureMapMode(next){if(mapMode===next)return;for(const m of Object.values(maps))m.remove();for(const k of Object.keys(basemaps))delete basemaps[k];markers.length=0;mapMode=next;['main-map','original-map','prediction-map','truth-map'].forEach(initMap);}
function inspectPixel(latlng,select){if(!grid||!scene)return;const crs=maps['main-map'].options.crs,sw=crs.project(L.latLng(scene.bounds[0])),ne=crs.project(L.latLng(scene.bounds[1])),pt=crs.project(latlng),x=Math.floor((pt.x-sw.x)/(ne.x-sw.x)*grid.width),y=Math.floor((ne.y-pt.y)/(ne.y-sw.y)*grid.height);if(x<0||y<0||x>=grid.width||y>=grid.height)return;updateCursors(latlng);const i=y*grid.width+x,valid=!grid.valid||grid.valid[i];if(isReconstruction()){const u=grid.units||'units unknown',fmt=k=>valid&&grid[k]?.[i]!=null?grid[k][i].toFixed(3):'unknown',msg='Pixel '+x+', '+y+' · observed '+fmt('observed')+' · coarse '+fmt('coarse')+' · reconstructed '+fmt('prediction')+' · residual '+fmt('residual')+' '+u;text('pixel-inspector',msg);text('compare-pixel-inspector',msg);}else{const p=valid?grid.probability?.[i]:null,t=valid?grid.truth?.[i]:null;text('pixel-inspector','Pixel '+x+', '+y+' · p '+(p==null?'unknown':p.toFixed(4))+' · reference '+(t==null?'unknown':t===1?'positive':'negative'));text('compare-pixel-inspector',$('pixel-inspector').textContent);}if(select){profileRow=y;drawPlots();}}
function probabilityColor(p){const stops=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],v=Math.min(3.99999,Math.max(0,p)*4),i=Math.floor(v),f=v-i;return stops[i].map((a,k)=>Math.round(a+(stops[i+1][k]-a)*f));}
function reconstructionScale(){
 if(reconstructionRangeCache)return reconstructionRangeCache;
 let lo=Infinity,hi=-Infinity;
 for(const key of ['observed','coarse','prediction'])for(const [i,v] of (grid?.[key]||[]).entries())if(v!=null&&(!grid.valid||grid.valid[i])){lo=Math.min(lo,v);hi=Math.max(hi,v);}
 lo=Number.isFinite(grid?.display?.min)?grid.display.min:Number.isFinite(lo)?lo:0;
 hi=Number.isFinite(grid?.display?.max)?grid.display.max:Number.isFinite(hi)?hi:1;
 return reconstructionRangeCache=[lo,hi>lo?hi:lo+1];
}
function reconstructionColor(v,kind){
 if(kind==='residual'){const m=grid?.display?.residualMax||1000,q=Math.max(-1,Math.min(1,v/m));return q<0?[Math.round(255*(1+q)),Math.round(255*(1+q)),255]:[255,Math.round(255*(1-q)),Math.round(255*(1-q))];}
 const [lo,hi]=reconstructionScale();return probabilityColor(Math.max(0,Math.min(1,(v-lo)/(hi-lo))));
}
function makeReconstructionLayer(kind){
 if(!grid?.[kind])return null;
 const c=document.createElement('canvas');c.width=grid.width;c.height=grid.height;const ctx=c.getContext('2d'),im=ctx.createImageData(c.width,c.height);
 for(let i=0;i<c.width*c.height;i++){if(grid.valid&&!grid.valid[i]||grid[kind][i]==null)continue;im.data.set([...reconstructionColor(grid[kind][i],kind),255],i*4);}
 ctx.putImageData(im,0,0);return c.toDataURL();
}

function showAssetStatus() {
  const states = [...assetStates.values()];
  const failed = states.filter(s => s === 'error').length;
  const pending = states.filter(s => s === 'loading').length;
  let message = !scene ? 'No scene selected' : gridLoading ? 'Loading scene data…' :
    gridError ? gridError : failed ? failed + ' image layer(s) could not load. Refresh to retry.' :
    pending ? 'Loading imagery…' : states.length ? 'Scene imagery loaded' : 'No image layers available';
  text('asset-status', message);
  $('asset-status').dataset.state = gridError || failed ? 'error' : gridLoading || pending ? 'loading' : 'ready';
  $('data-status').hidden = !(gridLoading || gridError || failed);
  $('data-status').dataset.state = gridError || failed ? 'error' : 'loading';
  text('data-status', message);
}

function overlay(id, url, opacity = 1) {
  if (!url || !scene) return;
  const generation = renderGeneration, key = id + ':' + url;
  const layer = L.imageOverlay(assetUrl(url), scene.bounds, {opacity, interactive: false});
  if (!url.startsWith('data:')) {
    assetStates.set(key, 'loading');
    for (const event of ['load', 'error']) layer.on(event, () => {
      if (generation !== renderGeneration) return;
      assetStates.set(key, event === 'load' ? 'ready' : 'error');
      showAssetStatus();
    });
  } else assetStates.set(key, 'ready');
  layers[id].push(layer.addTo(maps[id]));
  showAssetStatus();
}

function clearLayers() {
  clearCursors();
  renderGeneration++;
  assetStates = new Map();
  for (const id of Object.keys(maps)) {
    layers[id].forEach(layer => maps[id].removeLayer(layer));
    layers[id] = [];
  }
  showAssetStatus();
}
function fit(){if(!scene)return;syncing=true;try{for(const [id,m] of Object.entries(maps)){m.fitBounds(scene.bounds,{padding:[16,16],maxZoom:mapMode==='pixel'?8:17,animate:false});homeZoom[id]=m.getZoom();}}finally{syncing=false;}}
function updateURL(){const u=new URL(location.href);if(experiment)u.searchParams.set('experiment',experiment.id);if(scene)u.searchParams.set('scene',scene.id);else u.searchParams.delete('scene');if(showArchive)u.searchParams.set('catalog','all');else u.searchParams.delete('catalog');u.hash=view;history.replaceState(null,'',u);}
function switchView(next, push = false) {
  const previous = location.href;
  const names = ['national', 'map', 'compare', 'imagery', 'experiments'];
  view = names.includes(next) ? next : 'national';
  document.body.dataset.view = view;
  for (const name of names) $(name + '-view').hidden = name !== view;
  document.querySelectorAll('[data-view]').forEach(button => {
    button.classList.toggle('selected', button.dataset.view === view);
    button.setAttribute('aria-pressed', String(button.dataset.view === view));
  });
  updateURL();
  if (push && previous !== location.href) {
    const destination = location.href;
    history.replaceState(null, '', previous);
    history.pushState(null, '', destination);
  }
  setTimeout(() => {
    Object.values(maps).forEach(map => map.invalidateSize());
    if (scene && ['map', 'compare'].includes(view)) fit();
    atlas.resize();
  }, 0);
}
function experimentContext(e){const a=e.nativeAggregate;return experimentLabel(e)+' · '+e.status+(a&&e.metricScope==='reused_development_fold'?' · Run-wide IoU '+(a.iou*100).toFixed(2)+'%, precision '+(a.precision*100).toFixed(2)+'% ('+a.scenes+' scenes)':'');}

function updateRunSummary() {
  const run = experiment.run;
  const count = experiment.scenes.length;
  const label = run ? run.state.charAt(0).toUpperCase() + run.state.slice(1) :
    experiment.lifecycle === 'archived' ? 'Archived evidence' : 'Saved evidence';
  $('run-progress').hidden = !run;
  text('run-progress', run ? label + (run.progress ? ' · ' + run.progress.completed + ' / ' + run.progress.total + ' ' + run.progress.unit : '') : '');
  $('run-progress').title = run?.message || '';
  $('run-summary').innerHTML = '<strong>' + esc(label) + '<span>' + count + ' scenes</span></strong>' +
    (run?.progress ? '<progress max="' + Math.max(1, run.progress.total) + '" value="' + run.progress.completed +
      '" aria-label="Export progress"></progress><small>' + run.progress.completed + ' / ' + run.progress.total +
      ' ' + esc(run.progress.unit) + '</small>' : '') +
    (run?.message ? '<small>' + esc(run.message) + '</small>' : '');
  text('context', experimentContext(experiment));
}

function populateExperiments() {
  $('experiment').innerHTML = catalogExperiments().map(e => '<option value="' + esc(e.id) + '">' +
    (e.lifecycle === 'archived' ? '[Archived] ' : '') + esc(experimentLabel(e)) + '</option>').join('');
  if (experiment) $('experiment').value = experiment.id;
  updateCatalogSummary();
}

function selectExperiment(id, requestedScene) {
  const previousScene = requestedScene || scene?.id;
  experiment = registry.experiments.find(e => e.id === id) ||
    catalogExperiments().find(e => e.id === registry.activeExperimentId) || catalogExperiments()[0] || registry.experiments[0];
  if (!catalogExperiments().some(e => e.id === experiment.id)) { showArchive = true; populateExperiments(); }
  $('experiment').value = experiment.id;
  $('search').value = '';
  $('filter').value = 'all';
  updateRunSummary();
  selectScene(experiment.scenes.find(s => s.id === previousScene)?.id ||
    experiment.scenes.find(s => s.id === presentationCase(experiment)?.sceneId)?.id ||
    experiment.scenes.find(hasPred)?.id || experiment.scenes[0]?.id);
  drawExperimentCards();
}

function drawList() {
  const query = $('search').value.toLowerCase(), filter = $('filter').value;
  const scenes = experiment.scenes.filter(s =>
    (s.name + ' ' + s.id + ' ' + (s.location || '')).toLowerCase().includes(query) &&
    (filter === 'all' || filter === 'predictions' && hasPred(s) ||
      filter === 'paired' && hasPred(s) && (s.truthImage || s.grid) ||
      filter === 'unscored' && !hasPred(s) || ['TP', 'FP', 'FN', 'TN'].includes(filter) && s.outcome === filter));
  const scroll = $('scenes').scrollTop;
  $('scenes').innerHTML = scenes.map(s => '<button class="scene ' + (scene?.id === s.id ? 'active' : '') +
    '" aria-pressed="' + (scene?.id === s.id) + '" data-scene="' + esc(s.id) + '">' +
    ((s.rgb || s.observedImage) ? '<img loading="lazy" src="' + esc(assetUrl(s.rgb || s.observedImage)) + '" alt="">' :
      '<span class="scene-placeholder" aria-hidden="true">◌</span>') + '<span><b>' + esc(s.name) + '</b><small>' +
    esc(s.location || s.id) + '</small><small>' + esc(s.date || 'Date not recorded') + ' · ' +
    (hasPred(s) ? 'Prediction' : 'Input only') + '</small></span></button>').join('');
  $('scenes').scrollTop = scroll;
  text('count', scenes.length + ' / ' + experiment.scenes.length);
  $('no-scenes').hidden = scenes.length > 0;
  document.querySelectorAll('[data-scene]').forEach(button => button.onclick = () => {
    selectScene(button.dataset.scene);
    document.body.classList.remove('scene-browser-open');
    $('scene-browser-toggle').setAttribute('aria-expanded', 'false');
  });
}

function sourceCard(title, url, description) {
  return '<div class="source">' + (url ? '<img src="' + esc(assetUrl(url)) + '" alt="' + esc(title) + '">' :
    '<div class="missing">—</div>') + '<span><strong>' + esc(title) + '</strong><small>' + esc(description) + '</small></span></div>';
}

function captureMapViews() {
  return Object.fromEntries(Object.entries(maps).map(([id, map]) => [id, {center: map.getCenter(), zoom: map.getZoom()}]));
}

function restoreMapViews(positions) {
  syncing = true;
  try {
    for (const [id, position] of Object.entries(positions)) maps[id].setView(position.center, position.zoom, {animate: false});
  } finally { syncing = false; }
}

async function selectScene(id, {preserveView = false} = {}) {
  const token = ++loadToken;
  gridRequest?.abort();
  gridRequest = new AbortController();
  const nextScene = experiment.scenes.find(s => s.id === id) || null;
  const retain = preserveView && scene?.id === nextScene?.id &&
    JSON.stringify(scene?.bounds) === JSON.stringify(nextScene?.bounds) && scene?.coordinateSystem === nextScene?.coordinateSystem;
  const positions = retain ? captureMapViews() : null;
  scene = nextScene;
  grid = null;
  gridError = '';
  gridLoading = !!scene?.grid;
  if (!retain) {
    profileRow = null;
    $('threshold').value = scene?.decisionThreshold ?? experiment.decisionThreshold ?? 0.5;
  }
  ensureMapMode(scene?.coordinateSystem === 'pixel' ? 'pixel' : 'geo');
  clearLayers();
  text('pixel-inspector', 'Hover a scene pixel to inspect values. Click to move the cross-section.');
  $('online-basemap').disabled = mapMode === 'pixel';
  $('overview').disabled = mapMode === 'pixel';
  text('overview', mapMode === 'pixel' ? 'Image coordinates only' : 'US overview');
  updateURL();
  drawList();
  markers.forEach(marker => maps['main-map'].removeLayer(marker));
  markers.length = 0;
  if (mapMode === 'geo') for (const item of experiment.scenes.filter(s => s.coordinateSystem !== 'pixel')) {
    const center = [(item.bounds[0][0] + item.bounds[1][0]) / 2, (item.bounds[0][1] + item.bounds[1][1]) / 2];
    markers.push(L.circleMarker(center, {radius: 4, color: '#fff', fillColor: '#276653', fillOpacity: 1, weight: 2})
      .addTo(maps['main-map']).bindTooltip(esc(item.name)).on('click', () => selectScene(item.id)));
  }
  $('map-empty').hidden = !!scene;
  text('scene-title', scene?.name || experiment.name);
  text('scene-subtitle', scene ? [scene.location, scene.date].filter(Boolean).join(' · ') : 'No scene export available');
  text('limitations', scene?.notes || experiment.notes || '');
  text('compare-notice', scene?.notes || experiment.notes || '');
  if (!scene) {
    text('map-empty', experiment.run ? 'Waiting for the first scene export. New results will appear here automatically.' :
      'No aligned imagery exported for this run. Open Runs to review its saved evidence.');
    $('source-cards').innerHTML = '';
    $('provenance').innerHTML = '';
    renderGrid();
    return;
  }
  $('source-cards').innerHTML = sourceCard('Target Sentinel-2', scene.rgb, scene.rgbNote || 'Observed input; display contrast only') +
    (scene.reference90Rgb ? sourceCard('Temporal reference −90d', scene.reference90Rgb, 'Reference window, not methane truth') : '') +
    (scene.reference365Rgb ? sourceCard('Temporal reference −365d', scene.reference365Rgb, 'Reference window, not methane truth') : '') +
    (scene.emit ? sourceCard(scene.emitLabel || 'EMIT reference', scene.emit, scene.emitNote || 'Paired observation') : '') +
    '<div id="reference-card">' + sourceCard(scene.truthLabel || 'Reference mask', scene.truthImage,
      scene.truthNote || 'No independent annotation available') + '</div>';
  const provenance = {Model: experiment.name, Scene: scene.id, Task: scene.task, Units: scene.units,
    Date: scene.date, Reference: scene.referenceDate, Grid: scene.resolution, Source: scene.source,
    Checkpoint: scene.checkpointHash, Split: scene.split,
    Coordinates: scene.coordinateSystem === 'pixel' ? 'Image pixels; footprint unavailable' : 'Georeferenced',
    Location: scene.locationPoint?.join(', '), 'Scene score': scene.sceneScore, Decision: scene.outcome, 'Prediction cache': scene.predictionHash};
  $('provenance').innerHTML = Object.entries(provenance).filter(([, value]) => value != null)
    .map(([key, value]) => '<dt>' + esc(key) + '</dt><dd>' + esc(value) + '</dd>').join('');

  // Clear the previous scene immediately, including its scores and charts.
  renderGrid();
  Object.values(maps).forEach(map => map.invalidateSize());
  if (positions) restoreMapViews(positions); else fit();
  if (scene.grid) {
    const controller = gridRequest;
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      let data = scene.grid;
      if (typeof data === 'string') {
        const response = await fetch(assetUrl(data), {cache: 'no-store', signal: controller.signal});
        if (!response.ok) throw Error('HTTP ' + response.status);
        data = await response.json();
      }
      const loaded = validateGrid(data);
      if (token !== loadToken) return;
      grid = loaded;
      if (scene.task === 'methane_reconstruction') grid.task = 'methane_reconstruction';
    } catch (error) {
      if (token !== loadToken) return;
      gridError = 'Pixel data unavailable (' + (error.name === 'AbortError' ? 'request timed out' : error.message) +
        '). Saved images may still be visible; interactive pixel scores are unavailable.';
    } finally { clearTimeout(timeout); }
  }
  if (token !== loadToken) return;
  gridLoading = false;
  // Do not reset the camera here: a user may already have panned while data loaded.
  renderGrid();
  showAssetStatus();
}
function makeLayer(kind){if(!grid)return null;const c=document.createElement('canvas');c.width=grid.width;c.height=grid.height;const ctx=c.getContext('2d'),im=ctx.createImageData(c.width,c.height),threshold=+$('threshold').value;for(let i=0;i<c.width*c.height;i++){if(grid.valid&&!grid.valid[i])continue;const p=grid.probability?.[i],t=grid.truth?.[i];let rgb=null,a=0;if(kind==='prediction'&&p!=null){if($('prediction-mode').value==='probability'){rgb=probabilityColor(p);a=240;}else if(p>=threshold){rgb=[255,135,35];a=230;}}if(kind==='truth'&&t===1){rgb=[0,225,230];const x=i%c.width,y=Math.floor(i/c.width);const edge=x===0||y===0||x===c.width-1||y===c.height-1||[i-1,i+1,i-c.width,i+c.width].some(j=>grid.truth[j]!==1);a=edge?240:30;}if(kind==='error'&&p!=null&&t!=null){if(p>=threshold&&t===1){rgb=[240,179,55];a=180;}else if(p>=threshold&&t===0){rgb=[247,71,122];a=215;}else if(p<threshold&&t===1){rgb=[152,121,240];a=215;}}if(rgb){im.data.set([...rgb,a],i*4);}}ctx.putImageData(im,0,0);return c.toDataURL();}
function renderGrid(){clearLayers();updateTaskUI();if(isReconstruction()){renderReconstruction();return;}const pred=grid?.probability?makeLayer('prediction'):scene?.predictionImage,truth=grid?.truth?makeLayer('truth'):(scene?.truthImage||scene?.emit),errors=grid?.probability&&grid?.truth?makeLayer('error'):scene?.disagreementImage;if($('reference-card'))$('reference-card').innerHTML=sourceCard(scene?.truthLabel||'Reference mask',truth,scene?.truthNote||'No reference annotation available');const alpha=+$('opacity').value,referenceAlpha=+$('reference-opacity').value;text('reference-opacity-value',Math.round(referenceAlpha*100)+'%');$('reference-opacity').disabled=!truth;syncComparisonControls();$('prediction-mode-compare').value=$('prediction-mode').value;text('prediction-status',pred?($('prediction-mode').value==='probability'?'Model score 0–1 · uncalibrated':'Mask ≥ '+(+$('threshold').value).toFixed(2)):'Prediction unavailable');text('reference-status',truth?(scene?.truthLabel||'Paired reference · not independent truth'):'Reference unavailable');for(const id of Object.keys(maps))if(scene)overlay(id,scene.rgb);
if(scene){
 for(const id of ['main-map','prediction-map']){
  if($('pred-toggle').checked)overlay(id,pred,alpha);
  if($('error-toggle').checked)overlay(id,errors,alpha);
  if($('truth-toggle').checked){overlay(id,truth,referenceAlpha);drawTruthBoundary(id,referenceAlpha);}
 }
 overlay('truth-map',truth,referenceAlpha);drawTruthBoundary('truth-map',referenceAlpha);for(const id of ['main-map','prediction-map'])if($('pred-toggle').checked)for(const p of scene.candidatePoints||[]){if(Number.isFinite(p.lat)&&Number.isFinite(p.lon))layers[id].push(L.circleMarker([p.lat,p.lon],{radius:5,color:'#ff8b23',fillColor:'#fff',fillOpacity:1,weight:2}).addTo(maps[id]).bindTooltip(esc(p.label)+' score '+Number(p.score).toFixed(3)));}
 if($('bounds-toggle').checked)for(const id of Object.keys(maps))layers[id].push(L.rectangle(scene.bounds,{color:'#fff',dashArray:'5,5',weight:1,fill:false}).addTo(maps[id]));
}
const support=atlas.supportImage();$('comparison-support').disabled=!support;if(support&&$('comparison-support').checked)for(const id of ['original-map','prediction-map','truth-map'])overlay(id,support);
for(const [id,on] of [['pred-toggle',!!pred],['truth-toggle',!!truth],['error-toggle',!!errors],['threshold',!!grid?.probability]]){$(id).disabled=!on;}$('reset-threshold').disabled=!grid?.probability;text('threshold-value',(+$('threshold').value).toFixed(2));if(scene&&pred)text('compare-notice',scene.notes||experiment.notes);if(scene&&!pred)text('compare-notice','Prediction unavailable for this scene. These maps show the available observations and reference layers. '+(scene.notes||''));let tp=0,fp=0,fn=0,tn=0,n=0;if(grid?.probability&&grid?.truth)for(let i=0;i<grid.width*grid.height;i++){if(grid.valid&&!grid.valid[i]||grid.probability[i]==null||grid.truth[i]==null)continue;n++;const p=grid.probability[i]>=+$('threshold').value,t=grid.truth[i]===1;if(p&&t)tp++;else if(p)fp++;else if(t)fn++;else tn++;}const positiveOnly=scene?.positiveOnlyReference===true;const vals=positiveOnly?{'Covered plume pixels':n?tp:null,'Missed plume pixels':n?fn:null,'Outside reference':'Unknown','False positives':'Not measured'}:{TP:n?tp:null,FP:n?fp:null,FN:n?fn:null,TN:n?tn:null};$('pixel-metrics').innerHTML=Object.entries(vals).map(([k,v])=>`<div class="metric"><span>${k}</span><strong>${v??'—'}</strong></div>`).join('');text('metric-note',positiveOnly?'EMIT positive support only. Exterior is unknown: precision, IoU and false-positive rate are not available. Coverage is not source verification.':n?`${n.toLocaleString()} jointly valid display pixels. IoU ${tp+fp+fn?(tp/(tp+fp+fn)).toFixed(3):'undefined (no positives)'}. Selected-crop agreement; not the aggregate held-out benchmark.`:'Pixel-level prediction and a valid reference mask are both required. No missing values are treated as negatives.');drawComparisonSummary({tp,fp,fn,tn,n});drawDiagnostics();drawPlots();atlas.renderSelected();}
function renderReconstruction(){
 reconstructionRangeCache=null;
 const u=grid?.units||scene?.units||'units unknown',alpha=+$('opacity').value,panels=!$('comparison-layout').classList.contains('overlay-layout');
 const observed=makeReconstructionLayer('observed')||scene?.observedImage,coarse=makeReconstructionLayer('coarse')||scene?.coarseImage,pred=makeReconstructionLayer('prediction')||scene?.predictionImage,residual=makeReconstructionLayer('residual')||scene?.residualImage;
 $('source-cards').innerHTML=(scene?.rgb?sourceCard('Sentinel-2 RGB context',scene.rgb,scene.rgbNote||'Display imagery; not a methane measurement'):'')+sourceCard(scene?.truthLabel||'Observed target',observed,scene?.truthNote||u)+sourceCard('Coarse input / bilinear baseline',coarse,u)+sourceCard('Actual reconstruction',pred,u)+sourceCard('Residual: prediction minus target',residual,u);
 for(const id of Object.keys(maps))if(scene?.rgb)overlay(id,scene.rgb);
 if(observed)overlay('original-map',observed,1);if(coarse)overlay('truth-map',coarse,1);
 for(const id of ['prediction-map','main-map']){const opacity=id==='prediction-map'&&panels?1:alpha;
  if(pred&&$('pred-toggle').checked)overlay(id,pred,opacity);
  if(observed&&$('truth-toggle').checked)overlay(id,observed,alpha);
  if(residual&&$('error-toggle').checked)overlay(id,residual,opacity);
 }
 for(const [id,value] of [['pred-toggle',pred],['truth-toggle',observed],['error-toggle',residual]])$(id).disabled=!value;
 syncComparisonControls();$('compare-opacity').disabled=panels;text('compare-opacity-value',panels?'100% panels':Math.round(alpha*100)+'%');for(const [a,b] of [['compare-pred','pred-toggle'],['compare-truth','truth-toggle'],['compare-errors','error-toggle']])$(a).disabled=$(b).disabled;
 text('prediction-status','Model output · '+u);text('reference-status',(scene?.coarseLabel||'Interpolated coarse input')+' · '+u);
 text('scene-flag','CONTINUOUS RECONSTRUCTION · '+u);text('compare-scene-label',experiment.name+' / '+(scene?.name||''));
 text('coordinate-note','Common display grid; quantitative run metrics use the original evaluation grid');
 text('map-diagnostics','Colored footprint = observed data coverage, not a learned plume outline. Unknown pixels remain blank.');
 text('diagnostics',scene?.notes||experiment.notes);text('compare-notice',scene?.notes||experiment.notes);$('compare-empty').hidden=!!scene;
 $('comparison-stats').innerHTML='';const [lo,hi]=reconstructionScale();
 document.querySelector('.compare-legend').textContent='Shared scale: '+lo+' → '+hi+' '+u+' · residual blue −'+(grid?.display?.residualMax||1000)+' / white 0 / red +'+(grid?.display?.residualMax||1000)+'. Footprint shows observed coverage, not plume detection.';
 document.querySelector('.map-legend').textContent='Continuous values: '+lo+' → '+hi+' '+u+'. Blank = unobserved.';
 const values=[];for(let i=0;i<(grid?.prediction?.length||0);i++)if(grid.valid?.[i]!==0&&grid.prediction[i]!=null&&grid.observed?.[i]!=null)values.push(grid.prediction[i]-grid.observed[i]);
 const n=values.length,mae=n?values.reduce((a,v)=>a+Math.abs(v),0)/n:null,rmse=n?Math.sqrt(values.reduce((a,v)=>a+v*v,0)/n):null;
 $('pixel-metrics').innerHTML=[['Observed pixels',grid?n:'—'],['MAE',mae?.toFixed(2)||'—'],['RMSE',rmse?.toFixed(2)||'—'],['Units',u]].map(([k,v])=>'<div class="metric"><span>'+esc(k)+'</span><strong>'+esc(v)+'</strong></div>').join('');
 text('metric-note','Display-grid comparison on observed support. See run report for native-grid scores.');drawPlots();atlas.renderSelected();
}
 function drawPlots(){if(isReconstruction()){drawReconstructionPlots();return;}for(const id of ['histogram','profile']){const c=$(id),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#69808e';ctx.font='13px Segoe UI';if(!grid?.probability){ctx.fillText('Prediction raster not available',16,35);continue;}ctx.strokeStyle='#d4e0e6';ctx.beginPath();ctx.moveTo(30,10);ctx.lineTo(30,c.height-25);ctx.lineTo(c.width-12,c.height-25);ctx.stroke();if(id==='histogram'){const bins=Array(20).fill(0);grid.probability.forEach((p,i)=>{if(p!=null&&(!grid.valid||grid.valid[i]))bins[Math.min(19,Math.floor(p*20))]++;});const max=Math.max(...bins,1);ctx.fillStyle='#e99d3c';bins.forEach((v,i)=>ctx.fillRect(35+i*(c.width-55)/20,c.height-26-v/max*(c.height-45),(c.width-55)/20-3,v/max*(c.height-45)));ctx.fillStyle='#69808e';ctx.fillText('0',30,c.height-7);ctx.fillText('1 probability',c.width-92,c.height-7);}else{const y=profileRow??Math.floor(grid.height/2);for(const [key,color] of [['probability','#e99632'],['truth','#009eaf']]){if(!grid[key])continue;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();let started=false;for(let x=0;x<grid.width;x++){const i=y*grid.width+x,v=grid[key][i];if(v==null||grid.valid&&!grid.valid[i]){started=false;continue;}const px=30+x/(grid.width-1)*(c.width-45),py=c.height-25-v*(c.height-40);if(!started)ctx.moveTo(px,py);else ctx.lineTo(px,py);started=true;}ctx.stroke();}ctx.fillStyle='#69808e';ctx.fillText('0',12,c.height-25);ctx.fillText('1',12,17);ctx.fillText('Pixel position →',c.width-115,c.height-5);}}}
function drawReconstructionPlots(){for(const id of ['histogram','profile']){const c=$(id),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#69808e';ctx.font='13px Segoe UI';if(!grid){ctx.fillText('Reconstruction raster not available',16,35);continue;}const y=profileRow??Math.floor(grid.height/2),keys=id==='histogram'?['observed','coarse','prediction','residual']:['observed','coarse','prediction','residual'],colors=['#3c83c7','#e99d3c','#00a39e','#d64f66'];if(id==='histogram'){const all=[];keys.forEach(k=>(grid[k]||[]).forEach((v,i)=>{if(v!=null&&(!grid.valid||grid.valid[i]))all.push(v);}));const [lo,hi]=reconstructionScale();const bins=Array(20).fill(0);all.forEach(v=>bins[Math.min(19,Math.max(0,Math.floor((v-lo)/(hi-lo)*20)))]++);const max=Math.max(...bins,1);ctx.fillStyle='#6b9fb0';bins.forEach((v,i)=>ctx.fillRect(35+i*(c.width-55)/20,c.height-26-v/max*(c.height-45),(c.width-55)/20-3,v/max*(c.height-45)));ctx.fillText(lo.toFixed(2),30,c.height-7);ctx.fillText(hi.toFixed(2)+' '+(grid.units||''),c.width-120,c.height-7);}else{const [lo,hi]=reconstructionScale();keys.forEach((key,j)=>{if(!grid[key])return;ctx.strokeStyle=colors[j];ctx.lineWidth=2;ctx.beginPath();let started=false;for(let x=0;x<grid.width;x++){const i=y*grid.width+x,v=grid[key][i];if(v==null||grid.valid&&!grid.valid[i]){started=false;continue;}const px=30+x/(grid.width-1)*(c.width-45),py=c.height-25-(v-lo)/(hi-lo)*(c.height-40);if(!started)ctx.moveTo(px,py);else ctx.lineTo(px,py);started=true;}ctx.stroke();});ctx.fillStyle='#69808e';ctx.fillText(lo.toFixed(2),12,c.height-25);ctx.fillText(hi.toFixed(2),12,17);ctx.fillText('Continuous value · '+(grid.units||'units unknown'),c.width-190,c.height-5);}}}
function drawExperimentCards(){$('experiments').innerHTML=catalogExperiments().map(e=>`<article class="experiment-card"><div class="eyebrow">${esc(e.status)}</div><h2>${esc(experimentLabel(e))}</h2><p>${esc(e.description||'')}</p>${e.metrics?`<table><thead><tr>${Object.keys(e.metrics).map(k=>`<th>${esc(k)}</th>`).join('')}</tr></thead><tbody><tr>${Object.values(e.metrics).map(v=>`<td>${esc(v)}</td>`).join('')}</tr></tbody></table>`:''}<p>${esc(e.notes)}</p><p class="readonly">${e.scenes.length} inspection scenes available · ${e.reportUrl?`<a href="${esc(allowedAsset(e.reportUrl))}">Evidence report ↗</a>`:esc(e.report||'Local export')}</p><button data-experiment="${esc(e.id)}">Open experiment</button></article>`).join('');document.querySelectorAll('[data-experiment]').forEach(b=>b.onclick=()=>{selectExperiment(b.dataset.experiment);switchView('map');});}

function drawDiagnostics() {
  if (scene?.outputKind === 'rendered_archive') { drawArchiveStatus(); return; }
  if (!grid?.probability) {
    const label = gridLoading ? 'Loading pixel data' : gridError ? 'Pixel data unavailable' :
      scene ? hasPred(scene) ? 'Saved image only · pixel scores unavailable' : 'Observation only · no model prediction' : 'No spatial export';
    text('scene-flag', label);
    text('diagnostics', gridError || scene?.notes || experiment.notes || label);
    text('map-diagnostics', label);
    return;
  }
  const values = grid.probability.filter((p, i) => p != null && (!grid.valid || grid.valid[i]));
  const n = values.length, positives = values.filter(p => p >= +$('threshold').value).length;
  const support = grid.truth?.filter((t, i) => t != null && (!grid.valid || grid.valid[i])).length || 0;
  let lo = 1, hi = 0;
  for (const value of values) { lo = Math.min(lo, value); hi = Math.max(hi, value); }
  const ratio = n ? positives / n : 0;
  const rule = Number.isFinite(scene.sceneScore) ? ' Scene score ' + scene.sceneScore.toFixed(4) +
    ' / frozen cutoff ' + Number(scene.sceneThreshold ?? experiment.sceneThreshold).toFixed(4) +
    ' · ' + (scene.outcome || 'separate scene decision') + '.' : '';
  const message = (ratio > .95 ? 'NEAR-ALL-POSITIVE MASK. ' : '') + (ratio * 100).toFixed(2) +
    '% of valid prediction pixels ≥ ' + (+$('threshold').value).toFixed(2) + '. Probability range ' +
    lo.toFixed(3) + '–' + hi.toFixed(3) + '. Reference coverage on this support ' +
    (n ? support / n * 100 : 0).toFixed(1) + '%.' + rule + ' Blank pixels are unscored.';
  text('scene-flag', (ratio > .95 ? 'MASK SATURATION · ' : 'PIXEL CHECK · ') + (ratio * 100).toFixed(1) +
    '% positive at threshold ' + (+$('threshold').value).toFixed(2) + (scene.outcome ? ' · Scene-level ' + scene.outcome : ''));
  text('diagnostics', message);
  text('map-diagnostics', message);
  $('diagnostics').classList.toggle('warning', ratio > .95);
}

function drawTruthBoundary(id,opacity=1){
 if(!grid?.truth||!scene)return;
 const w=grid.width,h=grid.height,crs=maps[id].options.crs,sw=crs.project(L.latLng(scene.bounds[0])),ne=crs.project(L.latLng(scene.bounds[1]));
 const point=(x,y)=>crs.unproject(L.point(sw.x+x/w*(ne.x-sw.x),ne.y-y/h*(ne.y-sw.y)));
 const positive=(x,y)=>x>=0&&x<w&&y>=0&&y<h&&grid.truth[y*w+x]===1&&(!grid.valid||grid.valid[y*w+x]);
 const edges=[];
 for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(positive(x,y)){
  if(!positive(x,y-1))edges.push([point(x,y),point(x+1,y)]);
  if(!positive(x+1,y))edges.push([point(x+1,y),point(x+1,y+1)]);
  if(!positive(x,y+1))edges.push([point(x+1,y+1),point(x,y+1)]);
  if(!positive(x-1,y))edges.push([point(x,y+1),point(x,y)]);
 }
 if(!edges.length)return;
 for(const [color,weight] of [['#102833',5],['#38f5ed',2.5]])layers[id].push(L.polyline(edges,{color,weight,opacity,interactive:false,lineCap:'square',lineJoin:'miter'}).addTo(maps[id]));
}
function syncComparisonControls(){
 for(const [a,b] of [['compare-pred','pred-toggle'],['compare-truth','truth-toggle'],['compare-errors','error-toggle']]){$(a).checked=$(b).checked;}
 $('compare-opacity').value=$('opacity').value;text('compare-opacity-value',Math.round(+$('opacity').value*100)+'%');
}
function drawComparisonSummary({tp,fp,fn,tn,n}){
 $('comparison-stats').title='Frozen native scores use original prediction support at the saved cutoff. Display scores may change with resampling or the exploratory threshold.';
 const iou=tp+fp+fn?((tp/(tp+fp+fn))*100).toFixed(1)+'%':'—';
 const native=scene?.nativeMetrics, frozenThreshold=scene?.decisionThreshold??experiment.decisionThreshold??0.5;
 const frozenIoU=native&&Number.isFinite(native.iou)&&native.tp+native.fp+native.fn>0?(native.iou*100).toFixed(1)+'%':'—';
 const stats=scene?.positiveOnlyReference?[['EMIT support covered',tp+fn?((tp/(tp+fn))*100).toFixed(1)+'%':'—'],['Outside reference','Unknown'],['Precision / IoU','Not measured'],['Emission origin','Unverified']]:native&&Number.isInteger(native.evaluated_pixels)?[['Native IoU at '+Number(frozenThreshold).toFixed(2),frozenIoU],['Display IoU at '+(+$('threshold').value).toFixed(2),n?iou:'—'],['Native FP pixels',native.fp?.toLocaleString() ?? '—'],['Native evaluated pixels',native.evaluated_pixels.toLocaleString()]]:[['Display IoU',n?iou:'—'],['Predicted area',n?((tp+fp)/n*100).toFixed(1)+'%':'—'],['Ground-truth area',n?((tp+fn)/n*100).toFixed(1)+'%':'—'],['Scene decision',scene?.outcome||'Not recorded']];
 $('comparison-stats').innerHTML=stats.map(([k,v])=>'<div><span>'+esc(k)+'</span><strong>'+esc(v)+'</strong></div>').join('');
 text('compare-scene-label',scene?experimentLabel(experiment)+' / '+scene.name:experimentLabel(experiment)+' / no spatial export');
 text('coordinate-note',scene?.coordinateSystem==='pixel'?'Native '+(grid?.width||32)+' × '+(grid?.height||32)+' pixels · geographic footprint unavailable':native?.evaluated_pixels!=null?'Georeferenced display · native scores cover '+native.evaluated_pixels.toLocaleString()+' / '+(native.source_valid_pixels?.toLocaleString() ?? 'unknown')+' source-valid pixels; other pixels are unscored':'Georeferenced imagery · shared display grid');
 text('compare-pixel-inspector','Hover over imagery to inspect a pixel.');
 $('compare-empty').hidden=!!scene;
 text('compare-empty','No prediction imagery exported for this experiment yet. Its saved metrics are under Experiments.');
 for(const [a,b] of [['compare-pred','pred-toggle'],['compare-truth','truth-toggle'],['compare-errors','error-toggle']])$(a).disabled=$(b).disabled;
 if(grid?.truth&&n&&tp+fn===0)text('reference-status',(scene.truthLabel||'Reference mask')+' · no labeled plume in this crop');
}
function initComparisonControls(){
 for(const [a,b] of [['compare-pred','pred-toggle'],['compare-truth','truth-toggle'],['compare-errors','error-toggle']])$(a).oninput=()=>{$(b).checked=$(a).checked;renderGrid();};
 $('compare-opacity').oninput=()=>{$('opacity').value=$('compare-opacity').value;renderGrid();};
 for(const mode of ['overlay','panels'])$('layout-'+mode).onclick=()=>{
  $('comparison-layout').classList.toggle('overlay-layout',mode==='overlay');
  for(const m of ['overlay','panels'])$('layout-'+m).setAttribute('aria-pressed',String(m===mode));
  renderGrid();setTimeout(()=>{Object.values(maps).forEach(m=>m.invalidateSize());fit();},0);
 };
 $('scene-browser-toggle').onclick=()=>{const open=document.body.classList.toggle('scene-browser-open');$('scene-browser-toggle').setAttribute('aria-expanded',String(open));};
}

let reconstructionRangeCache=null,lastRenderTask=null;
const taskOriginals=new Map();
function drawArchiveStatus(){
 const note='Saved red rendering only. Raw probabilities, acquisition date and matched methane reference are unavailable.';
 text('scene-flag','ORIGINAL TEAM ARCHIVE | visual review only');
 text('prediction-status','Original colors and alpha retained; threshold is baked into the saved image');
 text('diagnostics',note);text('map-diagnostics',note);
 taskText('.compare-toolbar h1','Original team: archived prediction',true);
 taskText('.prediction-panel h3','Saved prediction overlay',true);
 taskText('.overlay-key','Original red overlay; no matched ground-truth boundary',true);
 $('comparison-stats').innerHTML='<div><span>Archive date</span><strong>March 27, 2026</strong></div><div><span>Acquisition date</span><strong>Unknown</strong></div><div><span>Raw scores / reference</span><strong>Unavailable</strong></div>';
 $('comparison-stats').title='Archive date is not acquisition date. No raw probability or matched reference survives in this rendering.';
 text('coordinate-note','Georeferenced saved rendering. Compare with Model6 visually; different scenes cannot establish model superiority.');
 text('compare-pixel-inspector','Pixel probabilities cannot be recovered from this colored export.');
}
function taskText(selector,value,on){const el=document.querySelector(selector);if(!el)return;if(!taskOriginals.has(el))taskOriginals.set(el,el.innerHTML);if(on)el.textContent=value;else el.innerHTML=taskOriginals.get(el);}
function taskLabel(id,label,on){const el=$(id).closest('label');if(!el)return;let span=el.querySelector('.task-label');if(!span){span=document.createElement('span');span.className='task-label';const texts=[...el.childNodes].filter(n=>n.nodeType===3);span.dataset.original=texts.map(n=>n.textContent).join('');texts.forEach(n=>n.remove());el.append(span);}span.textContent=on?label:span.dataset.original;}
function updateTaskUI(){
 document.body.classList.toggle('rendered-archive',scene?.outputKind==='rendered_archive');
 taskText('.prediction-panel .panel-badge','ARCHIVED OUTPUT',scene?.outputKind==='rendered_archive');
 const on=isReconstruction();if(!on)$('compare-opacity').disabled=false;if(on!==lastRenderTask){$('truth-toggle').checked=!on;$('error-toggle').checked=false;lastRenderTask=on;}
 for(const id of ['prediction-mode','prediction-mode-compare'])$(id).closest('label').style.display=on?'none':'';
 document.querySelector('.analysis article').style.display=on?'none':'';
 $('threshold').disabled=on;$('reset-threshold').disabled=on;
 const positiveOnly=scene?.positiveOnlyReference&&!on;
 taskText('.compare-toolbar h1',on?'Methane reconstruction':'Prediction over EMIT positive support',on||positiveOnly);taskText('.observation-panel h3','Observed target',on);taskText('.observation-panel .layer-status',scene?.truthLabel||'Observed reference grid',on);
 taskText('.truth-panel h3',on?'Coarse input / baseline':'EMIT positive outline',on||positiveOnly);taskText('.prediction-panel h3',on?scene?.predictionLabel||'Actual reconstruction':'Prediction + EMIT positive outline',on||positiveOnly);taskText('.analysis article:nth-child(2) h3',on?'Reconstruction error':'EMIT support coverage',on||positiveOnly);
 taskText('.evidence .micro','Cross-section: blue observed, orange coarse, teal prediction, red residual.',on);
 taskText('.overlay-key',on?'Colored footprint is observed coverage, not a learned outline.':'Model prediction + EMIT positive boundary',on||positiveOnly);
 for(const [id,label] of [['truth-toggle','Observed target'],['compare-truth','Observed target'],['error-toggle','Residual'],['compare-errors','Residual']])taskLabel(id,label,on);
 if(positiveOnly)taskLabel('compare-truth','EMIT positive outline',true);
 for(const selector of ['.compare-legend','.map-legend']){const el=document.querySelector(selector);if(el){if(!el.dataset.originalHtml)el.dataset.originalHtml=el.innerHTML;if(!on)el.innerHTML=el.dataset.originalHtml;}}
 if(positiveOnly){const labels=document.querySelectorAll('.compare-legend span');labels[1].lastChild.textContent=' EMIT positive support';labels[2].lastChild.textContent=' Outside outline: unknown';labels[3].lastChild.textContent=' Missed positive support';}
}


try {
  if (!window.L) throw Error('Map library could not load');
  Object.assign(registry, viewerLive.validate(registry));
  ['main-map', 'original-map', 'prediction-map', 'truth-map'].forEach(initMap);
  populateExperiments();
  document.querySelector('.brand').onclick = event => { event.preventDefault(); switchView('national', true); };
  $('experiment').onchange = () => selectExperiment($('experiment').value);
  $('catalog-mode').onchange = () => {
    showArchive = $('catalog-mode').value === 'all';
    populateExperiments();
    if (!catalogExperiments().some(entry => entry.id === experiment.id)) selectExperiment(registry.activeExperimentId);
    drawExperimentCards();
    atlas.refresh();
    updateURL();
  };
  $('start-presentation').onclick = startPresentation;
  $('search').oninput = drawList;
  $('filter').onchange = drawList;
  document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => switchView(button.dataset.view, true));
  for (const id of ['pred-toggle', 'truth-toggle', 'error-toggle', 'bounds-toggle', 'opacity', 'threshold', 'prediction-mode', 'reference-opacity', 'comparison-support'])
    $(id).oninput = renderGrid;
  $('reset-threshold').onclick = () => {
    $('threshold').value = scene?.decisionThreshold ?? experiment.decisionThreshold ?? 0.5;
    renderGrid();
  };
  $('prediction-mode-compare').oninput = () => { $('prediction-mode').value = $('prediction-mode-compare').value; renderGrid(); };
  $('fit').onclick = fit;
  initComparisonControls();
  $('overview').onclick = () => { switchView('map'); setTimeout(() => maps['main-map'].setView([39, -98], 4), 25); };
  $('import').onchange = async event => {
    try {
      const file = event.target.files[0];
      if (!file) return;
      if (file.size > 50 * 1024 * 1024) throw Error('Bundle exceeds 50 MiB');
      const parsed = JSON.parse(await file.text()), entry = validateExperiment(parsed.experiment || parsed);
      if (registry.experiments.some(item => item.id === entry.id)) throw Error('Experiment ID already exists');
      registry.experiments.push(entry);
      viewerLive.imports.set(entry.id, entry);
      populateExperiments();
      selectExperiment(entry.id);
      text('message', 'Opened in this browser only. Nothing uploaded.');
    } catch (error) { text('message', 'Import failed: ' + error.message); }
    finally { event.target.value = ''; }
  };
  $('online-basemap').onchange = () => {
    for (const [id, layer] of Object.entries(basemaps))
      $('online-basemap').checked ? layer.addTo(maps[id]) : maps[id].removeLayer(layer);
  };
  const query = new URLSearchParams(location.search), initialView = location.hash.slice(1) || registry.presentation?.defaultView || 'compare';
  text('local-folder', registry.local?.bundleDirectory || 'outputs/GTM_viewer_bundles');
  text('message', registry.local?.warnings?.join(' · ') || '');
  selectExperiment(query.get('experiment') || registry.activeExperimentId || registry.experiments[0].id, query.get('scene'));
  switchView(initialView);
  atlas.start();
  viewerLive.start();
  window.addEventListener('popstate', () => {
    const state = new URLSearchParams(location.search), previousView = location.hash.slice(1);
    showArchive = state.get('catalog') === 'all';
    populateExperiments();
    if (state.get('experiment') !== experiment.id || state.get('scene') !== scene?.id)
      selectExperiment(state.get('experiment'), state.get('scene'));
    switchView(previousView);
    atlas.refresh();
  });
} catch (error) {
  document.body.dataset.connection = 'offline';
  text('live-status', 'Viewer could not start');
  text('message', error.message);
  console.error(error);
}
