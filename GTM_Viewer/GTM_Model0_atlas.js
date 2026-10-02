'use strict';

// One catalog, two connected views. All imagery and scores come from saved exports.
const atlas = {
  map: null,
  locator: null,
  records: [],
  visible: [],
  selected: null,
  states: null,
  markers: null,
  overlays: null,
  locatorLayers: null,
  dates: [],
  layerRevision: '',
  camera: {center: [38.5, -98], zoom: 4},
  resizing: false,
  satelliteOn: false,
  stateLayer: null,
  outsideMask: null,
  stateLabels: [],

  percent(value, digits = 1) { return Number.isFinite(value) ? (value * 100).toFixed(digits) + '%' : 'Unavailable'; },
  date(value) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(value || '')) return value || 'Acquisition date unavailable';
    return new Date(value + 'T12:00:00Z').toLocaleDateString('en-GB', {day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC'});
  },
  branch(entry) {
    const match = (entry.name + ' ' + entry.id).match(/(?:\|\s*|_)(mean|s16|s32|s64|s128)(?:\s*\||\s|$)/i);
    return match ? match[1].toLowerCase() : null;
  },
  family(entry) {
    return entry.name.toLowerCase().split('|').map(s => s.trim()).filter(s =>
      !['mean', 's16', 's32', 's64', 's128'].includes(s) && !s.includes('portable')).join('|');
  },
  point(item) {
    if (item.coordinateSystem !== 'pixel' && item.bounds) return [
      (item.bounds[0][0] + item.bounds[1][0]) / 2, (item.bounds[0][1] + item.bounds[1][1]) / 2];
    if (Array.isArray(item.locationPoint) && item.locationPoint.length === 2 &&
        item.locationPoint.every(Number.isFinite) && Math.abs(item.locationPoint[0]) <= 85 && Math.abs(item.locationPoint[1]) <= 180)
      return item.locationPoint;
    return null;
  },
  inUS(point) {
    const [lat, lon] = point;
    return lat >= 24 && lat <= 50 && lon >= -125 && lon <= -66 ||
      lat >= 51 && lat <= 72 && (lon <= -129 && lon >= -180 || lon >= 172) ||
      lat >= 18 && lat <= 23 && lon >= -161 && lon <= -154;
  },
  evidence(item) {
    if (!hasPred(item)) return {key: 'input', label: 'Input only'};
    if (item.positiveOnlyReference) return {key: 'unscored', label: 'Positive-only reference'};
    const n = item.nativeMetrics;
    if (n && n.tp === 0 && n.fn > 0) return {key: 'missed', label: 'Missed plume'};
    if (n && n.tp > 0) return {key: 'partial', label: n.fn ? 'Partial plume overlap' : 'Plume overlap'};
    if (n && n.fp > 0 && n.tp === 0 && n.fn === 0) return {key: 'false-positive', label: 'False positive'};
    if (n && n.tp === 0 && n.fn === 0 && n.fp === 0) return {key: 'clear', label: 'Reviewed negative'};
    return {key: 'unscored', label: item.outputKind === 'rendered_archive' ? 'Rendered archive' : 'Unscored prediction'};
  },
  title(item) {
    if (item.dataset === 'MARS') return item.label === 'PLUME' ? 'MARS · reviewed plume' : 'MARS · reviewed control';
    if (item.dataset === 'EMIT') return 'EMIT · positive support';
    return item.name;
  },
  chosen(record) {
    const filter = $('atlas-experiment').value;
    return record.variants.find(v => v.experiment.id === filter) ||
      record.variants.find(v => v.experiment.id === experiment?.id) || record.variants[0];
  },

  rebuild() {
    const grouped = new Map();
    // Prefer complete exports over portable copies when both contain identical outputs.
    const entries = [...registry.experiments].sort((a, b) =>
      Number(a.id.includes('portable')) - Number(b.id.includes('portable')) || b.scenes.length - a.scenes.length);
    for (const entry of entries) for (const item of entry.scenes) {
      const point = this.point(item);
      if (point && !this.inUS(point)) continue;
      const key = [item.dataset || '', item.id, item.date || ''].join('|');
      if (!grouped.has(key)) grouped.set(key, {key, point, variants: []});
      const record = grouped.get(key);
      // A cache/checkpoint can contain several branches; its hash alone is not an output identity.
      const signature = [this.family(entry), this.branch(entry), item.predictionHash || item.checkpointHash || '',
        item.outputKind || '', JSON.stringify(item.nativeMetrics || {})].join('|');
      if (!record.variants.some(v => v.signature === signature)) record.variants.push({experiment: entry, scene: item, signature});
    }
    this.records = [...grouped.values()].sort((a, b) => (b.variants[0].scene.date || '').localeCompare(a.variants[0].scene.date || '') || a.key.localeCompare(b.key));
    const filterValue = $('atlas-experiment').value;
    const available = new Map();
    for (const record of this.records) for (const variant of record.variants) available.set(variant.experiment.id, variant.experiment);
    $('atlas-experiment').innerHTML = '<option value="all">All saved experiments</option>' + [...available.values()]
      .map(entry => '<option value="' + esc(entry.id) + '">' + esc(entry.name) + '</option>').join('');
    if (available.has(filterValue)) $('atlas-experiment').value = filterValue;
    this.filter(false);
  },

  filter(writeURL = true) {
    if (!this.map) return;
    const query = $('atlas-search').value.trim().toLowerCase(), run = $('atlas-experiment').value;
    const from = $('atlas-from').value, through = $('atlas-to').value, review = $('atlas-filter').value;
    const coordinateQuery = query.match(/^(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)$/);
    this.visible = this.records.filter(record => {
      if (run !== 'all' && !record.variants.some(v => v.experiment.id === run)) return false;
      const item = this.chosen(record).scene;
      if (from && (!item.date || item.date < from) || through && (!item.date || item.date > through)) return false;
      if (review === 'unmapped' && item.coordinateSystem !== 'pixel') return false;
      if (review !== 'all' && review !== 'unmapped') {
        if (review === 'false-positive') { if (!(item.nativeMetrics?.fp > 0)) return false; }
        else if (this.evidence(item).key !== review) return false;
      }
      if (coordinateQuery) return record.point && Math.abs(record.point[0] - Number(coordinateQuery[1])) < .1 &&
        Math.abs(record.point[1] - Number(coordinateQuery[2])) < .1;
      return !query || [item.id, item.name, item.location, item.source, item.date, record.point?.map(n => n.toFixed(3)).join(', ')]
        .join(' ').toLowerCase().includes(query);
    });
    this.renderList();
    this.drawMarkers();
    $('atlas-selection-note').hidden = !scene || this.visible.some(r => r.variants.some(v => v.scene.id === scene.id && v.scene.date === scene.date));
    this.dates = [...new Set(this.visible.map(record => this.chosen(record).scene.date).filter(date => /^\d{4}-\d{2}-\d{2}$/.test(date || '')))].sort();
    $('atlas-date').max = Math.max(0, this.dates.length - 1);
    $('atlas-date').disabled = !this.dates.length;
    this.syncDate();
    this.drawSelectedLayers();
    if (writeURL) this.saveState();
  },

  renderList() {
    const previousScroll = $('atlas-scenes').scrollTop;
    $('atlas-scenes').innerHTML = this.visible.map((record, index) => {
      const item = this.chosen(record).scene, status = this.evidence(item), active = item.id === scene?.id && item.date === scene?.date;
      return '<button class="atlas-scene ' + (active ? 'active' : '') + '" data-observation="' + index +
        '" aria-pressed="' + active + '">' + (item.rgb || item.observedImage ? '<img loading="lazy" src="' +
          esc(assetUrl(item.rgb || item.observedImage)) + '" alt="">' : '<span class="scene-placeholder">◌</span>') +
        '<span><b>' + esc(this.title(item)) + '</b><small>' + esc(this.date(item.date)) + ' · ' +
        esc(record.point ? record.point.map(n => n.toFixed(3)).join(', ') : 'Footprint unavailable') +
        '</small><span class="evidence-badge ' + status.key + '">' + status.label + '</span></span></button>';
    }).join('');
    $('atlas-scenes').scrollTop = previousScroll;
    $('atlas-scenes').querySelectorAll('[data-observation]').forEach(button => button.onclick = () => {
      this.choose(this.visible[Number(button.dataset.observation)]);
      document.body.classList.remove('atlas-browser-open');
      $('atlas-browser-toggle').setAttribute('aria-expanded', 'false');
    });
    text('atlas-count', '(' + this.visible.length + ')');
    $('atlas-empty').hidden = this.visible.length > 0;
    const unmapped = this.visible.filter(record => !record.point).length;
    text('atlas-scope', this.visible.length + ' saved observations · ' + unmapped +
      ' without a mapped location. Multiple model outputs share one observation. Unmarked areas have no result.');
  },

  choose(record, variant = this.chosen(record)) {
    if (!record || !variant) return;
    if (experiment.id !== variant.experiment.id) {
      experiment = variant.experiment;
      $('experiment').value = experiment.id;
      updateRunSummary();
      drawExperimentCards();
    }
    selectScene(variant.scene.id);
    if (record.point) this.map.panTo(record.point, {animate: false});
  },

  drawMarkers() {
    if (!this.map) return;
    this.markers.clearLayers();
    const buckets = new Map(), size = this.map.getZoom() < 10 ? 46 : 28;
    for (const record of this.visible.filter(r => r.point)) {
      const p = this.map.project(record.point), key = Math.floor(p.x / size) + ':' + Math.floor(p.y / size);
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(record);
    }
    for (const records of buckets.values()) {
      const selected = records.some(r => r.variants.some(v => v.scene.id === scene?.id && v.scene.date === scene?.date));
      const point = records.reduce((p, r) => [p[0] + r.point[0] / records.length, p[1] + r.point[1] / records.length], [0, 0]);
      const clustered = records.length > 1, item = this.chosen(records[0]).scene;
      const label = clustered ? records.length + ' saved observations; zoom to inspect' : this.title(item) + ', ' + this.date(item.date);
      const icon = L.divIcon({className: 'observation-dot ' + (clustered ? 'cluster' : this.evidence(item).key) +
        (selected ? ' selected' : ''), html: clustered ? String(records.length) : '', iconSize: clustered ? [30, 30] : [13, 13]});
      const marker = L.marker(point, {icon, title: label, alt: label, keyboard: true}).addTo(this.markers);
      marker.bindTooltip(esc(label));
      marker.on('click', () => {
        if (!clustered) { this.choose(records[0]); return; }
        if (this.map.getZoom() < 12) {
          this.map.fitBounds(records.map(r => r.point), {padding: [70, 70], maxZoom: 13, animate: false});
        } else {
          const choices = document.createElement('div'); choices.className = 'cluster-choices';
          for (const record of records) {
            const button = document.createElement('button'), item = this.chosen(record).scene;
            button.textContent = this.date(item.date) + ' · ' + item.id.slice(-8);
            button.onclick = () => { this.map.closePopup(); this.choose(record); };
            choices.append(button);
          }
          marker.bindPopup(choices).openPopup();
        }
      });
    }
  },

  syncDate() {
    const focusedDate = document.activeElement?.closest('[data-date-index]')?.dataset.dateIndex;
    const index = this.dates.indexOf(scene?.date);
    $('atlas-date').value = Math.max(0, index);
    text('atlas-date-label', index >= 0 ? this.date(scene.date) : 'Select a saved acquisition');
    $('atlas-previous').disabled = index <= 0;
    $('atlas-next').disabled = !this.dates.length || index >= this.dates.length - 1;
    $('atlas-date').setAttribute('aria-valuetext', index >= 0 ? this.date(scene.date) : 'No filtered acquisition selected');
    text('atlas-date-count', this.dates.length + ' saved acquisition' + (this.dates.length === 1 ? '' : 's'));
    if (!this.dates.length) {
      $('atlas-date-marks').innerHTML = '<span class="timeline-empty">No dated exports in this view</span>';
      $('atlas-date-range').innerHTML = '';
      return;
    }
    const firstYear = Number(this.dates[0].slice(0, 4)), lastYear = Number(this.dates.at(-1).slice(0, 4)) + 1;
    const start = Date.UTC(firstYear, 0, 1), end = Date.UTC(lastYear, 0, 1);
    const position = timestamp => (timestamp - start) / (end - start) * 100;
    const yearTicks = [];
    for (let year = firstYear; year <= lastYear; year++) yearTicks.push('<span class="acquisition-year" style="left:' +
      position(Date.UTC(year, 0, 1)) + '%"><i></i><span>' + year + '</span></span>');
    $('atlas-date-range').innerHTML = yearTicks.join('');
    $('atlas-date-marks').innerHTML = this.dates.map((date, number) => {
      const selected = number === index, count = this.visible.filter(record => this.chosen(record).scene.date === date).length;
      const label = this.date(date) + ' · ' + count + ' saved scene' + (count === 1 ? '' : 's');
      return '<button class="acquisition-dot ' + (selected ? 'selected' : '') + '" style="left:' + position(Date.parse(date + 'T00:00:00Z')) +
        '%" data-date-index="' + number + '" aria-label="' + esc(label) + '" aria-pressed="' + selected + '" title="' +
        esc(label) + '" tabindex="' + (selected || index < 0 && number === 0 ? '0' : '-1') + '"><span></span></button>';
    }).join('');
    $('atlas-date-marks').querySelectorAll('[data-date-index]').forEach(button => {
      button.onclick = () => this.pickDate(Number(button.dataset.dateIndex));
      button.onkeydown = event => {
        const current = Number(button.dataset.dateIndex);
        const next = event.key === 'ArrowRight' ? current + 1 : event.key === 'ArrowLeft' ? current - 1 :
          event.key === 'Home' ? 0 : event.key === 'End' ? this.dates.length - 1 : null;
        if (next == null) return;
        event.preventDefault();
        this.pickDate(Math.max(0, Math.min(this.dates.length - 1, next)));
        $('atlas-date-marks').querySelector('.selected')?.focus();
      };
    });
    if (focusedDate != null) $('atlas-date-marks').querySelector('[data-date-index="' + focusedDate + '"]')?.focus();
  },
  pickDate(index) {
    const date = this.dates[index];
    const record = this.visible.find(r => this.chosen(r).scene.date === date);
    if (record) this.choose(record);
  },

  supportImage() {
    if (!grid?.probability || !scene) return null;
    const canvas = document.createElement('canvas'); canvas.width = grid.width; canvas.height = grid.height;
    const context = canvas.getContext('2d'), pixels = context.createImageData(grid.width, grid.height);
    for (let i = 0; i < grid.width * grid.height; i++) {
      if ((!grid.valid || grid.valid[i]) && grid.probability[i] != null) continue;
      const stripe = (i % grid.width + Math.floor(i / grid.width)) % 7 < 2;
      pixels.data.set(stripe ? [151, 168, 180, 120] : [7, 20, 28, 85], i * 4);
    }
    context.putImageData(pixels, 0, 0);
    return canvas.toDataURL();
  },

  drawSelectedLayers() {
    if (!this.overlays) return;
    this.overlays.clearLayers();
    if (!scene || scene.coordinateSystem === 'pixel') return;
    if (!this.visible.some(r => r.variants.some(v => v.scene.id === scene.id && v.scene.date === scene.date))) return;
    if ($('atlas-footprint').checked) L.rectangle(scene.bounds, {color: '#56d9de', weight: 1.5, fillOpacity: .025}).addTo(this.overlays);
    if (this.map.getZoom() < 10) return;
    const add = (url, opacity = 1) => {
      if (url) L.imageOverlay(assetUrl(url), scene.bounds, {opacity, interactive: false}).on('error', () =>
        text('atlas-map-status', 'A saved layer is unavailable. Open the scene for details.')).addTo(this.overlays);
    };
    add(scene.rgb);
    if ($('atlas-prediction').checked) add(grid?.probability ? makeLayer('prediction') : scene.predictionImage, .65);
    if ($('atlas-reference').checked) add(grid?.truth ? makeLayer('truth') : scene.truthImage, 1);
    if ($('atlas-support').checked) add(this.supportImage());
  },

  renderSelected() {
    if (!this.map) return;
    const changedSelection = this.lastScene !== scene?.id;
    this.lastScene = scene?.id;
    this.selected = this.records.find(r => r.variants.some(v => v.scene.id === scene?.id && v.scene.date === scene?.date));
    $('atlas-selection-note').hidden = !scene || this.visible.includes(this.selected);
    const point = scene && this.point(scene), n = scene?.nativeMetrics;
    text('atlas-title', scene ? this.title(scene) : 'Select an observation');
    text('atlas-location', point ? point.map((value, i) => Math.abs(value).toFixed(3) + '° ' + (i ? value < 0 ? 'W' : 'E' : value < 0 ? 'S' : 'N')).join(', ') : 'Geographic footprint unavailable');
    text('atlas-acquired', this.date(scene?.date));
    const variants = this.selected?.variants || [];
    $('atlas-model').innerHTML = variants.map(v => '<option value="' + esc(v.experiment.id) + '">' + esc(v.experiment.name) + '</option>').join('');
    if (!variants.some(v => v.experiment.id === experiment.id) && scene) {
      const option = document.createElement('option'); option.value = experiment.id; option.textContent = experiment.name;
      $('atlas-model').append(option);
    }
    $('atlas-model').value = experiment.id;
    $('atlas-model').disabled = !scene;
    const prediction = grid?.probability ? makeLayer('prediction') : scene?.predictionImage;
    const truth = grid?.truth ? makeLayer('truth') : scene?.truthImage;
    const support = this.supportImage();
    const image = (url, label, opacity = 1) => url ? '<img src="' + esc(assetUrl(url)) + '" alt="' + esc(label) + '" style="opacity:' + opacity + '">' : '';
    $('atlas-preview').innerHTML = scene && (scene.rgb || scene.observedImage || prediction) ?
      image(scene.rgb || scene.observedImage, 'Saved observed scene') +
      ($('atlas-prediction').checked ? image(prediction, 'Saved prediction', .65) : '') +
      ($('atlas-reference').checked ? image(truth, 'Reference annotation') : '') +
      ($('atlas-support').checked ? image(support, 'Hatched pixels have no model output') : '') +
      '<span class="preview-caption">Saved data · ' + (scene.outputKind === 'rendered_archive' ? 'original red rendering; no matched reference' : prediction ? 'orange prediction / cyan reference' : 'no model prediction') + '</span>' :
      '<div class="preview-empty">No image exported for this selection.</div>';
    $('atlas-preview').querySelectorAll('img').forEach(img => img.onerror = () => {
      img.dataset.state = 'error';
      const caption = $('atlas-preview').querySelector('.preview-caption');
      if (caption) caption.textContent = 'An image layer could not load';
    });
    $('atlas-metrics').innerHTML = this.metrics([['Scene IoU', this.percent(n?.iou)], ['Evaluated area', this.percent(n?.support_fraction, 2)],
      ['Run IoU', this.percent(experiment.nativeAggregate?.iou, 2)]]);
    text('atlas-run-status', experiment.status || 'Run status not recorded');
    const entries = {Scene: scene?.id, Center: point?.map(v => v.toFixed(4)).join(', '), Acquired: this.date(scene?.date),
      Reference: scene?.referenceDate ? this.date(scene.referenceDate) : 'Unavailable',
      Coordinates: scene?.coordinateSystem === 'pixel' ? point ? 'Point only; no footprint' : 'Image pixels only' : scene ? 'Exported geographic bounds' : 'Unavailable'};
    $('atlas-provenance').innerHTML = Object.entries(entries).map(([key, value]) => '<dt>' + key + '</dt><dd>' + esc(value) + '</dd>').join('');
    text('atlas-note', scene?.notes || experiment.notes || 'Select a saved observation.');
    $('atlas-fit').disabled = !point;
    $('atlas-footprint').disabled = !scene || scene.coordinateSystem === 'pixel';
    $('atlas-reference').disabled = !truth;
    $('atlas-prediction').disabled = !prediction;
    $('atlas-support').disabled = !support;
    for (const id of ['open-viewer', 'atlas-open-viewer', 'compare-open-viewer']) {
      const url = new URL(location.href); url.hash = 'compare';
      if (experiment) url.searchParams.set('experiment', experiment.id);
      if (scene) url.searchParams.set('scene', scene.id);
      $(id).href = url.href;
    }
    this.renderList();
    if (changedSelection) $('atlas-scenes').querySelector('.active')?.scrollIntoView({block: 'nearest'});
    this.drawMarkers();
    this.syncDate();
    this.drawSelectedLayers();
    this.renderComparison();
    this.renderImagery();
  },

  metrics(values) {
    return values.map(([label, value]) => '<div><span>' + esc(label) + '</span><strong>' + esc(value) + '</strong></div>').join('');
  },
  renderComparison() {
    $('comparison-experiment').innerHTML = registry.experiments.map(entry => '<option value="' + esc(entry.id) + '">' + esc(entry.name) + '</option>').join('');
    $('comparison-experiment').value = experiment.id;
    $('comparison-scene').innerHTML = experiment.scenes.map(item => '<option value="' + esc(item.id) + '">' + esc(item.name) + '</option>').join('');
    $('comparison-scene').value = scene?.id || '';
    $('comparison-scene').disabled = !scene;
    const candidates = registry.experiments.filter(e => this.family(e) === this.family(experiment)).sort((a, b) =>
      Number(a.id.includes('portable')) - Number(b.id.includes('portable')) || b.scenes.length - a.scenes.length);
    $('comparison-branches').innerHTML = ['mean', 's16', 's32', 's64', 's128'].map(branch => {
      const match = candidates.find(e => this.branch(e) === branch && e.scenes.some(s => s.id === scene?.id));
      return '<button data-branch="' + esc(match?.id || '') + '" aria-pressed="' + (this.branch(experiment) === branch) +
        '" ' + (match ? '' : 'disabled') + ' title="' + (match ? esc(match.name) : 'No saved output for this scene') + '">' +
        (branch === 'mean' ? 'Mean' : branch.slice(1)) + '</button>';
    }).join('');
    $('comparison-branches').querySelectorAll('[data-branch]').forEach(button => button.onclick = () => {
      experiment = registry.experiments.find(e => e.id === button.dataset.branch);
      $('experiment').value = experiment.id; updateRunSummary(); selectScene(scene.id, {preserveView: true});
    });
    const n = scene?.nativeMetrics;
    const reference = scene?.truthLabel || 'Unavailable';
    $('native-scene-summary').innerHTML = '<div class="atlas-metrics">' + this.metrics([
      ['IoU', this.percent(n?.iou)], ['Precision', this.percent(n?.precision)], ['Recall', this.percent(n?.recall)]]) +
      '</div><dl><dt>Evaluated area</dt><dd>' + esc(this.percent(n?.support_fraction, 2)) + '</dd><dt>Reference</dt><dd>' +
      esc(reference) + '</dd><dt>Acquired</dt><dd>' + esc(this.date(scene?.date)) + '</dd></dl>';
    const aggregate = experiment.nativeAggregate;
    $('whole-run-summary').innerHTML = '<p>' + (aggregate?.scenes != null ? esc(aggregate.scenes) + ' evaluated scenes' :
      experiment.scenes.length + ' exported scenes; aggregate unavailable') + '</p><p>Pooled native IoU <strong>' +
      esc(this.percent(aggregate?.iou, 2)) + '</strong></p><p class="run-warning">' + esc(experiment.status) +
      '</p><p class="micro">' + esc(experiment.metricScope === 'reused_development_fold' ? 'Reused development data. Selected examples do not establish generalization.' :
        experiment.scenes.some(s => s.positiveOnlyReference) ? 'Positive-only references do not establish specificity.' : 'See the run report for evaluation scope.') + '</p>';
    this.locatorLayers.clearLayers();
    const point = scene && this.point(scene);
    text('locator-note', point ? point.map(n => n.toFixed(4)).join(', ') + ' · image center, not an emission source' : 'No verified geographic location for this scene.');
    if (point) {
      L.circleMarker(point, {color: '#60e3e4', radius: 5, weight: 2, fillOpacity: .6}).addTo(this.locatorLayers);
      if (scene.coordinateSystem !== 'pixel') L.rectangle(scene.bounds, {color: '#60e3e4', weight: 1}).addTo(this.locatorLayers);
      this.locator.setView(point, 6, {animate: false});
    } else this.locator.setView([39, -98], 3, {animate: false});
  },

  renderImagery() {
    text('imagery-description', scene ? scene.name + ' · saved acquisitions and reference layers' : 'Select a scene on the map.');
    const layers = scene ? [
      ['Sentinel-2 target', scene.rgb, this.date(scene.date) + ' · ' + (scene.rgbNote || 'Display contrast stretch')],
      ['Temporal reference −90d', scene.reference90Rgb, 'Nominal reference window; actual date must be recorded in the export'],
      ['Temporal reference −365d', scene.reference365Rgb, 'Nominal reference window; actual date must be recorded in the export'],
      [scene.emitLabel || 'EMIT observation', scene.emit, scene.emitNote || this.date(scene.referenceDate)],
      [scene.truthLabel || 'Reference annotation', scene.truthImage || scene.observedImage, scene.truthNote || this.date(scene.referenceDate)],
      ['Coarse input', scene.coarseImage, scene.units || 'Units not recorded']
    ].filter(([, url]) => url) : [];
    $('imagery-layers').innerHTML = layers.map(([label, url, note]) => '<article><img src="' + esc(assetUrl(url)) +
      '" alt="' + esc(label) + '"><h2>' + esc(label) + '</h2><p>' + esc(note) + '</p></article>').join('') ||
      '<p class="note">No observed imagery exported for this selection.</p>';
    text('imagery-metadata', scene ? [scene.resolution, scene.source, scene.notes].filter(Boolean).join(' · ') : '');
  },

  saveState() {
    if (!this.map) return;
    const url = new URL(location.href), center = this.map.getCenter();
    for (const [key, value] of Object.entries({nlat: center.lat.toFixed(5), nlon: center.lng.toFixed(5), nz: this.map.getZoom().toFixed(2),
      nb: this.satelliteOn ? 'satellite' : '',
      nq: $('atlas-search').value, nf: $('atlas-filter').value, ne: $('atlas-experiment').value,
      from: $('atlas-from').value, to: $('atlas-to').value})) {
      if (value && value !== 'all') url.searchParams.set(key, value); else url.searchParams.delete(key);
    }
    history.replaceState(null, '', url);
    for (const id of ['open-viewer', 'atlas-open-viewer', 'compare-open-viewer']) {
      const link = new URL(url); link.hash = 'compare'; $(id).href = link.href;
    }
  },

  refresh() { if (this.map) { this.rebuild(); this.renderSelected(); } },
  updateStateLabels() {
    const abbreviations = {
      'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
      'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
      'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
      'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
      'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
      'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
      'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
      'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
      'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
      'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
      'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    };
    for (const {label, name} of this.stateLabels) label.setContent('<span title="' + esc(name) + '">' +
      esc(this.map.getZoom() < 6 ? abbreviations[name] || name : name) + '</span>');
  },
  updateBasemapStyle() {
    this.stateLayer?.setStyle({fillOpacity: this.satelliteOn ? 0 : .72,
      color: this.satelliteOn ? '#b8c8cf' : '#47606e', weight: this.satelliteOn ? 1 : .8});
    if (this.outsideMask) {
      if (this.satelliteOn) this.outsideMask.addTo(this.map).bringToBack();
      else this.map.removeLayer(this.outsideMask);
    }
  },
  resize() {
    if (!this.map) return;
    this.resizing = true;
    if (view === 'national') {
      this.map.invalidateSize({pan: false});
      this.map.setView(this.camera.center, this.camera.zoom, {animate: false});
    }
    if (view === 'compare') {
      this.locator.invalidateSize({pan: false});
      const point = scene && this.point(scene);
      this.locator.setView(point || [39, -98], point ? 6 : 3, {animate: false});
    }
    this.resizing = false;
  },

  async start() {
    if (this.map) return;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.map = L.map('atlas-map', {zoomControl: false, trackResize: false, zoomSnap: .25, minZoom: 2, maxZoom: 19, zoomAnimation: !reduced, fadeAnimation: !reduced})
      .setView([38.5, -98], 4);
    L.control.zoom({position: 'topright'}).addTo(this.map);
    L.control.scale({position: 'bottomright', imperial: false}).addTo(this.map);
    this.map.attributionControl.addAttribution('Boundaries: <a href="https://github.com/topojson/us-atlas">US Census / us-atlas</a>');
    this.locator = L.map('locator-map', {zoomControl: false, attributionControl: false, dragging: false, scrollWheelZoom: false,
      doubleClickZoom: false, boxZoom: false, touchZoom: false, keyboard: false}).setView([39, -98], 3);
    this.markers = L.layerGroup().addTo(this.map);
    this.overlays = L.layerGroup().addTo(this.map);
    this.locatorLayers = L.layerGroup().addTo(this.locator);
    const query = new URLSearchParams(location.search);
    for (const [id, key] of [['atlas-search', 'nq'], ['atlas-filter', 'nf'], ['atlas-from', 'from'], ['atlas-to', 'to']])
      if (query.has(key)) $(id).value = query.get(key);
    this.rebuild();
    if (query.has('ne') && [...$('atlas-experiment').options].some(o => o.value === query.get('ne'))) $('atlas-experiment').value = query.get('ne');
    if (query.has('nlat') && query.has('nlon') && query.has('nz')) {
      const lat = +query.get('nlat'), lon = +query.get('nlon'), zoom = +query.get('nz');
      if ([lat, lon, zoom].every(Number.isFinite) && Math.abs(lat) < 85 && Math.abs(lon) <= 180 && zoom >= 2 && zoom <= 19)
        this.camera = {center: [lat, lon], zoom};
    }
    this.map.setView(this.camera.center, this.camera.zoom, {animate: false});
    this.map.on('moveend', () => {
      if (this.resizing || view !== 'national') return;
      this.camera = {center: this.map.getCenter(), zoom: this.map.getZoom()};
      this.drawMarkers(); this.drawSelectedLayers(); this.updateStateLabels(); this.saveState();
    });
    window.addEventListener('resize', () => this.resize());
    for (const id of ['atlas-search', 'atlas-filter', 'atlas-experiment', 'atlas-from', 'atlas-to'])
      $(id).addEventListener(id === 'atlas-search' ? 'input' : 'change', () => this.filter());
    $('atlas-reset').onclick = () => {
      for (const id of ['atlas-search', 'atlas-from', 'atlas-to']) $(id).value = '';
      $('atlas-filter').value = $('atlas-experiment').value = 'all'; this.filter();
    };
    $('atlas-browser-toggle').onclick = () => $('atlas-browser-toggle').setAttribute('aria-expanded',
      String(document.body.classList.toggle('atlas-browser-open')));
    $('atlas-model').onchange = () => {
      const variant = this.selected?.variants.find(v => v.experiment.id === $('atlas-model').value);
      if (variant) this.choose(this.selected, variant);
    };
    $('comparison-experiment').onchange = () => selectExperiment($('comparison-experiment').value);
    $('comparison-scene').onchange = () => selectScene($('comparison-scene').value);
    $('atlas-compare').onclick = () => switchView('compare', true);
    $('atlas-open-scene').onclick = () => switchView('map', true);
    for (const id of ['atlas-footprint', 'atlas-reference', 'atlas-prediction', 'atlas-support']) $(id).onchange = () => this.renderSelected();
    $('atlas-home').onclick = () => this.map.fitBounds([[24, -125], [50, -66]], {padding: [20, 20], animate: false});
    $('atlas-alaska').onclick = () => this.map.fitBounds([[51, -170], [72, -130]], {animate: false});
    $('atlas-hawaii').onclick = () => this.map.fitBounds([[18.8, -160.4], [22.3, -154.8]], {animate: false});
    $('atlas-fit').onclick = () => {
      if (scene?.coordinateSystem !== 'pixel' && scene?.bounds) this.map.fitBounds(scene.bounds, {padding: [35, 35], maxZoom: 17, animate: false});
      else if (scene && this.point(scene)) this.map.setView(this.point(scene), 12, {animate: false});
    };
    $('atlas-date').oninput = () => this.pickDate(+$('atlas-date').value);
    $('atlas-previous').onclick = () => this.pickDate(+$('atlas-date').value - 1);
    $('atlas-next').onclick = () => this.pickDate(Math.max(0, this.dates.indexOf(scene?.date) + 1));
    const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {maxZoom: 19, attribution: 'Imagery © Esri and contributors'});
    satellite.on('tileerror', () => text('atlas-map-status', 'Satellite unavailable · local boundaries and results retained'));
    $('atlas-satellite').onclick = () => {
      this.satelliteOn = true;
      this.updateBasemapStyle();
      satellite.addTo(this.map); $('atlas-satellite').setAttribute('aria-pressed', 'true'); $('atlas-offline').setAttribute('aria-pressed', 'false');
      text('atlas-map-status', 'Online satellite context · Esri');
      this.saveState();
    };
    $('atlas-offline').onclick = () => {
      this.satelliteOn = false;
      this.updateBasemapStyle();
      this.map.removeLayer(satellite); $('atlas-satellite').setAttribute('aria-pressed', 'false'); $('atlas-offline').setAttribute('aria-pressed', 'true');
      text('atlas-map-status', 'Local boundaries · no tile download');
      this.saveState();
    };
    if (query.get('nb') === 'satellite') $('atlas-satellite').click();
    this.filter(false);
    this.renderSelected();
    this.resize();
    try {
      const response = await fetch('assets/us-states-10m.json');
      if (!response.ok) throw Error('Boundary file unavailable');
      const topology = await response.json();
      this.states = topojson.feature(topology, topology.objects.states);
      this.stateLayer = L.geoJSON(this.states, {
        style: {color: '#47606e', weight: .8, fillColor: '#223744', fillOpacity: .72}, interactive: false
      }).addTo(this.map).bringToBack();
      L.geoJSON(this.states, {style: {color: '#47606e', weight: .8, fillColor: '#223744', fillOpacity: .72},
        interactive: false}).addTo(this.locator);
      // A world polygon with US land rings cut out. The even-odd fill leaves US imagery unobscured.
      const nation = topojson.feature(topology, topology.objects.nation);
      const features = nation.type === 'FeatureCollection' ? nation.features : [nation];
      const polygons = features.flatMap(feature => feature.geometry.type === 'MultiPolygon' ?
        feature.geometry.coordinates : [feature.geometry.coordinates]);
      const rings = [[[-85, -180], [-85, 180], [85, 180], [85, -180]],
        ...polygons.map(polygon => polygon[0].map(([lon, lat]) => [lat, lon]))];
      this.outsideMask = L.polygon(rings, {stroke: false, fillColor: '#18242d', fillOpacity: .72,
        fillRule: 'evenodd', interactive: false});
      this.updateBasemapStyle();
      for (const feature of this.states.features) {
        const bounds = L.geoJSON(feature).getBounds(), center = bounds.getCenter();
        if (center.lat < 24 || center.lat > 50 || center.lng < -125 || center.lng > -66) continue;
        const label = L.tooltip({permanent: true, direction: 'center', className: 'state-label'}).setLatLng(center)
          .setContent(esc(feature.properties.name)).addTo(this.map);
        this.stateLabels.push({label, name: feature.properties.name});
      }
      this.updateStateLabels();
    } catch (error) { text('atlas-map-status', 'Boundaries unavailable · saved scene coordinates still shown'); }
  }
};
