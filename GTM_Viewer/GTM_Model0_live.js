'use strict';

// Catalog polling only reads published results. It never starts training or inference.
const viewerLive = {
  timer: null,
  request: null,
  failures: 0,
  imports: new Map(),

  validate(catalog, previous = []) {
    if (!catalog || !Array.isArray(catalog.experiments)) throw Error('Invalid experiment catalog');
    const warnings = [...(catalog.local?.warnings || [])], seen = new Set(), valid = [];
    for (const entry of catalog.experiments) {
      try {
        if (seen.has(entry.id)) throw Error('Duplicate experiment id');
        validateExperiment(entry);
        seen.add(entry.id);
        valid.push(entry);
      } catch (error) {
        warnings.push((entry?.name || 'Experiment') + ': ' + error.message);
        const saved = previous.find(item => item.id === entry?.id);
        if (saved && !seen.has(saved.id)) { valid.push(saved); seen.add(saved.id); }
      }
    }
    for (const [id, entry] of this.imports) if (!seen.has(id)) valid.push(entry);
    if (!valid.length) throw Error('No valid experiments available');
    return {...catalog, experiments: valid, local: {...catalog.local, warnings}};
  },

  connection(state, label) {
    document.body.dataset.connection = state;
    text('live-status', label);
    text('backend-name', registry.live?.serverName || 'Saved experiment results');
    if (state === 'connected') {
      const now = new Date();
      $('live-updated').dateTime = now.toISOString();
      $('live-updated').textContent = now.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
      $('live-updated').title = 'Last successful results check: ' + now.toLocaleString();
    }
  },

  schedule() {
    clearTimeout(this.timer);
    if (!$('auto-refresh').checked || document.hidden) return;
    const seconds = Math.min(60, Math.max(2, registry.live?.pollSeconds || 5) * 2 ** Math.min(this.failures, 4));
    this.timer = setTimeout(() => this.refresh(), seconds * 1000);
  },

  async refresh(force = false) {
    if (this.request) return;
    clearTimeout(this.timer);
    this.request = new AbortController();
    $('refresh-experiments').disabled = true;
    const timeout = setTimeout(() => this.request?.abort(), 10000);
    try {
      const headers = registry.live?.revision && !force ? {'If-None-Match': '"' + registry.live.revision + '"'} : {};
      const response = await fetch('api/experiments', {headers, cache: 'no-store', signal: this.request.signal});
      if (response.status !== 304) {
        if (!response.ok) throw Error('HTTP ' + response.status);
        const next = this.validate(await response.json(), registry.experiments);
        if (next.live?.revision !== registry.live?.revision || !next.live?.revision) this.apply(next);
        else if (force) await selectScene(scene?.id, {preserveView: true});
      }
      this.failures = 0;
      this.connection('connected', $('auto-refresh').checked ? 'Connected · watching results' : 'Auto refresh paused');
      text('message', registry.local?.warnings?.join(' · ') || '');
    } catch (error) {
      this.failures++;
      this.connection('offline', 'Backend unavailable · saved view retained');
      text('message', 'Reconnect to the backend to see new results. ' + ($('auto-refresh').checked ? 'Retrying automatically.' : 'Use Refresh now to retry.'));
    } finally {
      clearTimeout(timeout);
      this.request = null;
      $('refresh-experiments').disabled = false;
      this.schedule();
    }
  },

  apply(next) {
    const prior = new Map(registry.experiments.map(e => [e.id, e]));
    const additions = next.experiments.reduce((total, entry) => total + entry.scenes.filter(s =>
      !prior.get(entry.id)?.scenes.some(old => old.id === s.id)).length, 0);
    const selected = next.experiments.find(e => e.id === experiment.id);
    const removedSelection = !selected || scene && !selected.scenes.some(s => s.id === scene.id);
    if (removedSelection) {
      // Keep the evidence currently being inspected, explicitly marked as an older publication.
      next.experiments = next.experiments.filter(e => e.id !== experiment.id).concat(experiment);
      next.local.warnings.push('Selected scene is absent from the latest publication. Its previous saved version is retained.');
    }
    const oldRevision = experiment.live?.revision;
    Object.assign(registry, next);
    if (selected && !removedSelection) experiment = selected;
    populateExperiments();
    updateRunSummary();
    drawList();
    drawExperimentCards();
    text('local-folder', registry.local?.bundleDirectory || 'outputs/GTM_viewer_bundles');
    if (selected && !removedSelection && (oldRevision !== selected.live?.revision || !oldRevision)) {
      selectScene(scene?.id || selected.scenes[0]?.id, {preserveView: true});
    }
    if (additions) {
      $('new-results').hidden = false;
      text('new-results', additions + ' new scene' + (additions === 1 ? '' : 's') + ' · view runs');
    }
    atlas.refresh();
  },

  start() {
    this.connection('connected', 'Connected · watching results');
    $('auto-refresh').onchange = () => {
      if ($('auto-refresh').checked) this.refresh();
      else {
        clearTimeout(this.timer);
        this.connection('paused', 'Auto refresh paused');
      }
    };
    $('refresh-experiments').onclick = () => this.refresh(true);
    $('new-results').onclick = () => { $('new-results').hidden = true; switchView('experiments', true); };
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && $('auto-refresh').checked) this.refresh();
      else clearTimeout(this.timer);
    });
    window.addEventListener('pagehide', () => { clearTimeout(this.timer); this.request?.abort(); });
    this.schedule();
  }
};
