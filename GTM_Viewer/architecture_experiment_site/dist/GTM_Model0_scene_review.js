'use strict';

// These are display groups for the saved cutoff, never new model-selection rules.
(function (root) {
  const runId = 'GTM_Model7_hard_v11_20261025_candidate';
  const groups = [
    {id: 'strong', label: 'Strong match', rule: 'Labeled plume; native IoU at least 50%.'},
    {id: 'partial', label: 'Partial match', rule: 'Labeled plume; native IoU from 10% to below 50%.'},
    {id: 'weak', label: 'Weak trace', rule: 'Some correct plume pixels, but native IoU below 10%.'},
    {id: 'miss', label: 'Missed plume', rule: 'No correct plume pixels. Marks elsewhere can still be false alarms.'},
    {id: 'false', label: 'False alarm', rule: 'Reviewed no-plume scene with at least one marked native pixel.'},
    {id: 'clear', label: 'Correct clear', rule: 'Reviewed no-plume scene with no marked native pixels.'}
  ];
  function category(item) {
    const m = item?.nativeMetrics;
    if (!m || !['tp', 'fp', 'fn', 'tn'].every(k => Number.isInteger(m[k]) && m[k] >= 0) ||
        m.tp + m.fp + m.fn + m.tn === 0) return 'unknown';
    if (m.tp + m.fn === 0) return m.fp ? 'false' : 'clear';
    if (m.tp === 0) return 'miss';
    const iou = m.tp / (m.tp + m.fp + m.fn);
    return iou >= .5 ? 'strong' : iou >= .1 ? 'partial' : 'weak';
  }
  const escape = value => String(value ?? '').replace(/[&<>"']/g,
    c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  let api, activeEntry, filter = 'all', query = '';
  const el = id => document.getElementById(id);
  function list(entry) {
    if (entry?.id !== runId) return entry?.scenes || [];
    if (activeEntry && activeEntry !== entry.id) { filter = 'all'; query = ''; }
    activeEntry = entry.id;
    const rank = key => groups.findIndex(g => g.id === key);
    return entry.scenes.filter(item => (filter === 'all' || category(item) === filter) &&
      [item.id, item.name, item.date].join(' ').toLowerCase().includes(query.trim().toLowerCase()))
      .sort((a, b) => rank(category(a)) - rank(category(b)) ||
        (b.nativeMetrics.iou || 0) - (a.nativeMetrics.iou || 0) || b.nativeMetrics.fp - a.nativeMetrics.fp ||
        a.name.split('|')[1].trim().localeCompare(b.name.split('|')[1].trim()) ||
        a.date.localeCompare(b.date) || a.id.localeCompare(b.id));
  }
  function refresh(entry, selected) {
    if (!api || !el('review-browser')) return;
    const enabled = entry?.id === runId;
    el('review-browser').hidden = !enabled;
    if (!enabled) { activeEntry = entry?.id; return; }
    const items = list(entry), index = items.findIndex(item => item.id === selected?.id);
    const counts = Object.fromEntries(groups.map(g => [g.id, 0]));
    entry.scenes.forEach(item => { const key = category(item); counts[key] = (counts[key] || 0) + 1; });
    el('review-outcome').innerHTML = '<option value="all">All predictions (' + entry.scenes.length + ')</option>' +
      groups.map(g => '<option value="' + g.id + '">' + g.label + ' (' + counts[g.id] + ')</option>').join('');
    el('review-outcome').value = filter;
    el('review-search').value = query;
    el('review-position').textContent = index < 0 ? items.length + ' results' : (index + 1) + ' of ' + items.length;
    el('review-prev').disabled = index <= 0;
    el('review-next').disabled = index < 0 || index >= items.length - 1;
    const key = category(selected), group = groups.find(g => g.id === key);
    el('review-badge').textContent = group?.label || 'No matching scene';
    el('review-badge').dataset.category = key;
    el('review-rule').textContent = group ? group.rule + ' Saved cutoff 0.65; development evidence.' :
      'No saved predictions match these filters. Clear the search or choose All predictions.';
    // An outside selection can be opened from the map or the separate scene browser.
    if (selected && index < 0) {
      const option = document.createElement('option');
      option.value = selected.id; option.textContent = 'Outside current filter: ' + selected.name;
      el('comparison-scene').prepend(option); el('comparison-scene').value = selected.id;
    }
  }
  function start(callbacks) {
    api = callbacks;
    const params = new URLSearchParams(location.search);
    filter = groups.some(g => g.id === params.get('outcome')) ? params.get('outcome') : 'all';
    query = params.get('rq') || '';
    function choose() {
      const items = list(api.getExperiment());
      const current = api.getScene();
      const chosen = items.find(item => item.id === current?.id) || items[0];
      const url = new URL(location.href);
      filter === 'all' ? url.searchParams.delete('outcome') : url.searchParams.set('outcome', filter);
      query ? url.searchParams.set('rq', query) : url.searchParams.delete('rq');
      history.replaceState(null, '', url);
      api.selectScene(chosen?.id || null);
    }
    el('review-outcome').onchange = event => { filter = event.target.value; choose(); };
    el('review-search').oninput = event => { query = event.target.value; choose(); };
    el('review-reset').onclick = () => { filter = 'all'; query = ''; choose(); };
    for (const [id, offset] of [['review-prev', -1], ['review-next', 1]]) {
      el(id).onclick = () => {
        const items = list(api.getExperiment()), index = items.findIndex(item => item.id === api.getScene()?.id);
        if (index >= 0 && items[index + offset]) api.selectScene(items[index + offset].id);
      };
    }
  }
  root.GTMSceneReview = {runId, groups, category, list, refresh, start, escape};
})(window);
