(() => {
  'use strict';
  const MODELS = [
    ['Native control','GTM_Model8_native_control_20261031_validation_final_epoch039','Local · epoch 39'],
    ['Temporal CNN','GTM_Model9_shared_temporal_20261031_validation_final_epoch090','Local · epoch 90'],
    ['Temporal attention','GTM_Model10_temporal_attention_20261031_validation_final_epoch039','Local · epoch 39'],
    ['Model 11','GTM_Model11_mars_unet_attention_20261101_validation_final_epoch049','Local · epoch 49'],
    ['Model 12','GTM_Model12_mars_unet_attn128_ffn1_20261101_validation_final_epoch041','Local · epoch 41'],
    ['Model 13','GTM_Model13_mars_unet_attn128_ffn2_20261101_validation_final_epoch022','Local · epoch 22'],
    ['Model 14','GTM_Model14_mars_unet_attn256_ffn1_20261101_validation_final_epoch041','Local · epoch 41'],
    ['Official MARS-S2L','GTM_Task0010_official_MARSS2L_20250326_saved24_20261102','Released onshore checkpoint']
  ];
  const $ = id => document.getElementById(id);
  let catalog = new Map(), scenes = [], generation = 0;
  const grids = new Map();
  const percent = n => typeof n === 'number' && Number.isFinite(n) ? `${(100*n).toFixed(1)}%` : 'N/A';
  function match(a,b) {
    return !!a && !!b && a.id === b.id && !!a.source && a.source === b.source
      && a.coordinateSystem === b.coordinateSystem
      && JSON.stringify(a.imageSize || null) === JSON.stringify(b.imageSize || null)
      && JSON.stringify(a.bounds || null) === JSON.stringify(b.bounds || null);
  }
  function image(path, cls, alt) {
    const img = document.createElement('img'); img.src = new URL(path, location.href).href; img.className = cls; img.alt = alt;
    return img;
  }
  async function gridMatches(base, scene) {
    if (!base.grid || !scene.grid) return false;
    async function read(path) {
      if (typeof path === 'object') return path;
      if (!grids.has(path)) grids.set(path,fetch(new URL(path,location.href)).then(r => { if (!r.ok) throw Error('grid unavailable'); return r.json(); }));
      return grids.get(path);
    }
    const [a,b] = await Promise.all([read(base.grid),read(scene.grid)]);
    return a.width === b.width && a.height === b.height && a.crs === b.crs
      && JSON.stringify(a.truth) === JSON.stringify(b.truth)
      && JSON.stringify(a.valid) === JSON.stringify(b.valid);
  }
  async function card(model, base) {
    const [name,id,detail] = model, run = catalog.get(id), scene = run?.scenes?.find(s => s.id === base.id);
    const article = document.createElement('article'); article.className = `card ${name.startsWith('Official')?'official':''}`;
    const head = document.createElement('div'); head.className = 'card-head';
    const badge = document.createElement('span'); badge.className = 'badge'; badge.textContent = name.startsWith('Official')?'REFERENCE':'SAVED';
    const h = document.createElement('h3'); h.textContent = name;
    const sub = document.createElement('div'); sub.className = 'sub'; sub.textContent = detail;
    head.append(badge,h,sub); article.append(head);
    const frame = document.createElement('div'); frame.className = 'image-frame';
    const foot = document.createElement('div'); foot.className = 'card-foot';
    let unavailable = !run ? 'Final experiment is not in this catalog.' : !scene ? 'No saved prediction for this scene.' : !match(base,scene) ? 'Source image or bounds differ from selected scene.' : scene.decisionThreshold !== 0.5 ? 'Saved pixel cutoff is not 0.50.' : !base.rgb || !base.truthImage || !scene.predictionImage ? 'RGB, reference, or prediction asset missing.' : '';
    if (!unavailable) {
      try { if (!(await gridMatches(base,scene))) unavailable = 'Reference or valid-pixel grid differs from selected scene.'; }
      catch { unavailable = 'Saved grid could not be verified.'; }
    }
    if (unavailable) {
      const missing = document.createElement('div'); missing.className = 'missing'; missing.textContent = unavailable; frame.append(missing);
      foot.className += ' error'; foot.textContent = `Unavailable · ${id}`;
    } else {
      const layers = [image(base.rgb,'rgb',''),image(scene.predictionImage,'pred',`${name} prediction`),image(base.truthImage,'truth','Shared reference')];
      for (const layer of layers) layer.onerror = () => {
        frame.replaceChildren(); const missing = document.createElement('div'); missing.className = 'missing';
        missing.textContent = 'Saved image could not load.'; frame.append(missing);
        foot.className += ' error'; foot.textContent = 'Unavailable · image request failed';
      };
      frame.append(...layers);
      const legend = document.createElement('div'); legend.className = 'legend'; legend.textContent = 'Orange prediction · Cyan reference';
      const metrics = document.createElement('div'); metrics.className = 'metrics';
      const m = scene.nativeMetrics || {};
      for (const [label,value] of [['F1',percent(m.f1)],['IoU',percent(m.iou)],['FP',m.fp ?? 'N/A'],['FN',m.fn ?? 'N/A']]) {
        const span = document.createElement('span'); span.textContent = `${label} ${value}`; metrics.append(span);
      }
      const outcome = document.createElement('div'); outcome.className = 'outcome'; outcome.textContent = `Scene outcome: ${scene.outcome || 'unrecorded'} · ${m.evaluated_pixels ?? 'unknown'} scored pixels`;
      foot.append(metrics,outcome); article.append(legend);
    }
    article.append(frame,foot); return article;
  }
  async function render() {
    const current = ++generation;
    const base = scenes.find(s => s.id === $('scene').value);
    if (!base) return;
    $('scene-name').textContent = base.name || base.id;
    $('scene-detail').textContent = `${base.id} · ${base.date || 'date unrecorded'} · ${base.source || 'source unrecorded'}`;
    $('scene-type').textContent = base.label === 'PLUME' ? 'PLUME / POSITIVE' : base.label === 'CLEAR' ? 'CLEAR / NEGATIVE' : 'LABEL UNRECORDED';
    $('status').textContent = 'Checking saved grids and predictions…';
    const cards = await Promise.all(MODELS.map(model => card(model,base)));
    if (current !== generation) return;
    $('cards').replaceChildren(...cards);
    const count = cards.filter(node => !node.querySelector('.missing')).length;
    $('status').textContent = `${count} of 8 saved predictions verified for this exact scene. All scenes remain selectable.`;
    history.replaceState(null,'',`?scene=${encodeURIComponent(base.id)}`);
  }
  async function load() {
    $('status').textContent = 'Loading saved experiment catalog…';
    try {
      const response = await fetch('api/experiments',{cache:'no-store'});
      if (!response.ok) throw Error(`catalog returned HTTP ${response.status}`);
      const data = await response.json(), runs = Array.isArray(data) ? data : data.experiments;
      if (!Array.isArray(runs)) throw Error('catalog has no experiments list');
      catalog = new Map(runs.map(run => [run.id,run]));
      const base = catalog.get(MODELS[0][1]) || MODELS.map(m => catalog.get(m[1])).find(Boolean);
      if (!base) throw Error('none of the eight final experiments is available in this viewer');
      scenes = base.scenes || [];
      const select = $('scene'); select.replaceChildren();
      for (const scene of scenes) {
        const option = document.createElement('option'); option.value = scene.id;
        option.textContent = `${scene.label === 'PLUME' ? 'Plume' : scene.label === 'CLEAR' ? 'Clear' : 'Unscored'} · ${scene.name || scene.id}`;
        select.append(option);
      }
      $('scene-count').textContent = `${scenes.length} saved scenes · no favorable-case filter`;
      select.disabled = scenes.length === 0;
      const requested = new URLSearchParams(location.search).get('scene');
      if (requested && scenes.some(s => s.id === requested)) select.value = requested;
      render();
    } catch (error) {
      $('status').textContent = `Could not load saved evidence: ${error.message}. Start the local viewer or reconnect to its private address, then refresh.`;
      $('scene').disabled = true;
    }
  }
  $('scene').addEventListener('change',render);
  $('retry').addEventListener('click',load);
  load();
})();
