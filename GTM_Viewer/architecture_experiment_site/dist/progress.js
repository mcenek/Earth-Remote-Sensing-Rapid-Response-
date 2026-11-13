'use strict';
const cases = [
  {id:'cde9d349', label:'LABELED PLUME', badge:'STRONG MATCH', title:'A plume the model traces', description:'This US scene has 86.4% intersection-over-union for the saved v11 mask. It is one example, not the run-wide score.'},
  {id:'df4817f5', label:'LABELED PLUME', badge:'COMPLETE MISS', title:'A plume the model misses', description:'The reviewed plume is visible in cyan, but v11 marks no plume pixels in this Uzbekistan scene.'},
  {id:'eaf034e7', label:'REVIEWED NO-PLUME', badge:'FALSE ALARM', title:'A clear scene with a mark', description:'The model marked 892 false-positive pixels in this US no-plume scene. Clear labels do not prove zero methane.'}
];
const escapeText = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function image(src, alt, extra = '') { return `<img src="${escapeText(src)}" alt="${escapeText(alt)}" ${extra}>`; }
fetch('data/progress-cases.json?v=20261030a').then(r => {if (!r.ok) throw Error('HTTP '+r.status); return r.json();}).then(data => {
  document.getElementById('cases').innerHTML = cases.map(example => {
    const item = data[example.id];
    if (!item?.rgb || !item?.predictionImage || !item?.truthImage) throw Error('Missing scene imagery: '+example.id);
    const link = 'index.html?experiment=GTM_Model7_hard_v11_20261025_candidate&scene='+encodeURIComponent(item.id)+'&layout=panels#compare';
    return `<article class="case"><div class="case-header"><span>${escapeText(example.label)}</span><b class="${example.badge === 'STRONG MATCH' ? '' : 'warn'}">${escapeText(example.badge)}</b></div><div class="case-tiles"><div class="tile">${image(item.rgb,'Sentinel input', 'loading="lazy"')}<span class="tile-label">ORIGINAL</span></div><div class="tile">${image(item.rgb,'', 'loading="lazy"')}${image(item.predictionImage,'Model prediction', 'loading="lazy"')}${image(item.truthOutlineImage || item.truthImage,'Reviewed plume annotation', 'loading="lazy"')}<span class="tile-label">PREDICTION + LABEL</span></div></div><div class="case-body"><h3>${escapeText(example.title)}</h3><p>${escapeText(example.description)}</p><a href="${escapeText(link)}">Inspect this scene ↗</a></div></article>`;
  }).join('');
}).catch(error => { document.getElementById('cases').innerHTML = '<div class="loading">Saved images could not load. Open the viewer to retry.</div>'; console.error(error); });
