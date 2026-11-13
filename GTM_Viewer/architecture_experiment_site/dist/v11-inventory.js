'use strict';
(async function () {
  const root = document.getElementById('v11-inventory');
  if (!root) return;
  const escape = GTMSceneReview.escape, percent = n => n == null ? 'n/a' : (n * 100).toFixed(1) + '%';
  let category = 'all', search = '', country = 'all', data;
  try {
    const response = await fetch('data/v11-predictions.json?v=20261030c');
    if (!response.ok) throw Error('HTTP ' + response.status);
    data = await response.json();
    document.getElementById('v11-groups').innerHTML = [{id: 'all', label: 'All predictions', count: data.total}, ...data.groups]
      .map(g => '<button class="inventory-group" data-category="' + g.id + '" aria-pressed="' + (g.id === category) + '"><strong>' +
        g.count + '</strong><span>' + escape(g.label) + '</span></button>').join('');
    document.getElementById('v11-country').innerHTML = '<option value="all">All countries</option>' +
      [...new Set(data.predictions.map(s => s.country))].sort().map(c => '<option>' + escape(c) + '</option>').join('');
    function render() {
      const items = data.predictions.filter(item => (category === 'all' || item.category === category) &&
        (country === 'all' || country === item.country) &&
        [item.id, item.country, item.date, item.categoryLabel].join(' ').toLowerCase().includes(search));
      document.querySelectorAll('.inventory-group').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.category === category)));
      document.getElementById('v11-count').textContent = 'Showing ' + items.length + ' of ' + data.total + ' saved predictions';
      document.getElementById('v11-table-body').innerHTML = items.map(item => '<tr><td><a class="inventory-scene" href="' + escape(item.viewerUrl) + '">' +
        '<img loading="lazy" src="' + escape(item.rgb) + '" alt=""><span><strong>' + escape(item.country) + '</strong><small>' +
        escape(item.date) + ' · ' + escape(item.shortId) + '</small></span></a></td><td><span class="inventory-badge" data-category="' +
        item.category + '">' + escape(item.categoryLabel) + '</span></td><td>' + (item.label === 'PLUME' ? percent(item.iou) : 'n/a') +
        '</td><td>' + percent(item.precision) + '</td><td>' + percent(item.recall) + '</td><td>' + item.tp.toLocaleString() +
        '</td><td>' + item.fp.toLocaleString() + '</td><td>' + item.fn.toLocaleString() + '</td><td><a class="inspect-scene" href="' +
        escape(item.viewerUrl) + '">Inspect ↗</a></td></tr>').join('');
      document.getElementById('v11-empty').hidden = items.length > 0;
      const group = data.groups.find(g => g.id === category);
      document.getElementById('v11-group-rule').textContent = group?.rule ||
        'All 198 development observations are included. The groups use the saved mask at cutoff 0.65. No predictions were retrained or changed.';
      const first = items[0];
      const open = document.getElementById('v11-open-filter');
      open.hidden = !first;
      if (first) {
        const url = new URL(first.viewerUrl, location.href);
        if (category !== 'all') url.searchParams.set('outcome', category);
        open.href = url.pathname.split('/').pop() + url.search + url.hash;
      }
    }
    document.querySelectorAll('.inventory-group').forEach(b => b.onclick = () => { category = b.dataset.category; render(); });
    document.getElementById('v11-search').oninput = e => { search = e.target.value.trim().toLowerCase(); render(); };
    document.getElementById('v11-country').onchange = e => { country = e.target.value; render(); };
    document.getElementById('v11-reset').onclick = () => {
      category = country = 'all'; search = '';
      document.getElementById('v11-search').value = ''; document.getElementById('v11-country').value = 'all'; render();
    };
    render();
  } catch (error) {
    document.getElementById('v11-count').textContent = 'The prediction list could not load. Refresh to retry.';
    console.error(error);
  }
})();
