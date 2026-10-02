(() => {
  'use strict';

  const REVIEW_URL = 'portable_bundles/GTM_Model6_emit_2025_intake/review.json';
  const $ = (id) => document.getElementById(id);

  async function start() {
    const response = await fetch(REVIEW_URL, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Receipt unavailable (${response.status})`);
    const review = await response.json();
    const receiptUrl = new URL(REVIEW_URL, window.location.href);
    const imageUrl = (name) => new URL(name, receiptUrl).toString();
    const [left, bottom, right, top] = review.bounds;
    const bounds = [[bottom, left], [top, right]];
    const map = L.map('intake-map', {
      crs: L.CRS.Simple,
      zoomControl: true,
      zoomSnap: 0.25,
      zoomDelta: 0.5,
      minZoom: -4,
      maxZoom: 4,
      attributionControl: false,
      maxBoundsViscosity: 0.72,
    });
    const rgb = L.imageOverlay(imageUrl(review.rgb), bounds, { alt: 'Percentile-stretched Sentinel RGB' });
    const retrieval = L.imageOverlay(imageUrl(review.retrieval), bounds, { opacity: 0, className: 'native-pixels', alt: 'EMIT matched-filter retrieval' });
    const mask = L.imageOverlay(imageUrl(review.mask), bounds, { opacity: 0.9, className: 'native-pixels', alt: 'Provider mask boundary' });
    rgb.addTo(map);
    retrieval.addTo(map);
    mask.addTo(map);
    const marker = L.circleMarker([review.location.y, review.location.x], {
      radius: 5, color: '#f7f8ec', weight: 2, fillColor: '#365d49', fillOpacity: 1,
    }).addTo(map);
    marker.bindTooltip('Listed facility location; emission origin unverified', { direction: 'top', offset: [0, -5] });
    map.fitBounds(bounds, { padding: [28, 28] });
    L.control.scale({ position: 'bottomright', metric: true, imperial: false, maxWidth: 120 }).addTo(map);

    $('title').textContent = review.title;
    $('offset').textContent = Number(review.offsetMinutes).toFixed(1);
    $('clear').textContent = Number(review.clearPercent).toFixed(1);
    $('scene-id').textContent = review.sentinelId || 'Sentinel-2 L2A';
    $('range').textContent = `${Number(review.retrievalRange[0]).toPrecision(3)} to ${Number(review.retrievalRange[1]).toPrecision(3)}; units unconfirmed`;
    $('full-positive').src = imageUrl(review.fullPositive);
    $('full-control').src = imageUrl(review.fullControl);
    $('positive-link').href = imageUrl(review.fullPositive);
    $('control-link').href = imageUrl(review.fullControl);
    $('receipt-link').href = REVIEW_URL;

    const rgbButton = $('show-rgb');
    const mfButton = $('show-mf');
    const stamp = $('map-stamp');
    function setBackground(which) {
      const showRetrieval = which === 'mf';
      rgb.setOpacity(showRetrieval ? 0 : 1);
      retrieval.setOpacity(showRetrieval ? 1 : 0);
      rgbButton.classList.toggle('selected', !showRetrieval);
      rgbButton.setAttribute('aria-pressed', String(!showRetrieval));
      mfButton.classList.toggle('selected', showRetrieval);
      mfButton.setAttribute('aria-pressed', String(showRetrieval));
      stamp.textContent = showRetrieval ? 'EMIT MATCHED-FILTER RETRIEVAL' : 'SENTINEL-2 RGB';
    }
    rgbButton.addEventListener('click', () => setBackground('rgb'));
    mfButton.addEventListener('click', () => setBackground('mf'));
    $('mask-toggle').addEventListener('change', (event) => {
      if (event.target.checked) mask.addTo(map); else map.removeLayer(mask);
    });
    $('opacity').addEventListener('input', (event) => {
      const value = Number(event.target.value);
      mask.setOpacity(value);
      $('opacity-value').textContent = `${Math.round(value * 100)}%`;
    });
    $('reset-view').addEventListener('click', () => map.fitBounds(bounds, { padding: [28, 28] }));
    map.setMaxBounds(bounds);
  }

  start().catch((error) => {
    console.error('Intake review could not start:', error);
    $('load-error').hidden = false;
  });
})();
