'use strict';
// This public site is a deliberate snapshot, not the local live-results server.
fetch('data/research-catalog.json?v=20261030d').then(response => {
  if (!response.ok) throw Error('HTTP ' + response.status);
  return response.json();
}).then(data => {
  window.GTM_DATA = data;
  const script = document.createElement('script');
  script.src = 'GTM_Model0.js?v=20261030d';
  script.onerror = () => { document.getElementById('message').textContent = 'Viewer script could not load.'; };
  document.body.append(script);
}).catch(error => {
  document.getElementById('live-status').textContent = 'Snapshot unavailable';
  document.getElementById('message').textContent = 'Could not load saved results: ' + error.message;
});
