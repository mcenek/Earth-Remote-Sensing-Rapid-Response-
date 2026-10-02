'use strict';

async function connectViewer() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch('api/experiments', {cache: 'no-store', signal: controller.signal});
    if (!response.ok) throw Error('Results service returned HTTP ' + response.status);
    window.GTM_DATA = await response.json();
    const script = document.createElement('script');
    script.src = 'GTM_Model0.js';
    script.onerror = () => {
      document.getElementById('live-status').textContent = 'Viewer could not load';
      document.getElementById('message').textContent = 'Refresh this page to retry.';
    };
    document.body.append(script);
  } catch (error) {
    document.body.dataset.connection = 'offline';
    document.getElementById('live-status').textContent = 'Waiting for backend';
    document.getElementById('message').textContent = 'Start the local viewer or reconnect to its private address. Retrying in 5 seconds.';
    setTimeout(connectViewer, 5000);
  } finally {
    clearTimeout(timeout);
  }
}
connectViewer();
