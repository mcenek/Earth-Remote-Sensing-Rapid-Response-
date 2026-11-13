'use strict';
viewerLive.start = () => {
  document.body.dataset.connection = 'connected';
  text('backend-name', 'Published research snapshot');
  text('live-status', 'Saved results · not live');
  text('live-updated', '30 Oct 2026');
};
