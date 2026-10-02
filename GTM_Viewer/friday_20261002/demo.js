(() => {
  'use strict';

  const data = window.GTM_FRIDAY_DATA;
  const $ = (id) => document.getElementById(id);
  const state = { index: 0, scale: 1, x: 0, y: 0, drag: null };
  const hasFullViewer = location.protocol !== 'file:' && location.pathname.endsWith('/friday_20261002/index.html');
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  if (!data || !Array.isArray(data.cases) || data.cases.length === 0) {
    $('load-error').hidden = false;
    $('summary').textContent = 'The offline review bundle could not be loaded.';
    document.querySelectorAll('.workspace button').forEach((button) => { button.disabled = true; });
    return;
  }

  document.title = `${data.title || 'GTM Model6'} | Friday research review`;
  $('page-title').textContent = data.title || 'Friday research review';
  $('meeting-date').textContent = data.meetingDate || '';
  $('build-date').textContent = data.builtDate ? `Prepared ${data.builtDate}` : '';
  $('summary').textContent = data.summary || '';
  if (data.pdf) $('brief-link').href = data.pdf;
  $('intake-link').hidden = !hasFullViewer;

  function textElement(tag, className, content) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = content == null ? '' : String(content);
    return element;
  }

  function applyTransform() {
    const transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
    document.querySelectorAll('.image-viewport img').forEach((image) => {
      image.style.transform = transform;
    });
    $('zoom-level').textContent = `${Math.round(state.scale * 100)}%`;
    $('zoom-out').disabled = state.scale <= 1;
    $('zoom-in').disabled = state.scale >= 5;
  }

  function resetView() {
    state.scale = 1;
    state.x = 0;
    state.y = 0;
    applyTransform();
  }

  function zoomBy(factor) {
    state.scale = clamp(Math.round(state.scale * factor * 100) / 100, 1, 5);
    if (state.scale === 1) { state.x = 0; state.y = 0; }
    applyTransform();
  }

  function makeImagePanel(panel, index) {
    const figure = textElement('figure', 'image-panel', '');
    const top = textElement('div', 'panel-top', '');
    top.append(textElement('h4', '', panel.title || `View ${index + 1}`), textElement('span', 'panel-index', `0${index + 1}`));
    figure.append(top);

    const viewport = textElement('div', 'image-viewport', '');
    viewport.setAttribute('role', 'img');
    viewport.setAttribute('aria-label', panel.title || `View ${index + 1}`);

    if (panel.image) {
      const image = document.createElement('img');
      image.src = panel.image;
      image.alt = panel.title || `View ${index + 1}`;
      image.draggable = false;
      image.addEventListener('error', () => {
        viewport.classList.add('empty', 'broken');
        image.remove();
        addEmpty(viewport, 'Image unavailable', 'This local image could not be opened. Check the review bundle.');
      });
      viewport.append(image);
      viewport.addEventListener('wheel', (event) => {
        event.preventDefault();
        zoomBy(event.deltaY < 0 ? 1.2 : 1 / 1.2);
      }, { passive: false });
      viewport.addEventListener('pointerdown', (event) => {
        if (event.button !== 0 || state.scale <= 1 || !viewport.querySelector('img')) return;
        state.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, startX: state.x, startY: state.y };
        viewport.classList.add('dragging');
        viewport.setPointerCapture(event.pointerId);
      });
      viewport.addEventListener('pointermove', (event) => {
        if (!state.drag || state.drag.id !== event.pointerId) return;
        state.x = state.drag.startX + event.clientX - state.drag.x;
        state.y = state.drag.startY + event.clientY - state.drag.y;
        applyTransform();
      });
      const endDrag = (event) => {
        if (!state.drag || state.drag.id !== event.pointerId) return;
        state.drag = null;
        viewport.classList.remove('dragging');
      };
      viewport.addEventListener('pointerup', endDrag);
      viewport.addEventListener('pointercancel', endDrag);
      viewport.addEventListener('lostpointercapture', endDrag);
    } else {
      viewport.classList.add('empty');
      addEmpty(viewport, 'No model output', panel.emptyReason || 'No image is available for this view.');
    }
    figure.append(viewport, textElement('figcaption', 'panel-caption', panel.caption || ''));
    return figure;
  }

  function addEmpty(container, title, reason) {
    const content = textElement('div', 'empty-content', '');
    content.append(textElement('span', 'empty-mark', '∅'), textElement('span', 'empty-title', title), textElement('span', 'empty-reason', reason));
    container.append(content);
  }

  function renderCaseTabs() {
    const list = $('case-list');
    list.replaceChildren();
    data.cases.forEach((item, index) => {
      const button = textElement('button', `case-tab${index === state.index ? ' active' : ''}`, '');
      button.type = 'button';
      button.dataset.caseIndex = String(index);
      button.setAttribute('aria-pressed', String(index === state.index));
      button.append(textElement('span', 'tab-kind', item.kind === 'data' ? 'NEW DATA' : 'SAVED RUN'), textElement('span', '', item.label || item.id || `Case ${index + 1}`));
      button.addEventListener('click', () => selectCase(index));
      list.append(button);
    });
  }

  function selectCase(index) {
    state.index = clamp(index, 0, data.cases.length - 1);
    state.drag = null;
    resetView();
    const item = data.cases[state.index];
    renderCaseTabs();
    const activeTab = $('case-list').querySelector('.active');
    activeTab?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    const isData = item.kind === 'data';
    $('case-kind').textContent = isData ? 'NEW DATA · NOT SCORED' : 'SAVED PREDICTION · FAILED RUN';
    $('case-kind').classList.toggle('prediction', !isData);
    $('case-title').textContent = item.label || item.id || `Case ${state.index + 1}`;
    $('case-date').textContent = item.date || '';
    $('case-description').textContent = item.description || '';
    $('case-lesson').textContent = item.lesson || '';
    $('panel-grid').replaceChildren(...(Array.isArray(item.panels) ? item.panels.slice(0, 3).map(makeImagePanel) : []));

    const metrics = $('case-metrics');
    metrics.replaceChildren();
    (Array.isArray(item.metrics) ? item.metrics : []).forEach((metric) => {
      const group = textElement('div', 'metric', '');
      group.append(textElement('span', 'metric-label', metric.label), textElement('span', 'metric-value', metric.value));
      metrics.append(group);
    });
    const viewer = $('full-viewer');
    viewer.href = item.fullViewerUrl || '../index.html';
    viewer.hidden = !hasFullViewer || !item.fullViewerUrl;
    $('case-prev').disabled = state.index === 0;
    $('case-next').disabled = state.index === data.cases.length - 1;

    const notice = $('prediction-notice');
    notice.hidden = isData;
    if (!isData) {
      const iou = Number(data.aggregate?.iou);
      const score = Number.isFinite(iou) ? `${(iou * 100).toFixed(2)}% pooled native IoU` : 'failed quality gate';
      notice.textContent = `Run-wide result: ${score} across ${data.aggregate?.scenes || '?'} scenes. These selected scenes are diagnostic examples, not representative performance.`;
    }
    applyTransform();
  }

  function renderArchitecture() {
    const image = $('architecture-image');
    if (data.architectureImage) image.src = data.architectureImage;
    image.addEventListener('error', () => {
      image.alt = 'Architecture image unavailable in this local bundle';
      image.style.display = 'none';
      $('architecture-caption').textContent = 'Architecture image unavailable. Check the review bundle.';
    });
    $('architecture-caption').textContent = data.architectureCaption || '';
    const steps = $('next-steps');
    steps.replaceChildren();
    (Array.isArray(data.nextSteps) ? data.nextSteps : []).forEach((step) => {
      const item = textElement('article', 'plan-item', '');
      const head = textElement('div', 'plan-head', '');
      head.append(textElement('h3', '', step.title), textElement('span', 'plan-status', step.status || ''));
      item.append(head, textElement('p', '', step.detail || ''));
      steps.append(item);
    });
  }

  function renderDecisions() {
    const list = $('decision-list');
    list.replaceChildren();
    (Array.isArray(data.decisions) ? data.decisions : []).forEach((decision) => {
      const item = textElement('li', 'decision-item', '');
      const content = textElement('div', '', '');
      content.append(textElement('h3', '', decision.question), textElement('p', '', decision.why || ''));
      item.append(content);
      list.append(item);
    });
  }

  $('zoom-in').addEventListener('click', () => zoomBy(1.25));
  $('zoom-out').addEventListener('click', () => zoomBy(1 / 1.25));
  $('zoom-reset').addEventListener('click', resetView);
  $('case-prev').addEventListener('click', () => selectCase(state.index - 1));
  $('case-next').addEventListener('click', () => selectCase(state.index + 1));
  document.addEventListener('keydown', (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
    if (event.key === 'ArrowRight' && state.index < data.cases.length - 1) selectCase(state.index + 1);
    if (event.key === 'ArrowLeft' && state.index > 0) selectCase(state.index - 1);
  });

  renderArchitecture();
  renderDecisions();
  selectCase(0);
})();
