// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 WCKDSMAT contributors
// Speech Movement Data Viewer: smooth playhead across all panels + keyboard shortcuts.
(function () {
  function offset() {
    const el = document.getElementById('audio-offset');
    return el ? parseFloat(el.textContent) || 0 : 0;
  }
  function tick() {
    const audio = document.getElementById('audio');
    const gd = document.querySelector('#graph .js-plotly-plot');
    const bar = document.getElementById('playhead');
    if (audio && gd && gd._fullLayout && gd._fullLayout.xaxis && bar) {
      const xa = gd._fullLayout.xaxis, size = gd._fullLayout._size;
      const t = audio.currentTime + offset();
      const [lo, hi] = xa.range;
      if (window.smvStopAt !== undefined && audio.currentTime >= window.smvStopAt) {
        audio.pause(); window.smvStopAt = undefined;
      }
      if (t >= lo && t <= hi) {
        bar.style.display = 'block';
        bar.style.left = (xa._offset + xa.l2p(t)) + 'px';
        bar.style.top = size.t + 'px';
        bar.style.height = size.h + 'px';
      } else {
        bar.style.display = 'none';
      }
    }
    window.requestAnimationFrame(tick);
  }
  window.requestAnimationFrame(tick);

  document.addEventListener('keydown', function (e) {
    const tag = (e.target && e.target.tagName) || '';
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) || (e.target && e.target.isContentEditable)) return;
    const audio = document.getElementById('audio');
    if (e.key === 'f' || e.key === 'F') {
      const b = document.getElementById('flag-btn'); if (b) { b.click(); e.preventDefault(); }
    } else if (e.key === ' ' && audio) {
      if (audio.paused) { audio.play(); } else { audio.pause(); }
      e.preventDefault();
    } else if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && audio) {
      audio.currentTime = Math.max(0, audio.currentTime + (e.key === 'ArrowLeft' ? -0.1 : 0.1));
      e.preventDefault();
    }
  });
})();
