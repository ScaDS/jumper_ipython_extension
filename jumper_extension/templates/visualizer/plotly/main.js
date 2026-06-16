/**
 * main.js  –  Orchestration IIFE for the interactive Plotly visualizer.
 *
 * Depends on data variables embedded by Python directly before this script:
 *   CID, FIGS, YLIMS, BND_F, BND_T, OPTS, LEVS, MAX, MIN_CELL, MAX_CELL, INIT_RNG
 *
 * Depends on component functions (loaded in order before this file):
 *   show_idle_checkbox : initShowIdle
 *   cell_range_slider  : getCellRange, initCellRangeSlider
 *   add_panel_button   : initAddPanelButton, disableAddPanelButton
 *   panel              : createPanelElement, attachPanelEvents,
 *                        buildBoundaryUpdates, xRangeForCells, renderPlotInPanel
 */
(function () {

  /* ── state ────────────────────────────────────────────────────────────── */
  var panelCount    = 0;
  var usedMetrics   = [];
  /* pid → { metricSel: <select>, levelSel: <select> } */
  var panelRegistry = {};

  /* ── helpers ──────────────────────────────────────────────────────────── */

  /** Returns the next unused metric, cycling back to the first when exhausted. */
  function nextMetric() {
    for (var i = 0; i < OPTS.length; i++) {
      if (usedMetrics.indexOf(OPTS[i][1]) < 0) {
        usedMetrics.push(OPTS[i][1]);
        return OPTS[i][1];
      }
    }
    return OPTS.length ? OPTS[0][1] : null;
  }

  /** Returns 'true' or 'false' based on the show-idle checkbox state. */
  function showIdleKey() {
    var cb = document.getElementById(CID + '-show-idle');
    return (cb && cb.checked) ? 'true' : 'false';
  }

  /* ── single-panel render ──────────────────────────────────────────────── */

  /**
   * Assembles layout (boundaries, x-axis range) and calls renderPlotInPanel
   * for the given panel and selected metric/level.
   */
  function renderPlot(pid, metric, level) {
    var plotDiv = document.getElementById(pid + '-plot');
    var key     = showIdleKey();
    var figData = ((FIGS[metric] || {})[level] || {})[key];

    if (!figData) {
      renderPlotInPanel(plotDiv, null, {});
      return;
    }

    var ylim   = (((YLIMS[metric] || {})[level]) || {})[key] || [0, 1];
    var rng    = getCellRange(CID, MIN_CELL, MAX_CELL);
    var bndArr = (key === 'true') ? BND_T : BND_F;
    var xRng   = xRangeForCells(bndArr, rng);

    /* Detect the BALI dual subplot layout: when the figure carries a
       second x/y axis pair the JS overlays each subplot with a different
       BALI colormap (left: Tokens/Second, right: Tokens/Joule). */
    var layoutSrc = figData.layout || {};
    var isDual    = !!(layoutSrc.xaxis2 || layoutSrc.yaxis2);

    /* BALI overlays only make sense when idle periods are hidden, mirroring
       the matplotlib backend.  When active, replace the cell-boundary
       rectangles with BALI rectangles (the cell number annotations are kept
       to preserve orientation). */
    var showBali = (
      key === 'false'
      && typeof isShowBali === 'function'
      && isShowBali(CID)
      && typeof BALI !== 'undefined'
      && BALI && BALI.segments && BALI.segments.length
    );

    var traces = (figData.data || []).slice();
    var shapes = [];
    /* Preserve the figure's original annotations (these include subplot
       titles emitted by ``make_subplots``).  Boundary annotations from
       ``buildBoundaryUpdates`` get appended to this list. */
    var baseAnnots = (layoutSrc.annotations || []).slice();
    var annots = baseAnnots.slice();

    var axisIndices = isDual ? [1, 2] : [1];
    axisIndices.forEach(function (idx) {
      var bnd = buildBoundaryUpdates(bndArr, rng, ylim, idx);
      annots = annots.concat(bnd.annotations);
      if (showBali) {
        var mode = isDual
          ? (idx === 1 ? 'tps' : 'tpj')
          : undefined;
        var bali = buildBaliShapes(
          BALI.segments, rng, ylim, metric, BALI_PWR,
          {
            axisIndex:   idx,
            colorMode:   mode,
            /* Offset segment indices into the running layout.shapes array
               so the hover handler can find the right rectangle. */
            shapeOffset: shapes.length
          }
        );
        shapes = shapes.concat(bali.shapes);
        if (bali.hoverTraces && bali.hoverTraces.length) {
          traces = traces.concat(bali.hoverTraces);
        }
      } else {
        shapes = shapes.concat(bnd.shapes);
      }
    });

    /* Clone layout to avoid mutating the shared stored object */
    var layout = JSON.parse(JSON.stringify(layoutSrc));
    layout.shapes      = shapes;
    layout.annotations = annots;
    layout.autosize    = true;
    if (xRng) {
      layout.xaxis = layout.xaxis || {};
      layout.xaxis.range = xRng;
      if (isDual) {
        layout.xaxis2 = layout.xaxis2 || {};
        layout.xaxis2.range = xRng;
      }
    }

    renderPlotInPanel(plotDiv, traces, layout);
  }

  /** Re-renders every registered panel (used by show-idle toggle and range slider). */
  function refreshAll() {
    if (typeof updateBaliScales === 'function') updateBaliScales(CID);
    Object.keys(panelRegistry).forEach(function (pid) {
      var p = panelRegistry[pid];
      renderPlot(pid, p.metricSel.value, p.levelSel.value);
    });
  }

  /* ── panel-row management ─────────────────────────────────────────────── */

  /**
   * Creates a two-panel row, appends it to the panels container, wires events,
   * and triggers the initial render for each new panel.
   */
  function addPanelRow() {
    if (panelCount >= MAX) return;

    var wrap   = document.getElementById(CID + '-panels');
    var row    = document.createElement('div');
    row.className = 'jump-vis-panel-row';

    /* In BALI dual-subplot mode each panel already spans two charts side by
       side, so we only put one panel per row to keep each chart wide enough
       to read.  Otherwise the legacy 2-panels-per-row layout is used. */
    var dualLayout = (
      typeof BALI !== 'undefined'
      && BALI && Array.isArray(BALI.segments)
      && BALI.segments.length > 0
    );
    var panelsPerRow = dualLayout ? 1 : 2;

    var pids = [];
    for (var i = 0; i < panelsPerRow && panelCount < MAX; i++) {
      var pid    = CID + '-panel-' + panelCount;
      var metric = nextMetric();
      var defLev = (LEVS.indexOf('process') >= 0) ? 'process' : (LEVS[0] || 'process');
      row.appendChild(createPanelElement(pid, metric, defLev, OPTS, LEVS));
      pids.push(pid);
      panelCount++;
    }

    if (pids.length > 0) {
      wrap.appendChild(row);
      /* Attach events and render after the row is in the DOM */
      pids.forEach(function (pid) {
        attachPanelEvents(pid, renderPlot);
        panelRegistry[pid] = {
          metricSel: document.getElementById(pid + '-metric'),
          levelSel:  document.getElementById(pid + '-level')
        };
        renderPlot(pid, panelRegistry[pid].metricSel.value,
                        panelRegistry[pid].levelSel.value);
      });
    }

    if (panelCount >= MAX) {
      disableAddPanelButton(CID);
      var notice       = document.createElement('p');
      notice.className = 'jump-vis-max-notice';
      notice.textContent = 'All panels have been added.';
      wrap.appendChild(notice);
    }
  }

  /* ── bootstrap ────────────────────────────────────────────────────────── */

  function init() {
    initCellRangeSlider(CID, MIN_CELL, MAX_CELL, INIT_RNG, refreshAll);
    initShowIdle(CID, refreshAll);
    if (typeof initShowBali === 'function') initShowBali(CID, refreshAll);
    initAddPanelButton(CID, addPanelRow);
    if (typeof updateBaliScales === 'function') updateBaliScales(CID);
    addPanelRow();   /* render initial two panels */
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
