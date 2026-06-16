/**
 * panel.js
 * Panel creation, event wiring, boundary helpers, and Plotly rendering.
 */

/* ── Panel DOM factory ───────────────────────────────────────────────────── */

/**
 * Builds and returns a panel <div> with metric and level dropdowns.
 * The element is NOT yet attached to the document; call appendChild first.
 *
 * @param {string}   pid    - unique panel ID (e.g. "jump-vis-abc123-panel-0")
 * @param {string}   metric - initially selected metric value
 * @param {string}   level  - initially selected level value
 * @param {Array}    opts   - [[label, value], …] metric options
 * @param {string[]} levs   - available level names
 * @returns {HTMLElement}
 */
function createPanelElement(pid, metric, level, opts, levs) {
  var metricOpts = opts.map(function (o) {
    return '<option value="' + o[1] + '"' + (o[1] === metric ? ' selected' : '') + '>'
           + o[0] + '</option>';
  }).join('');

  var levelOpts = levs.map(function (l) {
    return '<option value="' + l + '"' + (l === level ? ' selected' : '') + '>'
           + l + '</option>';
  }).join('');

  var div       = document.createElement('div');
  div.className    = 'jump-vis-panel';
  div.dataset.pid  = pid;
  div.innerHTML    =
    '<div class="jump-vis-ctrl-row">'
    + '<label>Metric: <select class="jump-vis-metric-sel" id="' + pid + '-metric">'
    + metricOpts + '</select></label>'
    + '<label>Level: <select class="jump-vis-level-sel" id="' + pid + '-level">'
    + levelOpts + '</select></label>'
    + '</div>'
    + '<div class="jump-vis-plot-area" id="' + pid + '-plot"></div>';

  return div;
}

/**
 * Attaches change listeners to a panel's metric and level dropdowns.
 * onUpdate(pid, metric, level) is called on every dropdown change.
 * Does NOT trigger an initial render — the caller is responsible for that.
 *
 * @param {string}   pid      - panel ID
 * @param {Function} onUpdate - (pid, metric, level) → void
 */
function attachPanelEvents(pid, onUpdate) {
  var mSel = document.getElementById(pid + '-metric');
  var lSel = document.getElementById(pid + '-level');
  if (!mSel || !lSel) return;
  mSel.addEventListener('change', function () { onUpdate(pid, mSel.value, lSel.value); });
  lSel.addEventListener('change', function () { onUpdate(pid, mSel.value, lSel.value); });
}

/* ── Boundary helpers ────────────────────────────────────────────────────── */

/**
 * Builds Plotly shapes (rectangles) and annotations (cell labels) for all
 * boundaries whose cell_index falls within [cellRange[0], cellRange[1]].
 *
 * @param {Object[]} bndData   - [{cell_index, x0, x1, color}, …]
 * @param {number[]} cellRange - [lo, hi]
 * @param {number[]} ylim      - [ymin, ymax] of the target figure axis
 * @returns {{ shapes: Object[], annotations: Object[] }}
 */
function buildBoundaryUpdates(bndData, cellRange, ylim, axisIndex) {
  var shapes = [], annots = [];
  var lo = cellRange[0], hi = cellRange[1];
  var ymin = ylim[0], ymax = ylim[1];
  var height = (ymax - ymin) || 1.0;
  var suffix = (axisIndex && axisIndex > 1) ? String(axisIndex) : '';
  var xref = 'x' + suffix;
  var yref = 'y' + suffix;

  for (var i = 0; i < bndData.length; i++) {
    var b = bndData[i];
    if (b.cell_index < lo || b.cell_index > hi) continue;

    shapes.push({
      type: 'rect',
      x0: b.x0, x1: b.x1, y0: ymin, y1: ymax,
      xref: xref, yref: yref,
      fillcolor: b.color, opacity: 0.4,
      line: { color: 'black', dash: 'dash', width: 1 },
      layer: 'below'
    });
    annots.push({
      x: (b.x0 + b.x1) / 2,
      y: ymax - height * 0.1,
      xref: xref, yref: yref,
      text: '#' + b.cell_index,
      showarrow: false,
      font: { size: 10 },
      bgcolor: 'rgba(255,255,255,0.8)'
    });
  }
  return { shapes: shapes, annotations: annots };
}

/**
 * Computes a comfortable x-axis range [xMin-pad, xMax+pad] that frames all
 * boundaries within the selected cell range.  Returns null when none match.
 *
 * @param {Object[]} bndData   - [{cell_index, x0, x1}, …]
 * @param {number[]} cellRange - [lo, hi]
 * @returns {number[]|null}
 */
function xRangeForCells(bndData, cellRange) {
  var lo = cellRange[0], hi = cellRange[1];
  var xMin = Infinity, xMax = -Infinity;

  for (var i = 0; i < bndData.length; i++) {
    var b = bndData[i];
    if (b.cell_index < lo || b.cell_index > hi) continue;
    if (b.x0 < xMin) xMin = b.x0;
    if (b.x1 > xMax) xMax = b.x1;
  }
  if (!isFinite(xMin)) return null;
  var pad = (xMax - xMin) * 0.05 || 0.5;
  return [xMin - pad, xMax + pad];
}

/* ── BALI overlay helpers ────────────────────────────────────────────────── */

/**
 * Builds Plotly rectangle shapes for BALI segments whose owning cell falls
 * within [cellRange[0], cellRange[1]].  Picks the colormap based on
 * ``metric``: GPU power metrics use the energy-efficiency colormap, all
 * other metrics use the tokens-per-second colormap.
 *
 * @param {Object[]} segments  - [{cell_index, x0, x1, color_tokens,
 *                                  color_energy, is_error, info}, …]
 * @param {number[]} cellRange - [lo, hi]
 * @param {number[]} ylim      - [ymin, ymax]
 * @param {string}   metric    - currently selected metric key
 * @param {string[]} powerMetrics - metric keys that use the energy colormap
 * @returns {{ shapes: Object[], hoverTrace: Object|null }}
 */
function buildBaliShapes(segments, cellRange, ylim, metric, powerMetrics, opts) {
  var shapes = [];
  var hoverTraces = [];
  if (!segments || !segments.length) {
    return { shapes: shapes, hoverTraces: hoverTraces };
  }
  opts = opts || {};
  var axisIndex = opts.axisIndex || 1;
  var suffix = axisIndex > 1 ? String(axisIndex) : '';
  var xref = 'x' + suffix;
  var yref = 'y' + suffix;
  // ``shapeOffset`` is the index of the first BALI shape this call adds
  // inside the overall layout.shapes array.  The renderer uses it so the
  // hover handler can map a hovered scatter point back to the originating
  // rectangle in layout.shapes and apply a glow effect.
  var shapeOffset = opts.shapeOffset || 0;

  var lo = cellRange[0], hi = cellRange[1];
  var ymin = ylim[0], ymax = ylim[1];

  // ``colorMode`` overrides the metric-based default: 'tps' forces the
  // tokens-per-second colormap, 'tpj' forces tokens-per-joule.  Without an
  // explicit mode we keep the legacy rule (GPU power metrics use energy).
  var mode = opts.colorMode;
  var isPower;
  if (mode === 'tpj') isPower = true;
  else if (mode === 'tps') isPower = false;
  else isPower = (powerMetrics || []).indexOf(metric) >= 0;

  for (var i = 0; i < segments.length; i++) {
    var s = segments[i];
    if (s.cell_index < lo || s.cell_index > hi) continue;

    var color = isPower ? s.color_energy : s.color_tokens;
    var isError = !!s.is_error || !color;
    var rectShape;
    if (isError) {
      rectShape = {
        type: 'rect',
        x0: s.x0, x1: s.x1, y0: ymin, y1: ymax,
        xref: xref, yref: yref,
        fillcolor: 'rgba(0,0,0,0)',
        line: { color: 'gray', width: 1, dash: 'dot' },
        layer: 'below'
      };
    } else {
      rectShape = {
        type: 'rect',
        x0: s.x0, x1: s.x1, y0: ymin, y1: ymax,
        xref: xref, yref: yref,
        fillcolor: color, opacity: 0.55,
        line: { color: 'gray', width: 1 },
        layer: 'below'
      };
    }
    var shapeIndex = shapeOffset + shapes.length;
    /* Original style is stashed on the shape itself so the hover handler
       can restore it after a glow-up. */
    rectShape._bali_base = {
      fillcolor: rectShape.fillcolor,
      opacity:   rectShape.opacity != null ? rectShape.opacity : 1,
      line:      JSON.parse(JSON.stringify(rectShape.line || {}))
    };
    shapes.push(rectShape);

    /* Hover text: assemble key/value lines from s.info */
    var lines = ['<b>BALI segment</b>'];
    var info  = s.info || {};
    Object.keys(info).forEach(function (k) {
      var v = info[k];
      if (v === null || v === undefined || v === '') return;
      lines.push(k + ': ' + v);
    });
    var hoverText = lines.join('<br>');

    /* Region hover: spread a grid of invisible markers across the
       rectangle so plotly fires the hover anywhere inside, not just at a
       single midpoint.  ``customdata`` carries the shape index so the
       glow handler can light up the matching rectangle. */
    var cols = 12, rows = 5;
    var hx = [], hy = [], hcd = [];
    for (var cx = 0; cx < cols; cx++) {
      for (var cy = 0; cy < rows; cy++) {
        var fx = (cx + 0.5) / cols;
        var fy = (cy + 0.5) / rows;
        hx.push(s.x0 + (s.x1 - s.x0) * fx);
        hy.push(ymin + (ymax - ymin) * fy);
        hcd.push(shapeIndex);
      }
    }
    hoverTraces.push({
      type: 'scatter',
      mode: 'markers',
      x: hx,
      y: hy,
      customdata: hcd,
      marker: { size: 24, color: 'rgba(0,0,0,0)', opacity: 0 },
      hoverinfo: 'text',
      hovertemplate: hoverText + '<extra></extra>',
      hoverlabel: {
        bgcolor: 'rgba(255,255,255,0.95)',
        bordercolor: isError ? 'gray' : color,
        font: { color: '#111' }
      },
      showlegend: false,
      name: '',
      xaxis: xref,
      yaxis: yref
    });
  }

  return { shapes: shapes, hoverTraces: hoverTraces };
}

/* ── Plotly rendering ────────────────────────────────────────────────────── */

/**
 * Renders a Plotly figure into a panel's plot-area div.
 * Polls for Plotly.js availability (CDN may still be loading) with a 10 s
 * timeout before showing an error message.
 *
 * @param {HTMLElement}  plotDiv - the .jump-vis-plot-area element
 * @param {Object[]|null} traces - Plotly data array; null shows a no-data message
 * @param {Object}        layout - Plotly layout (already augmented with shapes etc.)
 */
function renderPlotInPanel(plotDiv, traces, layout) {
  if (!plotDiv) return;

  if (!traces) {
    plotDiv.innerHTML =
      '<p class="jump-vis-no-data">No data for selected metric and level.</p>';
    return;
  }

  /* Stash per-shape baseline styles on the plot div so the glow handler
     can restore them after a hover.  Read straight from the shape dicts
     before plotly clones them internally. */
  var baliBases = {};
  if (layout && Array.isArray(layout.shapes)) {
    for (var si = 0; si < layout.shapes.length; si++) {
      var sh = layout.shapes[si];
      if (sh && sh._bali_base) {
        baliBases[si] = sh._bali_base;
        delete sh._bali_base;  // keep plotly's input clean
      }
    }
  }

  function attachBaliGlow(div) {
    /* Highlight the BALI shape under the cursor with a stronger border
       and higher opacity for a subtle "glow up"; restore on unhover. */
    if (!div || !div.on) return;
    var lastIdx = -1;

    function setHighlight(idx) {
      var updates = {};
      Object.keys(baliBases).forEach(function (key) {
        var i    = parseInt(key, 10);
        var base = baliBases[key];
        if (i === idx) {
          updates['shapes[' + i + '].opacity']    = 0.9;
          updates['shapes[' + i + '].line.color'] = '#111';
          updates['shapes[' + i + '].line.width'] = 2;
        } else {
          updates['shapes[' + i + '].opacity']    =
            base.opacity != null ? base.opacity : 1;
          updates['shapes[' + i + '].line.color'] =
            (base.line && base.line.color) || 'gray';
          updates['shapes[' + i + '].line.width'] =
            (base.line && base.line.width) || 1;
        }
      });
      if (Object.keys(updates).length) Plotly.relayout(div, updates);
    }

    div.on('plotly_hover', function (evt) {
      if (!evt || !evt.points || !evt.points.length) return;
      var p   = evt.points[0];
      var cd  = p.customdata;
      var idx = Array.isArray(cd) ? cd[0] : cd;
      if (typeof idx !== 'number' || idx === lastIdx) return;
      lastIdx = idx;
      setHighlight(idx);
    });
    div.on('plotly_unhover', function () {
      if (lastIdx < 0) return;
      lastIdx = -1;
      setHighlight(-1);
    });
  }

  function doPlot() {
    Plotly.newPlot(plotDiv, traces, layout, { responsive: true, displayModeBar: true })
      .then(function () { attachBaliGlow(plotDiv); });
  }

  if (typeof window.Plotly !== 'undefined') {
    doPlot();
  } else {
    var tries = 0;
    var timer = setInterval(function () {
      tries++;
      if (typeof window.Plotly !== 'undefined') {
        clearInterval(timer);
        doPlot();
      } else if (tries > 100) {   /* 10 s */
        clearInterval(timer);
        plotDiv.innerHTML =
          '<p class="jump-vis-no-data">Plotly.js could not be loaded. '
          + 'Check your network or try re-running the cell.</p>';
      }
    }, 100);
  }
}
