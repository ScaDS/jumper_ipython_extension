/**
 * cell_range_slider.js
 * Dual-range slider for selecting a contiguous cell index range.
 */

/**
 * Returns the current [lo, hi] cell-index range from the two slider thumbs.
 * Guarantees lo ≤ hi by swapping if necessary.
 *
 * @param {string} cid     - container ID prefix
 * @param {number} minCell - minimum allowed cell index
 * @param {number} maxCell - maximum allowed cell index
 * @returns {[number, number]}
 */
function getCellRange(cid, minCell, maxCell) {
  var rMin = document.getElementById(cid + '-range-min');
  var rMax = document.getElementById(cid + '-range-max');
  var lo   = rMin ? parseInt(rMin.value, 10) : minCell;
  var hi   = rMax ? parseInt(rMax.value, 10) : maxCell;
  if (lo > hi) { var t = lo; lo = hi; hi = t; }
  return [lo, hi];
}

/**
 * Syncs the filled-range track and the label text to the current thumb values.
 * Also prevents lo from exceeding hi by snapping the active thumb.
 *
 * @param {string} cid     - container ID prefix
 * @param {number} minCell
 * @param {number} maxCell
 */
function updateSliderUI(cid, minCell, maxCell) {
  var rMin  = document.getElementById(cid + '-range-min');
  var rMax  = document.getElementById(cid + '-range-max');
  var fill  = document.getElementById(cid + '-range-fill');
  var label = document.getElementById(cid + '-range-val');
  if (!rMin || !rMax) return;

  var lo = parseInt(rMin.value, 10);
  var hi = parseInt(rMax.value, 10);

  /* Prevent thumbs from crossing */
  if (lo > hi) {
    if (document.activeElement === rMin) { rMin.value = hi; lo = hi; }
    else                                  { rMax.value = lo; hi = lo; }
  }

  var total    = (maxCell - minCell) || 1;
  var leftPct  = ((lo - minCell) / total) * 100;
  var rightPct = ((hi - minCell) / total) * 100;

  if (fill)  { fill.style.left = leftPct + '%'; fill.style.width = (rightPct - leftPct) + '%'; }
  if (label) { label.textContent = lo + '–' + hi; }
}

/**
 * Initialises the dual-range slider:
 *   • sets both thumbs to initRange values,
 *   • draws the initial fill track,
 *   • wires input events → updateSliderUI + onRangeChange callback.
 *
 * @param {string}   cid           - container ID prefix
 * @param {number}   minCell
 * @param {number}   maxCell
 * @param {number[]} initRange     - [lo, hi] initial positions
 * @param {Function} onRangeChange - called (no arguments) after each thumb move
 */
function initCellRangeSlider(cid, minCell, maxCell, initRange, onRangeChange) {
  var rMin = document.getElementById(cid + '-range-min');
  var rMax = document.getElementById(cid + '-range-max');

  if (rMin && initRange && initRange.length >= 1) rMin.value = initRange[0];
  if (rMax && initRange && initRange.length >= 2) rMax.value = initRange[1];

  updateSliderUI(cid, minCell, maxCell);

  if (rMin) {
    rMin.addEventListener('input', function () {
      updateSliderUI(cid, minCell, maxCell);
      onRangeChange();
    });
  }
  if (rMax) {
    rMax.addEventListener('input', function () {
      updateSliderUI(cid, minCell, maxCell);
      onRangeChange();
    });
  }
}
