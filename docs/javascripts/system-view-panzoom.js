(() => {
  function clearStoredSystemViewTransform() {
    const heading = document.getElementById("system-view");
    if (!heading) return;

    let box = heading.nextElementSibling;
    while (box && !box.classList.contains("panzoom-box")) {
      box = box.nextElementSibling;
    }
    if (!box) return;

    const boxes = [...document.querySelectorAll(".panzoom-box")];
    const boxIndex = boxes.indexOf(box);
    if (boxIndex < 0) return;

    try {
      localStorage.removeItem(`panzoom-${window.location.pathname}-${boxIndex}`);
    } catch (_error) {
      // Storage may be unavailable; panzoom will then use its default scale.
    }
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(clearStoredSystemViewTransform);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", clearStoredSystemViewTransform);
  } else {
    clearStoredSystemViewTransform();
  }
})();
