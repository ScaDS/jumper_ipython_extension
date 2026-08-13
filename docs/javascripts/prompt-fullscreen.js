(() => {
  const selector = "[data-prompt-fullscreen]";
  const fallbackClass = "prompt-fullscreen-target--fallback";

  function setButtonState(target, expanded) {
    const button = target.querySelector(".prompt-fullscreen-toggle");
    if (!button) return;
    button.setAttribute("aria-pressed", String(expanded));
    button.setAttribute(
      "aria-label",
      expanded ? "Exit fullscreen" : "Open diagram in fullscreen",
    );
  }

  function leaveFallback(target) {
    target.classList.remove(fallbackClass);
    document.documentElement.classList.remove("prompt-fullscreen-lock");
    setButtonState(target, false);
  }

  document.addEventListener("click", async (event) => {
    const button = event.target.closest(".prompt-fullscreen-toggle");
    if (!button) return;
    const target = button.closest(selector);
    if (!target) return;

    if (document.fullscreenElement === target) {
      await document.exitFullscreen();
      return;
    }
    if (target.classList.contains(fallbackClass)) {
      leaveFallback(target);
      return;
    }

    try {
      await target.requestFullscreen();
    } catch (_error) {
      target.classList.add(fallbackClass);
      document.documentElement.classList.add("prompt-fullscreen-lock");
      setButtonState(target, true);
    }
  });

  document.addEventListener("fullscreenchange", () => {
    document.querySelectorAll(selector).forEach((target) => {
      setButtonState(target, document.fullscreenElement === target);
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (document.fullscreenElement?.matches(selector)) {
      document.exitFullscreen();
      return;
    }
    const target = document.querySelector(`${selector}.${fallbackClass}`);
    if (target) leaveFallback(target);
  });
})();
