(function () {
  var KEY = "open-rag-theme";
  var root = document.documentElement;
  var stored = null;

  try {
    stored = window.localStorage.getItem(KEY);
  } catch (e) {
    stored = null;
  }

  root.setAttribute("data-theme", stored === "light" ? "light" : "dark");

  function label(button) {
    var light = root.getAttribute("data-theme") === "light";
    button.textContent = light ? "Dark" : "Light";
    button.setAttribute("aria-label", light ? "Switch to dark theme" : "Switch to light theme");
  }

  function wire() {
    var button = document.getElementById("theme-toggle");
    if (!button) return;
    label(button);
    button.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      try {
        window.localStorage.setItem(KEY, next);
      } catch (e) {
      }
      label(button);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})();
