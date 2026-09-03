const methodContent = {
  "cycle-one": {
    tab: "cycle-one-tab",
    image: "assets/cycle-one-web-v2.png",
    alt: "Cycle one pipeline from video frames and optical flow to pseudo-instance labels and pairwise representation learning.",
    kicker: "Cycle 1 · high precision",
    title: "Turn motion boundaries into object supervision",
    description:
      "Optical flow and pixel clustering produce pseudo-instance labels. The labels organize dense features around object unity and instance separation through pairwise metric learning.",
  },
  "cycle-two": {
    tab: "cycle-two-tab",
    image: "assets/cycle-two-web-v3.png",
    alt: "Cycle two pipeline using model proposals and independent motion evidence to refine and verify expanded pseudo-instance labels.",
    kicker: "Cycle 2 · expanded coverage",
    title: "Recover the motion evidence Cycle 1 leaves behind",
    description:
      "A frozen encoder proposes complete masks. Proposal-independent optical flow then refines and verifies them before training, increasing coverage without accepting model proposals at face value.",
  },
};

const featureNames = {
  dinov3: "DINOv3",
  clip: "CLIP",
  mae: "MAE",
  ours: "Ours",
};

function renderMethod(methodKey) {
  const content = methodContent[methodKey];
  if (!content) return;

  document.querySelectorAll("[data-method]").forEach((button) => {
    const selected = button.dataset.method === methodKey;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    button.closest("li")?.classList.toggle("is-active", selected);
  });

  const panel = document.querySelector("#method-panel");
  const image = document.querySelector("#method-image");
  image.src = content.image;
  image.alt = content.alt;
  document.querySelector("#method-kicker").textContent = content.kicker;
  document.querySelector("#method-title").textContent = content.title;
  document.querySelector("#method-description").textContent = content.description;
  panel.setAttribute("aria-labelledby", content.tab);
}

function renderFeature(featureKey) {
  const name = featureNames[featureKey];
  if (!name) return;

  document.querySelectorAll("[data-feature]").forEach((button) => {
    const selected = button.dataset.feature === featureKey;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    button.closest("li")?.classList.toggle("is-active", selected);
  });

  document.querySelectorAll(".feature-name").forEach((label) => {
    label.textContent = name;
  });

  document.querySelectorAll(".feature-output").forEach((image) => {
    const scene = image.dataset.scene;
    image.src = `assets/teaser-${scene}-${featureKey}.jpg`;
    image.alt = `${name} PCA feature map for the ${scene} scene.`;
  });
}

document.querySelectorAll("[data-method]").forEach((button) => {
  button.addEventListener("click", () => renderMethod(button.dataset.method));
});

document.querySelectorAll("[data-feature]").forEach((button) => {
  button.addEventListener("click", () => renderFeature(button.dataset.feature));
});

const header = document.querySelector("[data-header]");
if (header) {
  const updateHeader = () => header.classList.toggle("scrolled", window.scrollY > 10);
  window.addEventListener("scroll", updateHeader, { passive: true });
  updateHeader();
}

const navToggle = document.querySelector(".nav-toggle, .navbar-burger");
const navLinks = document.querySelector(".nav-links, .navbar-menu");
if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    const open = navToggle.getAttribute("aria-expanded") === "true";
    navToggle.setAttribute("aria-expanded", String(!open));
    navToggle.classList.toggle("is-active", !open);
    navLinks.classList.toggle("is-active", !open);
    navLinks.classList.toggle("open", !open);
  });
}

navLinks?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    navLinks.classList.remove("open");
    navLinks.classList.remove("is-active");
    navToggle.setAttribute("aria-expanded", "false");
    navToggle.classList.remove("is-active");
  });
});

const resultChartData = {
  depth: {
    labels: ["IN-22K", "SimMIM", "DINOv2", "DINOv3", "Ours"],
    values: [0.977, 0.979, 0.98, 0.986, 0.988],
    min: 0.974,
    max: 0.99,
    digits: 3,
  },
  detection: {
    labels: ["IN-22K", "SimMIM", "Ours"],
    values: [54.59, 54.98, 55.89],
    min: 54,
    max: 56.2,
    digits: 2,
  },
  occupancy: {
    labels: ["IN-22K", "SimMIM", "DINOv2", "DINOv3", "Ours"],
    values: [37.6, 38.6, 39, 41.02, 40.04],
    min: 37,
    max: 42,
    digits: 2,
  },
  planning: {
    labels: ["DA-ViT-L", "DINOv2-L", "DINOv3-L", "Ours"],
    values: [90.5, 87.2, 89, 88.9],
    min: 86,
    max: 91,
    digits: 1,
  },
};

function initResultCharts() {
  if (typeof Chart === "undefined") return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.querySelectorAll("[data-result-chart]").forEach((canvas) => {
    const config = resultChartData[canvas.dataset.resultChart];
    if (!config) return;

    const colors = config.labels.map((label) => (label === "Ours" ? "#3273dc" : "#8b9bb0"));
    const borders = config.labels.map((label) => (label === "Ours" ? "#2458ad" : "#aebbc9"));

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: config.labels,
        datasets: [
          {
            data: config.values,
            backgroundColor: colors,
            borderColor: borders,
            borderWidth: 0,
            borderRadius: 3,
            borderSkipped: false,
            barPercentage: 0.72,
            categoryPercentage: 0.8,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: reducedMotion ? false : { duration: 550, easing: "easeOutQuart" },
        plugins: {
          legend: { display: false },
          tooltip: {
            displayColors: false,
            callbacks: {
              label: (context) => `${context.label}: ${Number(context.raw).toFixed(config.digits)}`,
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            border: { color: "#c4ccd6" },
            ticks: {
              color: (context) => context.tick.label === "Ours" ? "#2458ad" : "#64748b",
              font: (context) => ({
                size: 11,
                weight: context.tick.label === "Ours" ? "700" : "500",
                family: "ui-monospace, SFMono-Regular, Menlo, monospace",
              }),
              maxRotation: 0,
              autoSkip: false,
            },
          },
          y: {
            min: config.min,
            max: config.max,
            border: { display: false },
            grid: { color: "rgba(100, 116, 139, 0.18)" },
            ticks: {
              color: "#7b8794",
              maxTicksLimit: 5,
              padding: 6,
              callback: (value) => Number(value).toFixed(config.digits),
            },
          },
        },
      },
    });
  });
}

initResultCharts();

const copyButton = document.querySelector("[data-copy-citation]");
copyButton.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex").innerText;
  try {
    await navigator.clipboard.writeText(citation);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#bibtex"));
    selection.removeAllRanges();
    selection.addRange(range);
    document.execCommand("copy");
    selection.removeAllRanges();
  }

  const label = copyButton.querySelector("span");
  const icon = copyButton.querySelector("img");
  label.textContent = "Copied";
  icon.src = "assets/icons/check.svg";
  window.setTimeout(() => {
    label.textContent = "Copy BibTeX";
    icon.src = "assets/icons/copy.svg";
  }, 1800);
});
