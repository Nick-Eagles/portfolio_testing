(function () {
  const COLORS = {
    stock: "#2c7fb8",
    bond: "#7fcdbb",
    t_bill: "#edf8b1",
    pointStart: "#4b1d6b",
    pointEnd: "#f4df2e",
  };
  const ASSETS = [
    ["stock", "Stocks", COLORS.stock],
    ["bond", "Bonds", COLORS.bond],
    ["t_bill", "T-Bills", COLORS.t_bill],
  ];

  function pct(value) {
    return `${(value * 100).toFixed(1)}%`;
  }

  function svgEl(name, attrs = {}) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function simplexCoords(point, geometry) {
    const { left, right, top } = geometry;
    return {
      x: point.bond * left.x + point.t_bill * right.x + point.stock * top.x,
      y: point.bond * left.y + point.t_bill * right.y + point.stock * top.y,
    };
  }

  function colorForIndex(index, total) {
    const ratio = total <= 1 ? 0 : index / (total - 1);
    const hue = 278 - 220 * ratio;
    const light = 30 + 26 * ratio;
    return `hsl(${hue}, 62%, ${light}%)`;
  }

  function nearestByKey(points, key) {
    return points.reduce((best, point) => {
      if (!best || Math.abs(point.key - key) < Math.abs(best.key - key)) {
        return point;
      }
      return best;
    }, null);
  }

  function buildWidget(root, data) {
    const points = data.points;
    let selected = points[0];
    const state = {};
    root.classList.add("results-widget");

    root.innerHTML = `
      <div class="results-tabs" role="tablist">
        <button class="results-tab" role="tab" aria-selected="true" data-tab="simplex" type="button">Simplex</button>
        <button class="results-tab" role="tab" aria-selected="false" data-tab="allocation" type="button">Allocation</button>
      </div>
      <div class="results-panel active" data-panel="simplex">
        <div class="results-chart-wrap"><svg class="simplex-svg" viewBox="0 0 760 660" role="img"></svg><div class="results-tooltip"></div></div>
      </div>
      <div class="results-panel" data-panel="allocation">
        <div class="results-chart-wrap"><svg class="allocation-svg" viewBox="0 0 820 430" role="img"></svg><div class="results-tooltip"></div></div>
        <div class="results-legend"></div>
      </div>
      <div class="results-readout"></div>
    `;

    root.querySelectorAll(".results-tab").forEach((button) => {
      button.addEventListener("click", () => {
        const tab = button.dataset.tab;
        root.querySelectorAll(".results-tab").forEach((item) => {
          item.setAttribute("aria-selected", String(item === button));
        });
        root.querySelectorAll(".results-panel").forEach((panel) => {
          panel.classList.toggle("active", panel.dataset.panel === tab);
        });
      });
    });

    function setSelected(point) {
      selected = point;
      updateReadouts(root, data, point);
      updateSimplexState(state, point);
      updateAllocationState(state, point);
    }

    drawSimplex(root, data, state, setSelected);
    drawAllocation(root, data, state, setSelected);
    setSelected(points[0]);
  }

  function updateReadouts(root, data, point) {
    const label = `${data.label} ${point.key}${data.unit ? ` (${data.unit})` : ""}`;
    const compact = root.querySelector(".results-readout");
    compact.innerHTML = `
      <div class="selected-key">${label}</div>
      ${ASSETS.map(([key, name]) => `<div class="results-weight"><strong>${pct(point[key])}</strong><span>${name}</span></div>`).join("")}
    `;
  }

  function drawSimplex(root, data, state, setSelected) {
    const svg = root.querySelector(".simplex-svg");
    const tooltip = root.querySelector('[data-panel="simplex"] .results-tooltip');
    const geometry = {
      left: { x: 90, y: 575 },
      right: { x: 670, y: 575 },
      top: { x: 380, y: 85 },
    };

    svg.appendChild(svgEl("path", {
      d: `M ${geometry.left.x} ${geometry.left.y} L ${geometry.top.x} ${geometry.top.y} L ${geometry.right.x} ${geometry.right.y} Z`,
      class: "results-simplex-outline",
    }));
    [
      [geometry.top, "100% Stocks", 380, 45, "middle"],
      [geometry.left, "100% Bonds", 90, 625, "middle"],
      [geometry.right, "100% T-Bills", 670, 625, "middle"],
    ].forEach(([, label, x, y, anchor]) => {
      const text = svgEl("text", { x, y, "text-anchor": anchor, "font-size": 22, fill: "currentColor" });
      text.textContent = label;
      svg.appendChild(text);
    });

    const coords = data.points.map((point) => ({ point, ...simplexCoords(point, geometry) }));
    const path = coords.map((item, index) => `${index === 0 ? "M" : "L"} ${item.x} ${item.y}`).join(" ");
    svg.appendChild(svgEl("path", { d: path, class: "results-path" }));

    state.simplexPoints = new Map();
    coords.forEach((item, index) => {
      const circle = svgEl("circle", {
        cx: item.x,
        cy: item.y,
        r: 6,
        fill: colorForIndex(index, coords.length),
        class: "results-point",
      });
      circle.addEventListener("mouseenter", (event) => {
        setSelected(item.point);
        showTooltip(tooltip, event, data, item.point);
      });
      circle.addEventListener("mousemove", (event) => showTooltip(tooltip, event, data, item.point));
      circle.addEventListener("mouseleave", () => tooltip.classList.remove("visible"));
      svg.appendChild(circle);
      state.simplexPoints.set(item.point.key, circle);
    });
  }

  function updateSimplexState(state, point) {
    if (!state.simplexPoints) return;
    state.simplexPoints.forEach((node, key) => node.classList.toggle("active", key === point.key));
  }

  function drawAllocation(root, data, state, setSelected) {
    const svg = root.querySelector(".allocation-svg");
    const tooltip = root.querySelector('[data-panel="allocation"] .results-tooltip');
    const width = 820;
    const height = 430;
    const margin = { left: 56, right: 24, top: 24, bottom: 54 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const minKey = data.points[0].key;
    const maxKey = data.points[data.points.length - 1].key;
    const x = (key) => margin.left + ((key - minKey) / (maxKey - minKey)) * plotWidth;
    const y = (value) => margin.top + (1 - value) * plotHeight;

    for (let value = 0; value <= 1.001; value += 0.25) {
      const line = svgEl("line", {
        x1: margin.left,
        x2: width - margin.right,
        y1: y(value),
        y2: y(value),
        class: "results-grid",
      });
      svg.appendChild(line);
      const text = svgEl("text", {
        x: margin.left - 10,
        y: y(value) + 5,
        "text-anchor": "end",
        "font-size": 13,
        fill: "currentColor",
      });
      text.textContent = `${Math.round(value * 100)}%`;
      svg.appendChild(text);
    }

    svg.appendChild(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: y(0), y2: y(0), class: "results-axis" }));
    svg.appendChild(svgEl("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: y(0), class: "results-axis" }));

    const label = svgEl("text", {
      x: margin.left + plotWidth / 2,
      y: height - 12,
      "text-anchor": "middle",
      "font-size": 16,
      fill: "currentColor",
    });
    label.textContent = data.label;
    svg.appendChild(label);

    const series = [
      ["t_bill", 0, (p) => p.t_bill],
      ["bond", 1, (p) => p.t_bill + p.bond],
      ["stock", 2, () => 1],
    ];
    let lower = data.points.map(() => 0);
    series.forEach(([key, , upperFn]) => {
      const upper = data.points.map(upperFn);
      const top = data.points.map((point, index) => `${index === 0 ? "M" : "L"} ${x(point.key)} ${y(upper[index])}`).join(" ");
      const bottom = data.points.slice().reverse().map((point, reverseIndex) => {
        const index = data.points.length - 1 - reverseIndex;
        return `L ${x(point.key)} ${y(lower[index])}`;
      }).join(" ");
      svg.appendChild(svgEl("path", {
        d: `${top} ${bottom} Z`,
        fill: COLORS[key],
        opacity: 0.9,
      }));
      lower = upper;
    });

    const hoverLine = svgEl("line", {
      y1: margin.top,
      y2: margin.top + plotHeight,
      class: "results-hover-line",
    });
    svg.appendChild(hoverLine);
    state.hoverLine = hoverLine;
    state.xForKey = x;

    const hit = svgEl("rect", {
      x: margin.left,
      y: margin.top,
      width: plotWidth,
      height: plotHeight,
      fill: "transparent",
      cursor: "crosshair",
    });
    hit.addEventListener("mousemove", (event) => {
      const rect = svg.getBoundingClientRect();
      const mouseX = ((event.clientX - rect.left) / rect.width) * width;
      const key = minKey + ((mouseX - margin.left) / plotWidth) * (maxKey - minKey);
      const point = nearestByKey(data.points, key);
      setSelected(point);
      showTooltip(tooltip, event, data, point);
    });
    hit.addEventListener("mouseleave", () => tooltip.classList.remove("visible"));
    svg.appendChild(hit);

    const legend = root.querySelector(".results-legend");
    legend.innerHTML = ASSETS.map(([, name, color]) => `
      <span class="results-legend-item"><span class="results-swatch" style="background: ${color}"></span>${name}</span>
    `).join("");
  }

  function updateAllocationState(state, point) {
    if (!state.hoverLine || !state.xForKey) return;
    const x = state.xForKey(point.key);
    state.hoverLine.setAttribute("x1", x);
    state.hoverLine.setAttribute("x2", x);
  }

  function showTooltip(tooltip, event, data, point) {
    const panelRect = tooltip.parentElement.getBoundingClientRect();
    tooltip.style.left = `${event.clientX - panelRect.left}px`;
    tooltip.style.top = `${event.clientY - panelRect.top}px`;
    tooltip.innerHTML = `<strong>${data.label} ${point.key}</strong><br>Stocks ${pct(point.stock)} · Bonds ${pct(point.bond)} · T-Bills ${pct(point.t_bill)}`;
    tooltip.classList.add("visible");
  }

  async function init() {
    const roots = document.querySelectorAll("[data-results-widget]");
    await Promise.all(Array.from(roots).map(async (root) => {
      const response = await fetch(root.dataset.source);
      const data = await response.json();
      buildWidget(root, data);
    }));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
