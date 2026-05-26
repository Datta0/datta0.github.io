(() => {
  function initPpoClipWidget() {
    const epsilonSlider = document.getElementById("ppo-epsilon");
    const epsilonValue = document.getElementById("ppo-epsilon-value");
    const canvas = document.getElementById("ppo-clip-canvas");
    if (!epsilonSlider || !epsilonValue || !canvas) return;

    const ctx = canvas.getContext("2d");
    const cssWidth = 920;
    const cssHeight = 430;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    canvas.style.maxWidth = `${cssWidth}px`;
    ctx.scale(dpr, dpr);

    const rhoMin = 0.4;
    const rhoMax = 1.6;

    function drawPlot(epsilon) {
      const x = 82;
      const y = 92;
      const w = 760;
      const h = 230;
      const yMin = -1.65;
      const yMax = 1.65;
      const xScale = (rho) => x + ((rho - rhoMin) / (rhoMax - rhoMin)) * w;
      const yScale = (v) => y + h - ((v - yMin) / (yMax - yMin)) * h;
      const clip = (rho) => Math.min(Math.max(rho, 1 - epsilon), 1 + epsilon);
      const unclipped = (rho, advantage) => rho * advantage;
      const clippedTerm = (rho, advantage) => clip(rho) * advantage;
      const clippedSurrogate = (rho, advantage) =>
        Math.min(unclipped(rho, advantage), clippedTerm(rho, advantage));

      ctx.fillStyle = "#252629";
      ctx.strokeStyle = "#3f4246";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(x - 44, y - 56, w + 88, h + 108, 8);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "rgba(251, 188, 4, 0.12)";
      ctx.fillRect(xScale(1 - epsilon), y, xScale(1 + epsilon) - xScale(1 - epsilon), h);

      ctx.strokeStyle = "#55585f";
      ctx.lineWidth = 1;
      [1 - epsilon, 1, 1 + epsilon].forEach((rho, i) => {
        ctx.setLineDash(i === 1 ? [2, 5] : [6, 5]);
        ctx.beginPath();
        ctx.moveTo(xScale(rho), y);
        ctx.lineTo(xScale(rho), y + h);
        ctx.stroke();
      });
      ctx.setLineDash([]);

      ctx.strokeStyle = "#aeb0b4";
      ctx.beginPath();
      ctx.moveTo(x, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.moveTo(x, y);
      ctx.lineTo(x, y + h);
      ctx.moveTo(x, yScale(0));
      ctx.lineTo(x + w, yScale(0));
      ctx.stroke();

      function line(fn, color, dashed = false) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.setLineDash(dashed ? [8, 6] : []);
        ctx.beginPath();
        for (let i = 0; i <= 180; i++) {
          const rho = rhoMin + (i / 180) * (rhoMax - rhoMin);
          const px = xScale(rho);
          const py = yScale(fn(rho));
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      line((rho) => unclipped(rho, 1), "#8ab4f8", true);
      line((rho) => clippedSurrogate(rho, 1), "#8ab4f8");
      line((rho) => unclipped(rho, -1), "#fbbc04", true);
      line((rho) => clippedSurrogate(rho, -1), "#fbbc04");

      ctx.fillStyle = "#f0f1f3";
      ctx.font = "700 18px system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText("PPO clipped surrogate", x, y - 28);
      ctx.fillStyle = "#c8cbd1";
      ctx.font = "13px system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText("solid = min expression          dashed = rho * A", x, y - 8);
      ctx.fillStyle = "#e4c66a";
      ctx.fillText("trust region", xScale(1) - 34, y + 18);
      ctx.fillStyle = "#c8cbd1";
      ctx.fillText("rho", x + w / 2 - 10, y + h + 36);
      [1 - epsilon, 1, 1 + epsilon].forEach((rho) => {
        ctx.fillText(rho.toFixed(2), xScale(rho) - 14, y + h + 18);
      });
      ctx.fillText("objective", x + w + 8, yScale(0) - 8);
    }

    function draw() {
      const epsilon = Number(epsilonSlider.value);
      epsilonValue.textContent = epsilon.toFixed(2);
      ctx.clearRect(0, 0, cssWidth, cssHeight);
      ctx.fillStyle = "#1f2023";
      ctx.fillRect(0, 0, cssWidth, cssHeight);
      drawPlot(epsilon);
      ctx.font = "13px system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.strokeStyle = "#8ab4f8";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(82, 382);
      ctx.lineTo(114, 382);
      ctx.stroke();
      ctx.fillStyle = "#c8cbd1";
      ctx.fillText("positive advantage", 122, 387);
      ctx.strokeStyle = "#fbbc04";
      ctx.beginPath();
      ctx.moveTo(300, 382);
      ctx.lineTo(332, 382);
      ctx.stroke();
      ctx.fillText("negative advantage", 340, 387);
      ctx.strokeStyle = "#c8cbd1";
      ctx.setLineDash([8, 6]);
      ctx.beginPath();
      ctx.moveTo(535, 382);
      ctx.lineTo(567, 382);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText("dashed = unclipped", 575, 387);
    }

    epsilonSlider.addEventListener("input", draw);
    draw();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPpoClipWidget, { once: true });
  } else {
    initPpoClipWidget();
  }
})();
