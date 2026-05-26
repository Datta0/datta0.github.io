(() => {
  function initLlmRlMemoryWidget() {
    const widget = document.getElementById("llm-rl-memory-widget");
    if (!widget || widget.dataset.memoryWidgetReady === "true") return;
    widget.dataset.memoryWidgetReady = "true";

    const paramsBillion = 8;
    const gib = 1024 ** 3;
    const ids = {
      seqLen: "llm-rl-seq-len",
      batchSize: "llm-rl-batch-size",
      dtype: "llm-rl-weight-dtype",
      layers: "llm-rl-layers",
      hiddenSize: "llm-rl-hidden-size",
      queryHeads: "llm-rl-query-heads",
      kvHeads: "llm-rl-kv-heads",
      intermediateSize: "llm-rl-intermediate-size",
    };

    const controls = Object.fromEntries(
      Object.entries(ids).map(([key, id]) => [key, widget.querySelector(`#${id}`)])
    );

    const readPositive = (control, fallback) => {
      const value = Number(control && control.value);
      return Number.isFinite(value) && value > 0 ? value : fallback;
    };

    const formatNumber = (value) => new Intl.NumberFormat("en-US").format(Math.round(value));
    const formatGiB = (bytes) => {
      const value = bytes / gib;
      return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} GiB`;
    };

    function setText(id, value) {
      const el = widget.querySelector(`#${id}`);
      if (el) el.textContent = value;
    }

    function setWidth(id, bytes, total) {
      const el = widget.querySelector(`#${id}`);
      if (!el) return;
      const pct = total > 0 ? (bytes / total) * 100 : 0;
      const width = `${pct}%`;
      el.style.width = width;
      el.style.flexBasis = width;
      el.title = `${formatGiB(bytes)} (${pct.toFixed(1)}%)`;
    }

    function update() {
      const seqLen = readPositive(controls.seqLen, 4096);
      const batchSize = readPositive(controls.batchSize, 16);
      const bytesPerParam = readPositive(controls.dtype, 2);
      const layers = readPositive(controls.layers, 36);
      const hiddenSize = readPositive(controls.hiddenSize, 4096);
      const queryHeads = readPositive(controls.queryHeads, 32);
      const kvHeads = readPositive(controls.kvHeads, 8);
      const intermediateSize = readPositive(controls.intermediateSize, 12288);
      const headDim = hiddenSize / queryHeads;

      const weights = paramsBillion * 1e9 * bytesPerParam;
      const activations =
        4 * seqLen * batchSize * ((queryHeads + kvHeads) * headDim + 3 * intermediateSize);
      const kvCache = 4 * layers * batchSize * seqLen * (kvHeads * headDim);
      const total = weights + activations + kvCache;

      setText("llm-rl-seq-len-value", formatNumber(seqLen));
      setText("llm-rl-batch-size-value", formatNumber(batchSize));
      setText("llm-rl-weight-dtype-value", bytesPerParam === 4 ? "FP32" : "BF16 / FP16");
      setText("llm-rl-total-memory", formatGiB(total));
      setText("llm-rl-weights-memory", formatGiB(weights));
      setText("llm-rl-activations-memory", formatGiB(activations));
      setText("llm-rl-kv-cache-memory", formatGiB(kvCache));

      setWidth("llm-rl-weights-bar", weights, total);
      setWidth("llm-rl-activations-bar", activations, total);
      setWidth("llm-rl-kv-cache-bar", kvCache, total);
    }

    widget.addEventListener("input", update);
    widget.addEventListener("change", update);
    update();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLlmRlMemoryWidget, { once: true });
  } else {
    initLlmRlMemoryWidget();
  }
})();
