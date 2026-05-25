---
title: Systems for LLM RL
description: Foray into the systems challenges and approaches for LLM RL
author: datta0
date: 2026-05-30T23:00:00+05:30
categories: [LLM, Fine-tuning, RL, Math, Systems, GPU]
tags: [LLM, Fine-tuning, RL, Math, Systems GPU]
render_with_liquid: false
math: true
image:
  path: /assets/img/blogs/systems_for_llm_rl/rl_for_llm_header_wide.jpg
  alt: Reinforcement learning collage showing rewards, actions, and exploration
  no_bg: true
---

## Introduction

We have previously delved into [RL for LLMs](https://datta0.github.io/posts/rl-for-llms/) where we talked about how the math is formulated, how we go from a chat bot type model to one that thinks, reasons, solves math and writes code at super human levels of performance. But all this doesn't come for free. Today we look at the systems challenges for the same and also look at how people go about solving them. 


## A brief about RL for LLMs

If you haven't read the previous post yet, I strongly recommend you reading that first. But if you don't want to, here's a short summary.

### TLDR: RL for LLMs

- RL can go beyond SFT when the reward is clear. Learning over imitation.
- REINFORCE is the backbone for policy-gradient methods.
- Baseline subtraction reduces variance. KL divergence with respect to the SFT/reference model acts as an anchor. Trust-region clipping improves stability.
- PPO is a major RLHF algorithm for LLMs, but it commonly uses an actor, reward model, value model, and reference model.
- DPO makes the reward implicit. Only chosen-rejected preference pairs are needed. No separate reward or critic model during policy training.
- GRPO uses group statistics as the baseline. No critic is needed. Verifiable rewards are the big deal here.

| Method | Needs online rollouts? | Needs reward model? | Needs critic/value model? | Best fit |
|---|---:|---:|---:|---|
| PPO | Yes | Often | Yes | RLHF with learned rewards |
| DPO | No | No | No | Pairwise preference data |
| GRPO | Yes | No for RLVR | No | Math, code, and verifiable rewards |

### The systems requirements
Most of the RL algorithms need samples generated from the model we're optimising for. When you generate samples from the same policy/model that you're optimising, it is called **On Policy Learning**. On the other hand, if you generate samples from another policy perhaps different model or perhaps model that is few optimisation steps behind ([like we discussed in PPO](https://datta0.github.io/posts/rl-for-llms/#:~:text=So%20to%20make%20it%20more,needs%20to%20be%20mathematically%20addressed%2E)). This is called **Off Policy Learning**

In the case of PPO and GRPO because we generate the samples from the model that is being optimized those all come under online reinforcement learning. But we do accept some degree of policyness in the same. In case of a chess playing agent, on-policy learning is where it generates the moves and tries to learn from them. Whereas off-policy is basically it watching someone else generate the moves and learn from them. PPO is somewhere in between, where you generate one sequence of moves and try to do multiple gradient updates on the same taking advantage of [importance sampling](https://datta0.github.io/posts/rl-for-llms/#the-importance-sampling-correction).

As we previously discussed, unlike SFT, Generation phase involves multiple forward passes through the model. So it is safe to assume that around 90% of the time for RL on policy training step goes towards rollout generation especially at higher sequence lengths. So it is very important to optimize inference both from a speed standpoint as well as from a memory usage standpoint. A lot of frameworks rely on hyper optimized inference engines like [vLLM](https://vllm.ai/) or [SGLang](https://sgl-project.github.io/) and offload tasks to them. 

Though this does come with some benefits, it also comes with its own challenges. 
- Inference server, whether colocated (on same GPU(s)) or running standalone (on different GPUs), takes up GPU memory. 
- Trainer, the model we're optimising also takes up GPUs
- Reference model needs its own share
- In cases like PPO, you also need a Value Model and its own optimisation
- Once you introduce a separate process to generate rollouts (like vLLM), you need to make sure it stays up to date with the weights.

Lets go through this methodically and try to analyse the memory requirement of the same. 

### VRAM is gone...
For training these are the biggest components that consume VRAM. Assuming that the model is `n` billion params
- Weights: `n` billion = 2n GB in BF16
- Graidents: `n` billion = 2n GB in BF16
- Optimizer States: 2 per param for Adam: 2 moving averages (moments) .`2n` billion  = 8n GB in FP32 (typically higher precision)
- Master weights: `n` billion: 4nGB sometimes stored to perform accumulation with optimiser.
- Activations: Assuming you do activation checkpointing and use techniques like flash attention, this won't be much compared to the rest.

Do note that you can also offload optimizer states and gradients to CPU as well. But any offload is bound to cause slowdowns. For activations, you offload some, persist some, recalculate some etc.

For inference server:
Assuming that you have `l` layers, $n_h$ query heads, $n_{kv}$ key-value heads each of dimension hs and MLP dimension of `i` for sequence length `s` and batch size `b`...
- Weights: `n` billion = 2n GB in BF16
- Activations: Approximately $4 \times s \times b \times ((n_h + n_{kv}) \times hs + 3 \times i )$ bytes
- KVCache: Each attn head has its own and so does each layer: $4 \times l \times b \times s \times (n_{kv} \times hs)$ bytes

So lets take the example of [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) which has 36 layers, 32 attn heads, 8 KV heads, hidden size of 4096 and MLP dimension of 12288. This is how the memory requirement looks like:

<style>
  #llm-rl-memory-widget {
    --mem-bg: color-mix(in srgb, var(--card-bg, var(--main-bg, #fff)) 92%, var(--text-color, #222));
    --mem-border: color-mix(in srgb, var(--main-border-color, #d8d8d8) 72%, var(--text-muted-color, #777));
    --mem-panel: color-mix(in srgb, var(--main-bg, #fff) 88%, var(--text-color, #222));
    --mem-muted: var(--text-muted-color, #6f6f6f);
    --mem-text: var(--text-color, #2f2f35);
    --mem-heading: var(--heading-color, #222);
    --mem-blue: #3b82f6;
    --mem-green: #16a34a;
    --mem-amber: #f59e0b;
    margin: 1rem 0 1.6rem;
    padding: 1rem;
    border: 1px solid var(--mem-border);
    border-radius: 8px;
    background: var(--mem-bg);
    color: var(--mem-text);
  }

  #llm-rl-memory-widget * {
    box-sizing: border-box;
  }

  #llm-rl-memory-widget .memory-widget-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  #llm-rl-memory-widget h4 {
    margin: 0 0 0.2rem;
    color: var(--mem-heading);
    font-size: 1.05rem;
  }

  #llm-rl-memory-widget .memory-widget-subtitle,
  #llm-rl-memory-widget .memory-widget-note {
    margin: 0;
    color: var(--mem-muted);
    font-size: 0.88rem;
    line-height: 1.45;
  }

  #llm-rl-memory-widget .memory-total {
    min-width: 8rem;
    text-align: right;
  }

  #llm-rl-memory-widget .memory-total span {
    display: block;
    color: var(--mem-heading);
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
  }

  #llm-rl-memory-widget .memory-controls {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.9rem;
    margin-bottom: 1rem;
  }

  #llm-rl-memory-widget .memory-control {
    padding: 0.75rem;
    border: 1px solid var(--mem-border);
    border-radius: 8px;
    background: var(--mem-panel);
  }

  #llm-rl-memory-widget .memory-control label {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.8rem;
    margin-bottom: 0.55rem;
    color: var(--mem-heading);
    font-size: 0.92rem;
    font-weight: 700;
  }

  #llm-rl-memory-widget .memory-control output {
    color: var(--mem-muted);
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    white-space: nowrap;
  }

  #llm-rl-memory-widget input[type="range"] {
    width: 100%;
    accent-color: var(--link-color, #2563eb);
  }

  #llm-rl-memory-widget select,
  #llm-rl-memory-widget input[type="number"] {
    width: 100%;
    min-height: 2.25rem;
    padding: 0.35rem 0.5rem;
    border: 1px solid var(--mem-border);
    border-radius: 6px;
    background: var(--main-bg, #fff);
    color: var(--mem-text);
  }

  #llm-rl-memory-widget .memory-bar {
    display: flex;
    width: 100%;
    height: 2rem;
    overflow: hidden;
    border: 1px solid var(--mem-border);
    border-radius: 999px;
    background: var(--mem-panel);
  }

  #llm-rl-memory-widget .memory-bar-segment {
    min-width: 0.2rem;
    transition: width 140ms ease;
  }

  #llm-rl-memory-widget .memory-bar-segment.weights {
    background: var(--mem-blue);
  }

  #llm-rl-memory-widget .memory-bar-segment.activations {
    background: var(--mem-green);
  }

  #llm-rl-memory-widget .memory-bar-segment.kv-cache {
    background: var(--mem-amber);
  }

  #llm-rl-memory-widget .memory-breakdown {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.8rem 0 0.9rem;
  }

  #llm-rl-memory-widget .memory-stat {
    padding: 0.65rem 0.75rem;
    border: 1px solid var(--mem-border);
    border-radius: 8px;
    background: var(--mem-panel);
  }

  #llm-rl-memory-widget .memory-stat-label {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    color: var(--mem-muted);
    font-size: 0.82rem;
  }

  #llm-rl-memory-widget .memory-dot {
    width: 0.65rem;
    height: 0.65rem;
    flex: 0 0 auto;
    border-radius: 50%;
  }

  #llm-rl-memory-widget .memory-stat strong {
    display: block;
    margin-top: 0.25rem;
    color: var(--mem-heading);
    font-size: 1.25rem;
    font-variant-numeric: tabular-nums;
  }

  #llm-rl-memory-widget details {
    margin: 0.85rem 0;
  }

  #llm-rl-memory-widget summary {
    cursor: pointer;
    color: var(--link-color, #2563eb);
    font-weight: 700;
  }

  #llm-rl-memory-widget .memory-advanced-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.7rem;
    margin-top: 0.75rem;
  }

  #llm-rl-memory-widget .memory-number label {
    display: block;
    margin-bottom: 0.3rem;
    color: var(--mem-muted);
    font-size: 0.8rem;
    font-weight: 700;
  }

  @media (max-width: 680px) {
    #llm-rl-memory-widget .memory-widget-header,
    #llm-rl-memory-widget .memory-controls,
    #llm-rl-memory-widget .memory-breakdown,
    #llm-rl-memory-widget .memory-advanced-grid {
      grid-template-columns: 1fr;
    }

    #llm-rl-memory-widget .memory-widget-header {
      display: grid;
    }

    #llm-rl-memory-widget .memory-total {
      text-align: left;
    }
  }
</style>

<div id="llm-rl-memory-widget">
  <div class="memory-widget-header">
    <div>
      <h4>Qwen3-8B rollout memory sketch</h4>
      <p class="memory-widget-subtitle">Change the rollout shape and watch where the inference-side VRAM goes.</p>
    </div>
    <div class="memory-total">
      <span id="llm-rl-total-memory">0.0 GiB</span>
      <p class="memory-widget-subtitle">total</p>
    </div>
  </div>

  <div class="memory-controls">
    <div class="memory-control">
      <label for="llm-rl-seq-len">Sequence length <output id="llm-rl-seq-len-value">4096</output></label>
      <input id="llm-rl-seq-len" type="range" min="512" max="32768" step="512" value="4096">
    </div>
    <div class="memory-control">
      <label for="llm-rl-batch-size">Batch size <output id="llm-rl-batch-size-value">16</output></label>
      <input id="llm-rl-batch-size" type="range" min="1" max="128" step="1" value="16">
    </div>
    <div class="memory-control">
      <label for="llm-rl-weight-dtype">Weight dtype <output id="llm-rl-weight-dtype-value">BF16 / FP16</output></label>
      <select id="llm-rl-weight-dtype">
        <option value="2" selected>BF16 / FP16 weights</option>
        <option value="4">FP32 weights</option>
      </select>
    </div>
  </div>

  <div class="memory-bar" aria-label="Memory usage breakdown">
    <div id="llm-rl-weights-bar" class="memory-bar-segment weights"></div>
    <div id="llm-rl-activations-bar" class="memory-bar-segment activations"></div>
    <div id="llm-rl-kv-cache-bar" class="memory-bar-segment kv-cache"></div>
  </div>

  <div class="memory-breakdown">
    <div class="memory-stat">
      <div class="memory-stat-label"><span class="memory-dot" style="background: var(--mem-blue);"></span>Weights</div>
      <strong id="llm-rl-weights-memory">0.0 GiB</strong>
    </div>
    <div class="memory-stat">
      <div class="memory-stat-label"><span class="memory-dot" style="background: var(--mem-green);"></span>Activations</div>
      <strong id="llm-rl-activations-memory">0.0 GiB</strong>
    </div>
    <div class="memory-stat">
      <div class="memory-stat-label"><span class="memory-dot" style="background: var(--mem-amber);"></span>KV cache</div>
      <strong id="llm-rl-kv-cache-memory">0.0 GiB</strong>
    </div>
  </div>

  <details>
    <summary>Model shape controls</summary>
    <div class="memory-advanced-grid">
      <div class="memory-number">
        <label for="llm-rl-layers">Layers</label>
        <input id="llm-rl-layers" type="number" min="1" max="256" step="1" value="36">
      </div>
      <div class="memory-number">
        <label for="llm-rl-hidden-size">Hidden size</label>
        <input id="llm-rl-hidden-size" type="number" min="128" max="32768" step="128" value="4096">
      </div>
      <div class="memory-number">
        <label for="llm-rl-query-heads">Query heads</label>
        <input id="llm-rl-query-heads" type="number" min="1" max="256" step="1" value="32">
      </div>
      <div class="memory-number">
        <label for="llm-rl-kv-heads">KV heads</label>
        <input id="llm-rl-kv-heads" type="number" min="1" max="256" step="1" value="8">
      </div>
      <div class="memory-number">
        <label for="llm-rl-intermediate-size">MLP size</label>
        <input id="llm-rl-intermediate-size" type="number" min="128" max="131072" step="128" value="12288">
      </div>
    </div>
  </details>

  <p class="memory-widget-note">Approximate inference/rollout memory only. This excludes framework overhead, fragmentation, CUDA workspaces, paged KV metadata, and trainer-side gradients or optimizer states. Activation and KV-cache estimates use the constants from the formulas above; the dtype selector only changes stored model weights.</p>
</div>

<script>
(() => {
  const widget = document.getElementById("llm-rl-memory-widget");
  if (!widget) return;

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
    Object.entries(ids).map(([key, id]) => [key, document.getElementById(id)])
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
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function setWidth(id, bytes, total) {
    const el = document.getElementById(id);
    if (!el) return;
    const pct = total > 0 ? (bytes / total) * 100 : 0;
    el.style.width = `${pct}%`;
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
    const activations = 4 * seqLen * batchSize * (((queryHeads + kvHeads) * headDim) + (3 * intermediateSize));
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

  Object.values(controls).forEach((control) => {
    if (control) control.addEventListener("input", update);
  });

  update();
})();
</script>
