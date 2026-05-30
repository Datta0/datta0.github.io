---
title: Systems for LLM RL
description: Foray into the systems challenges and approaches for LLM RL
author: datta0
date: 2026-05-30T00:00:00+05:30
categories: [LLM, Fine-tuning, RL, Math, Systems, GPU]
tags: [LLM, RL, Systems, GPU, Training, Inference, GRPO]
math: true
image:
  path: /assets/img/blogs/systems_for_llm_rl/systems_for_llm_rl_header_wide.jpg
  alt: Collage of SFT versus RL memory, async rollout scheduling, and GPU server systems for LLM RL
  no_bg: true
---

## Introduction

We have previously delved into [RL for LLMs](https://datta0.github.io/posts/rl-for-llms/) where we talked about how the math is formulated, how we go from a chatbot-style model to one that thinks, reasons, solves math, and writes code at superhuman levels of performance. But all this doesn't come for free. Today we look at the systems challenges for the same and also look at how people go about solving them.


## A brief about RL for LLMs

If you haven't read the previous post yet, I strongly recommend reading that first. But if you don't want to, here's a short summary.

### TLDR: RL for LLMs

- RL can go beyond SFT when the reward is clear. Learning over imitation.
- REINFORCE is the backbone for policy-gradient methods.
- Baseline subtraction reduces variance. KL divergence with respect to the SFT/reference model acts as an anchor. Trust-region clipping improves stability.
- PPO is a major RLHF algorithm for LLMs, but it commonly uses an actor, reward model, value model, and reference model.
- DPO makes the reward implicit. Only chosen-rejected preference pairs are needed. No separate reward or critic model during policy training.
- GRPO uses group statistics as the baseline. No critic is needed. Verifiable rewards are the big deal here.

| Method | Online rollouts /<br>on-policy updates? | Reward<br>model? | Critic /<br>value model? | Best fit |
|---|---:|---:|---:|---|
| PPO | ~Yes | Often | Yes | RLHF with<br>learned rewards |
| DPO | No | No | No | Pairwise<br>preference data |
| GRPO | ~Yes | No<br>for RLVR | No | Math, code,<br>verifiable rewards |

### The systems requirements
Most of the RL algorithms need samples generated from the model we're optimising for. When you generate samples from the same policy/model that you're optimising, it is called **on-policy learning**. On the other hand, if you generate samples from another policy, perhaps a different model or a model that is a few optimisation steps behind ([like we discussed in PPO](https://datta0.github.io/posts/rl-for-llms/#:~:text=So%20to%20make%20it%20more,needs%20to%20be%20mathematically%20addressed%2E)), this is called **off-policy learning**.

In the case of PPO and GRPO, because we generate samples from the model that is being optimized, they come under online reinforcement learning. But we do accept some degree of off-policy drift. In the case of a chess-playing agent, on-policy learning is where it generates the moves and tries to learn from them. Off-policy is basically watching someone else generate the moves and learning from them. PPO is somewhere in between, where you generate one sequence of moves and try to do multiple gradient updates on it, taking advantage of [importance sampling](https://datta0.github.io/posts/rl-for-llms/#the-importance-sampling-correction).

As we previously discussed, unlike SFT, the generation phase involves multiple forward passes through the model. So it is safe to assume that a large fraction of each on-policy RL training step, often the majority at higher sequence lengths, goes towards rollout generation. So it is very important to optimize inference both from a speed standpoint and from a memory usage standpoint. Many frameworks rely on hyper-optimized inference engines like [vLLM](https://vllm.ai/) or [SGLang](https://sgl-project.github.io/) and offload tasks to them.

Though this does come with some benefits, it also comes with its own challenges. 
- Inference server, whether colocated (on same GPU(s)) or running standalone (on different GPUs), takes up GPU memory. 
- Trainer, the model we're optimising, also takes up GPU memory
- Reference model needs its own share
- In cases like PPO, you also need a value model and its own optimisation
- Once you introduce a separate process to generate rollouts (like vLLM), you need to make sure it stays up to date with the weights.

At a high level, this is why online RL feels much heavier than SFT. SFT mostly keeps one training stack alive: model weights, activations, gradients, and optimizer state. GRPO/RL adds a rollout engine that must generate thinking traces/completions, maintain inference-time state like KV cache, score samples with rewards/verifiers, and then sync updated trainer weights back into the inference engine. So the problem is not just "more memory"; it is also coordinating two systems that both need a fresh enough copy of the same policy.

![SFT versus GRPO systems](/assets/img/blogs/systems_for_llm_rl/sft_vs_rl.jpg)
*Compared with SFT, online RL adds rollout memory and a weight-sync path between trainer and inference.*

Let's go through this methodically and try to analyse the memory requirement of the same.

### VRAM is gone...
For training these are the biggest components that consume VRAM. Assuming that the model is `n` billion params
- Weights: `n` billion = 2n GB in BF16
- Gradients: `n` billion = 2n GB in BF16
- Optimizer states: 2 per param for Adam: 2 moving averages (moments). `2n` billion = 8n GB in FP32 (typically higher precision)
- Master weights: `n` billion = 4n GB, sometimes stored to perform accumulation with the optimiser.
- Activations: Assuming you do activation checkpointing and use techniques like flash attention, this won't be much compared to the rest.

Do note that you can also offload optimizer states and gradients to CPU as well. But any offload is bound to cause slowdowns. For activations, you offload some, persist some, recalculate some, etc.

For inference server:
Assuming that you have `l` layers, $n_h$ query heads, $n_{kv}$ key-value heads each of dimension `hs`, and MLP dimension `i` for sequence length `s` and batch size `b`...
- Weights: `n` billion = 2n GB in BF16
- Activations: Approximately $4 \times s \times b \times ((n_h + n_{kv}) \times hs + 3 \times i )$ bytes
- KVCache: Each attn head has its own and so does each layer: $4 \times l \times b \times s \times (n_{kv} \times hs)$ bytes

So let's take the example of [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json), which has 36 layers, 32 attention heads, 8 KV heads, a hidden size of 4096, and an MLP dimension of 12288. This is how the memory requirement looks like:

{% include widgets/llm-rl-memory-widget.html %}

Even if you have multiple GPUs, this much memory requirement is a lot to deal with. So we need to look for ways to optimise this.

## The lifecycle
First, let's see what steps are involved in this setup so that we can identify potential places to improve.

```
-> sync weights from trainer to inference engine
-> Inference Engine generates rollouts/completions # Inference engine is only involved here and trainer is not needed 
-> Trainer forward pass -> trainer backward pass
-> Gradients (accumulate) -> Optimizer step -> Update weights 
```

## Shut it down, spin it back up

If you look closely, the trainer phase and inference engine phase sequentially follow each other with no overlap. That should give us some idea of what we can do. If we can start up and tear down the inference server instantaneously, we can free up the space occupied by `weights + activations + KVCache` so that space can be used for optimiser states and gradients.

Before going further, it helps to separate two mechanisms that are easy to conflate:
- **Sharding (FSDP / DeepSpeed ZeRO):** each rank only ever holds `1/N` of the weights, gradients and optimiser states. This is always on and effectively free. It is the *primary* way training memory is kept in check, not offloading.
- **Host (CPU) offload:** moving that `1/N` shard across PCIe into CPU RAM. This is a separate, opt-in toggle. It frees HBM, but PCIe is slow, so round-tripping optimiser state (~12 bytes/param for Adam) every step costs real wall-clock time.

Now once the forward and backward passes finish, you *can* offload the trainer's resident shard to CPU RAM, assuming fast enough CPU-GPU communication and enough host memory. But whether this actually happens depends heavily on the topology:
- **Colocated** (trainer and inference share the same GPUs): offloading the trainer and/or sleeping vLLM is *necessary* because both contend for the same HBM. The swap happens at the rollout↔train phase boundary, not per micro-batch.
- **Disaggregated** (trainer and inference on different GPUs): no swap is needed at all. The inference GPUs simply never hold trainer state, and any CPU offload there is a static memory-pressure choice unrelated to the RL lifecycle.

I'd leave it as an exercise for you to think about why/how this works when using `gradient_accumulation_steps` and/or when `trainer_batch_size` and `inference_batch_size` are not the same. Hint: you might need to repeat the same thing multiple times, but smartly.

This way, you can get away with only needing `~max(trainer_memory, inference_memory)`. This can be anywhere from 30%-50% less than having both in memory at the same time. The savings scale with increases in sequence/completion length. If you are doing LoRA, this can be closer to 50% if not more, because the gradients and optimiser states are a very small fraction of the memory requirement.

![Sleep mode timeslices GPU memory across rollout and training phases](/assets/img/blogs/systems_for_llm_rl/sleep_mode_rl.jpg)
*The optimization is temporal: keep only the phase that is actively using VRAM.*

The two big questions to answer here are
1. Can you spin up and shut down the inference engine quickly?
2. Can you offload the trainer components to CPU RAM quickly?

### Inference Engine Startup/Shutdown
If you have ever interacted with vLLM, the startup time can take a few seconds to a few minutes. It goes through a whole lot of steps. We will take the example of Qwen3-32B on a B200 GPU for the time taken. Note that all these were benchmarked on a single GPU. If you do this on a multi-GPU system, profiling will take longer due to inter-GPU communication overhead.
- Weight loading
  The inference engine needs to load weights from disk to GPU memory. ~75s
- Profiling
  vLLM runs a dummy forward pass with the selected input sizes to estimate the activation memory requirement. This is done so that once the vLLM server starts up, there's ~0% chance of it OOMing.
  You profile, understand memory requirements, and fail fast if it is not enough. ~32s
- CUDA Graph capture
  This is a very important step for performance. I have seen ~20-30% performance loss if you do not do this. ~9s
- KVCache allocation
  This is done to pre-allocate the KVCache memory so that we don't have to allocate it during inference. This is pretty instantaneous.


The problem is that if you shut down and restart the server, you're wasting at least a couple of minutes every step. If you want to do a large training run, this will be a significant portion of your total training time. If the generated tokens aren't too many, this can potentially take more time than generation and the forward/backward pass. So you don't want to do that, but you also don't want to lose the memory advantages you get from freeing up the inference engine memory.

The middle ground is [Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/) from vLLM. What it does is discard/offload weights and KVCache, which is roughly the bulk of the inference engine memory anyway.

There are two sleep modes that vLLM exposes:
- Level 1: Offload model weights to CPU. Discard KVCache
  This took ~30s to sleep and ~0.6s to wake up in the benchmark I am referring to.
- Level 2: Discard model weights and KVCache. No CPU offload
  This took ~0.24s to sleep and ~0.4s to wake up in the same benchmark.

For full finetuning RL workloads, we can choose to do Level 2 because the model weights are generally updated between sleep and wakeup, and we need the inference engine to have the latest weights anyway. The exception is LoRA, where you want the shared base weights to stay resident (more on that [later](#lora---weight-sharing)).

## Chunked loss calculation

When you have a model like Qwen3-8B for GRPO, assume that you have a sequence length of 16K and a batch size of 4. In BF16, the memory usage looks like:

- **Hidden States**: `bsz * hidden_size * seq_len = 16384 * 4096 * 4 * 2 bytes = 0.5 GiB`
- **Logits**: `bsz * vocab_size * seq_len = 16384 * 128K * 4 * 2 bytes = 16 GiB`

So if you materialise the logits, you're going to pay a hefty price. The moral is to not materialise the logits. Hidden states are much easier on GPU memory, and we need to store or recompute them anyway because every subsequent layer works with the same shape/size.

$$
\begin{aligned}
\mu_G &= \frac{1}{G}\sum_{j=1}^{G} r_j \\
\sigma_G &= \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j-\mu_G)^2} \\
\hat{A}_i &= \frac{r_i - \mu_G}{\sigma_G + \epsilon_{\text{std}}}
\end{aligned}
$$

$$
\mathcal{L}_{\mathrm{GRPO}}(\theta)
=
-\mathbb{E}_{q,\{o_i\}_{i=1}^{G}}
\left[
\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\ell_{i,t}(\theta)
\right]
$$

$$
\ell_{i,t}(\theta)
=
\min\left(
\rho_{i,t}(\theta)\hat{A}_i,\,
\operatorname{clip}\left(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_i
\right)
- \beta D_{\mathrm{KL}}\left(
\pi_\theta(\cdot|s_{i,t})
\,\|\, \pi_{\mathrm{ref}}(\cdot|s_{i,t})
\right)
$$

The loss is just a sum over the tokens of a sequence which is then aggregated over the group. So one token's contribution to the loss is independent of other tokens in the same sequence once its log probability, advantage, and KL term are known. Instead of trying to materialise and calculate everything at once, what we can do is write a kernel which iterates over the tokens, calculates the loss for that token, and accumulates it. This way, we don't have to materialise the logits and can keep memory usage low. We talked more about GPU architecture, how streaming output can be calculated, and how kernels can be useful in our previous blog [The MathemaTricks behind Flash Attention](https://datta0.github.io/posts/flash-attention/#a-slight-detour-into-gpu-architecture).


> **Pro Tip**: Anytime there is an increase in dimension followed by a reduction, try to think if you can do the reduction in chunks. This can help you avoid materialising the intermediate result and save a lot of memory. You see the same trick in [Flash Attention](https://datta0.github.io/posts/flash-attention/), [Cut Cross Entropy](https://arxiv.org/abs/2411.09009), Fused MLP etc. But we also need to make sure our backward pass is appropriately dealt with and we save enough info for that. 
{: .prompt-tip }

```python
# Simplified pseudocode: real implementations gather only sampled-token logprobs.
# hidden_states: (bsz*seq_len, hidden_dim)
# lm_head: (hidden_size, vocab_size)
def batched_chunked_loss(old_hidden_states, new_hidden_states, lm_head):
  total_loss = 0
  old_hidden_states = old_hidden_states.reshape(-1, mini_batch_size, hidden_dim)
  new_hidden_states = new_hidden_states.reshape(-1, mini_batch_size, hidden_dim)
  for old_mini_batch, new_mini_batch in zip(old_hidden_states, new_hidden_states):
    # Process each mini-batch
    # (mini_batch_size, hidden_dim) @ (hidden_dim, vocab_size) -> (mini_batch_size, vocab_size)
    old_mini_logits = old_mini_batch @ lm_head  # (mini_batch_size, vocab_size)
    old_mini_logps = F.log_softmax(old_mini_logits, dim=-1)  # (mini_batch_size, vocab_size)
    new_mini_logits = new_mini_batch @ lm_head  # (mini_batch_size, vocab_size)
    new_mini_logps = F.log_softmax(new_mini_logits, dim=-1)  # (mini_batch_size, vocab_size)
    log_ratio = new_mini_logps - old_mini_logps  # (mini_batch_size, vocab_size)
    mini_batch_loss = clip(log_ratio, advantages).sum(dim=-1) # (mini_batch_size, 1)
    total_loss += mini_batch_loss
    pass
  return total_loss # (bsz * seq_len, 1)
```

Another optimization is to do the LM head multiplication in chunks where you split the LM head itself. But this would not directly resolve into any individual unit of the loss. So the accumulation would be slightly more involved. If you think about it, this is very similar to how we do [Column Parallel vs Row Parallel in Tensor Parallelism](https://datta0.github.io/posts/understanding-multi-gpu-parallelism-paradigms/#tensor-parallelism). Also for how online softmax is implemented, please refer to the same in [The MathemaTricks behind Flash Attention](https://datta0.github.io/posts/flash-attention/#stabilizing-the-exponent). The key idea here is that for a selected token, we need the target token logit and the softmax denominator over the full vocabulary. For the denominator, we need to look at all logits, but we need not store/persist them. They can be maintained via a running max and running sum while we iterate over the chunks.



```python
# Simplified pseudocode.
# hidden_states: (bsz*seq_len, hidden_dim)
# lm_head: (hidden_size, vocab_size)
def chunked_lm_head_mul(old_hidden_states, new_hidden_states, lm_head):
  def get_logps(hidden_states, lm_head):
    _, vocab_size = lm_head.shape
    N = hidden_states.shape[0] # bsz * seq_len: one per token per seq
    prev_max = None
    sampled_token_logit = torch.zeros((N,), device=device, dtype=torch.float32)
    for start_idx in range(0, vocab_size, chunk_size):
      end_idx = min(start_idx + chunk_size, vocab_size)
      chunk = lm_head[:, start_idx:end_idx] # (hidden_size, chunk_size)
      # Process chunk
      partial_logits = hidden_states @ chunk  # (bsz*seq_len, chunk_size)
      current_max = partial_logits.max(dim=-1, keepdim=True)
      partial_logps = online_softmax(partial_logits, current_max, prev_max) # (bsz*seq_len, chunk_size)
      prev_max = current_max
      mask = (sampled_tokens >= start_idx) & (sampled_tokens < end_idx) # update for the chunk
      # each token in a batch/seq has its own vocab ID. they might or might not be in the current chunk
      # so we do a masked update
      sampled_token_logit[mask] = partial_logps[mask, sampled_tokens[mask] - start_idx]
      pass
    lse_vocab = row_max + log(row_sum)
    logp = sampled_token_logit - lse_vocab # log(exp(target_logit) / sum(exp(logits)))
    return logp
  
  old_logp = get_logps(old_hidden_states, lm_head) # (bsz, seq_len, 1)
  new_logp = get_logps(new_hidden_states, lm_head)
  log_ratio = new_logp - old_logp
  loss = clip(log_ratio, advantage).sum(dim=-1) # (bsz, seq_len)
  return loss
```

![Chunked Loss Calculation](/assets/img/blogs/systems_for_llm_rl/chunked_loss_calc.jpg)
*Chunked loss calculation*

## LoRA - Weight Sharing
By now you wouldn't need a re-introduction to LoRA. Basically, instead of training full weights, you train an adapter (which is like a lego block) which keeps gradients and optimiser states in check, thus saving a lot of memory. For further details, read [The Lore behind LoRA](https://datta0.github.io/posts/the-lore-behind-lora/).

![LoRA shrinks train state but still duplicates base weights in the simple colocated setup](/assets/img/blogs/systems_for_llm_rl/lora_for_rl.jpg)

Because the base weights are frozen, for training, there is no need to track gradients or optimizer states for them, and the weights are left in inference-only mode. This is what vLLM also has anyway. So instead of storing two copies of the base weights, one for the trainer and one for the inference engine, we can load them once and reuse them. What we end up doing is loading the vLLM instance, loading a dummy Hugging Face trainer model, pointing each of the weights to the pointers of vLLM weights, and loading LoRA on top of that.

You can combine this optimization with sleep mode, but the only thing is you do not want to offload or discard the model weights because they are used by the trainer. Another side effect, which turns out to be an advantage, is that you save the time required to offload and onload weights. vLLM exposes a [LoRA request](https://docs.vllm.ai/en/latest/features/lora/) mechanism to load the LoRA adapter for serving inference requests. One final optimization is to not store the LoRA adapter to disk and reload it from disk for the LoRA request, but load it in memory so that we skip the synchronization and copy path.

![Weight sharing with vLLM sleep mode keeps shared weights resident](/assets/img/blogs/systems_for_llm_rl/lora_plus_opt.jpg)
*When weights are shared, sleep mode should free transient inference state, not the shared base weights.*

This can also be done for QLoRA as long as vLLM and the trainer agree on the quantized tensor storage layout. This can get trickier when kernels pack or reshape quantized tensors, especially for MoE models. So you have to be careful there.

The LoRA weight-sharing and sleep-mode style optimizations are a part of Unsloth, and it supports a lot of popular models like Qwen, Llama, Mistral, and Gemma, including their vision-language model families. All you need to do to enable this is

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
    fast_inference = True, # Enables weight sharing
    unsloth_vllm_standby = True, # Enables sleep mode optimisation
)
```

## Async GRPO

Assuming that you are not GPU poor and have multiple GPUs to play around with for GRPO, one optimization you can explore is to perform the rollouts and trainer passes simultaneously. In general, synchronous and sequential is preferred because it is mathematically cleaner. But it has been observed multiple times that even when the weights are nominally the same, kernel changes, batch-size changes, precision choices, and inference-engine differences can create a drift between the logits or completions generated by the trainer and those generated by the inference engine. People tend to use importance sampling and other correction methods to handle this. But if you are going that far anyway, why not accept that there is a slight drift and not enforce strictly sequential generation/rollout and training?

This is exactly what Async GRPO does, where the inference engine on one GPU generates rollouts for the second batch while the trainer is forward passing through the first batch. This means that the inference engine is behind by a step or two when it comes to the weights. But because we do trust-region-style clipping and other stabilizers, a couple of steps of lag may be recoverable with importance sampling.

![Async GRPO pipeline overlaps rollout and training on different GPUs](/assets/img/blogs/systems_for_llm_rl/async_rl.jpg)
*Async GRPO overlaps rollout and training, accepting small policy lag for higher utilization.*

You can go even further and make sure that the weight sync doesn't pause rollout generation. As in, while tokens are being generated, you update the weights one chunk at a time. So there is a possibility that within a single completion, some tokens are generated using older weights and some are generated using newer weights. The Prime Intellect 3 paper mentions that with Async/Pipeline GRPO, they observed a speedup of 2x compared to the version without those optimizations. But one needs to be careful to make sure that we are not overdoing these optimizations where there is a mismatch between the trainer and inference engine. If the difference becomes too high, then we are essentially doing off-policy learning instead of on-policy learning, which is less robust.

![Pipeline GRPO makes weight sync non-blocking](/assets/img/blogs/systems_for_llm_rl/pipeline_rl.jpg)
*Pipeline GRPO lets rollout continue while newer weights arrive, trading exact synchronization for utilization.*

The advantages may be more pronounced for larger-scale model training, especially across multiple nodes where weight sync time is bound by network speed and can bottleneck the entire training. For intra-node communication on PCIe or NVLink, especially for small-scale model training, you might get away without these optimizations, but they are still worth measuring.

### Delta weight sync

There is also a reason for exploring the possibility of compressing the weight updates. Typically, when you do mixed-precision training, any change that is less than half the local spacing of the data type will be rounded away. For example, in BF16 you have

```
    _        _ _ _ _ _ _ _ _      _ _ _ _ _ _ _ _
Sign bit     8 exponent bits      7 mantissa bits
```

So between any two consecutive powers of 2, we have 7 bits of precision, aka 128 intervals to play with. For a weight of magnitude `|w|`, the spacing is approximately `|w|/128` within that exponent range. Anything less than roughly half that spacing will round away. It's like doing `10^3 += 0.1` but you can only represent whole numbers. For this to make any change in the value, it needs to be at least `0.5` so that rounding will carry it over to the next representable value (a whole number in the example, BF16 in reality).

Typically, the weights in LLMs are around the scale $1e^{-3}$. So the BF16 spacing around that value is approximately $1e^{-3}/128 \approx 7.8e^{-6}$. If you set the learning rate to $\approx 1e^{-5}$, then the effective optimizer update has to be large enough to survive BF16 rounding. With norm-based gradient clipping, low RL learning rates, and small gradients, you can end up in a tight regime. It is also empirically observed that approximately 99% of the BF16 weight deltas are zero in some mixed-precision RL training settings with lower learning rates. So one thing you can do is, instead of sharing the entire weight tensor over the network, identify the specific positions where updates have taken place and share only those values. If the updates are dense or the model is small, this bookkeeping can cost more, especially on high-speed interconnects like PCIe or NVLink. But over the network for large-scale models, the bookkeeping cost can be amortized and save a lot of time. [You can read more here](https://huggingface.co/blog/delta-weight-sync).

So instead of sharing an entire tensor of shape `(m, n)`, you share a map `{(i, j): new_value}` where `(i, j)` are the indices of the non-zero updates. Assuming that indices are integers and values are BF16, you are transferring more data per changed value, but you're only sharing a tiny fraction of the values. Of course, this also assumes that the inference engine can hot-swap particular weights and indices on the fly. There are open PRs on vLLM, but otherwise, this can be done via a patch or extension.


## Conclusion

Anytime ML researchers come up with a new algorithm, function, or mechanism, ML engineers have to sit and optimise the system to squeeze out every last bit of performance. Sometimes model-hardware co-design is done to avoid a lack of cohesion. But when doing something novel, you cannot possibly foresee and optimise for every scenario. After every new research innovation, there are usually systems optimisations that become visible if you look at things from a utilization standpoint and try to eliminate gaps. This is one such example of making things faster and more efficient. With that, as promised, I present you the systems side of LLM-RL. I'll take your leave. Happy learning! Stay curious...
