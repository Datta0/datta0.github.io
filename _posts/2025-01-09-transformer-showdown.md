---
layout: post
title: Transformer showdown
description: Comparing various transformer architectures like MHA, GQA, Multi Latent Attention, nGPT, Differential Transformer.
date: 2025-01-09 00:27 +0530
categories: [Transformer, Architectures]
tags: [MLA, MHA, GQA, Multi Latent Attention, nGPT, Differential Transformer]
render_with_liquid: false
math: true
---

# Transformer Showdown

For a long time, I had the urge to understand what advantages each of the changes to transformer architecture gives. I created a small repo that I like to call [nanoformer](https://github.com/Datta0/nanoformer) where I compare various architectures like MHA, GQA, Multi Latent Attention, nGPT and Differential Transformer. Switching between the mentioned architectures is as easy as modifying a CLI argument.
So without further ado, lets get comparing. 

## Attention variants:

- ### Multi Head Attention (MHA)
    The standard attention that was introduced as part of Attention is all you need. Very commonly used in a lot of models before 2023. Each layer has equal number of query, key and value heads. So if a layer has `h` heads, we'd have `h` queries, `h` keys and `h` values.

- ### Multi Query Attention (MQA)
    A small modification of MHA. Instead of having one key and value per query head, we'd only have a single key per token and all the query heads try to find similarity with that. This results in each layer having `h` queries, `1` key and `1` value. The advantage here is that if you're saving KVCache for speeding up inference, your KVCache is reduced by `h` times. But this is not as performant as MHA as we're reducing the scope of information stored in keys to only single vector.

- ### Grouped Query Attention (GQA)
    This acts as a middle ground between MHA and MQA. Instead of 1 key and value catering to all the queries, we have 1 key and value catering to a group of queries. So we'd have `h` queries, `k` keys and `k` values where `k` divides `h`. So for each layer, you'd be storing `2 * k` embedding vectors. You'd find a lot of models that use this architecture. Generally speaking, a single query caters to `4` to `8` queries. You can identify whether a model uses this when you see [`num_attention_heads`](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json#L16)`≠`[`num_kv_heads`](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json#L18) in model's config.json

- ### Multi Latent Attention (MLA)
    A new architecture found in DeepSeek V2 family of models. Here, we compress the Keys and values into a latent space and uncompress them back to original space when inference takes place. The idea is to get the advantages of MHA while saving up on KVCache as it scales linearly with context length. 

This image from DeepSeek V2 paper gives a crisp view of the above mentioned architectures.

![MHA vs GQA vs MQA vs MLA](assets/img/blogs/transformer_showdown/attn_variants.png)
_MHA vs GQA vs MQA vs MLA_


## Other Transformer Variants:

- ### Differential Transformer
    Introduced in a [2024 paper from Microsoft](https://arxiv.org/abs/2410.05258). The main motivation is that attention scores have a lot of noise. So if we have two subnetworks calculating attention, subtracting one from the other would act as subtracting random noise from information induced with noise. This helps control the attention scores and logits (outliers are of lesser magnitude). This is said to improve convergence. We discussed this in great detail in one of our other blogs on substack, [check it out.](https://datta0.substack.com/i/150138108/differential-transformer)

- ### nGPT
    Introduced in a [2024 paper from NVIDIA](https://arxiv.org/abs/2410.01131). The main idea is, if normalisation layers are so important to the performance of deep networks and LLMs, why not make normalistion mathemtically implicit to the network. Given this assumption, at every step, we try to make sure we're interacting with normalized vectors and only normalised vectors are passed on after every step. This too is said to improve convergence. We discussed this in great detail in one of our other blogs on substack, [check it out.](https://datta0.substack.com/i/151875954/ngpt-normalized-transformer)

