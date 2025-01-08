---
layout: post
title: Transformer showdown MHA vs MLA vs nGPT vs Differential Transformer
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

    $$ Q_i = W_{q_i} X, \quad K_i = W_{k_i} X, \quad V_i = W_{v_i} X \quad \text {where X is the input}$$

    $$ A_i = \text{Attention}(Q_i, K_i, V_i) = softmax(\frac{Q_iK_i^T}{\sqrt{d_k}})V_i $$
    $$ A = [A_1, A_2, ..., A_h] \space @ \space  Wo $$

    - **Config** : `n` layers. Each layer has `h` heads. Each head has `d` dimensions. Total token count `t`. 
    - **Parameters**: $W_{q_i}$ is of shape $(h*d, d)$, so has $h^2*d^2$ parameters. Same for $W_{k_i}, W_{v_i}$. $W_o$ is of size $(h*d, h*d)$  so $h^2*d^2$ parameters. Total of $4*n*h^2*d^2$.
    - **Activations**: Each token's query is of size `d` (per head). The same is for key and value. Hence a total of $3*n*h*d$ per token. The final output is of same shape as well. The attention scores form a matrix of size $t*t$ hence $t^2$. So a total of $4*n*h*d*t + n*h*t^2$.
    - **KVCache**: Each key and value is of size `d` per token per head per layer. Hence a total of $2*n*h*d*t$.
        

- ### Multi Query Attention (MQA)
    A small modification of MHA. Instead of having one key and value per query head, we'd only have a single key per token and all the query heads try to find similarity with that. This results in each layer having `h` queries, `1` key and `1` value. The advantage here is that if you're saving KVCache for speeding up inference, your KVCache is reduced by `h` times. But this is not as performant as MHA as we're reducing the scope of information stored in keys to only single vector.

    $$ A_i = \text{Attention}(Q_i, K, V_i) = softmax(\frac{Q_iK^T}{\sqrt{d_k}})V_i $$
    $$ A = [A_1, A_2, ..., A_h] \space @ \space  Wo $$

    - **Parameters**: $W_{q_i}$ is of shape $(h*d, d)$, so has $h^2*d^2$ parameters. As for $W_{k_i}, W_{v_i}$, they output a single vector per head. So $(d,d)$ shape and hence $d^2$ parameters. $W_o$ is of size $(h*d, h*d)$  so $h^2*d^2$ parameters. Total of $2*n*h^2*d^2 + 2*n*h*d^2$.
    - **Activations**: Each token's query is of size `d` (per head) So $h*d$.There is only one key shared across all the heads hence only $2*d$ (key and value). Hence a total of $n*h*d*t + 2*n*d*t$. The final output is of same shape as query as well. The attention scores form a matrix of size $t*t$ hence $t^2$. So a total of $2*n*h*d*t + 2*n*d*t + n*h*t^2$.
    - **KVCache**: Each key and value is of size `d` per token per layer. Hence a total of $2*n*d*t$. A compression of `h` times compared to MHA.

- ### Grouped Query Attention (GQA)
    This acts as a middle ground between MHA and MQA. Instead of 1 key and value catering to all the queries, we have 1 key and value catering to a group of queries. So we'd have `h` queries, `k` keys and `k` values where `k` divides `h`. So for each layer, you'd be storing `2 * k` embedding vectors. You'd find a lot of models that use this architecture. Generally speaking, a single query caters to `4` to `8` queries. You can identify whether a model uses this when you see [`num_attention_heads`](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json#L16)`≠`[`num_kv_heads`](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json#L18) in model's config.json

    $$ A_i = \text{Attention}(Q_i, K_{i//g}, V_i) = softmax(\frac{Q_iK_{i//g}^T}{\sqrt{d_k}})V_i $$
    $$ A = [A_1, A_2, ..., A_h] \space @ \space  Wo $$

    - **Parameters**: $W_{q_i}$ is of shape $(h*d, d)$, so has $h^2*d^2$ parameters. As for $W_{k_i}, W_{v_i}$, they output a single vector per group of heads. So $(g*d,d)$ shape and hence $g*d^2$ parameters. $W_o$ is of size $(h*d, h*d)$  so $h^2*d^2$ parameters. Total of $2*n*h^2*d^2 + 2*n*g*h*d^2$.
    - **Activations**: Each token's query is of size `d` (per head) So $h*d$.There is one key per group of heads hence only $2*d*g$ (key and value). Hence a total of $n*h*d*t + 2*n*g*d*t$. The final output is of same shape as query as well. The attention scores form a matrix of size $t*t$ hence $t^2$. So a total of $2*n*h*d*t + 2*n*g*d*t + n*h*t^2$.
    - **KVCache**: Each key and value is of size `d` per token per layer per group. Hence a total of $2*n*g*d*t$. A compression of `h/g` times compared to MHA.

- ### MultiHead Latent Attention (MLA)
    A new architecture found in DeepSeek V2 family of models. Here, we compress the Keys and values into a latent space and uncompress them back to original space when inference takes place. The idea is to get the advantages of MHA while saving up on KVCache as it scales linearly with context length. Each key and value are compressed from `d` dimensions to `c` dimension space.

    $$ 
    c_t^{KV} = W^{DKV} X,  \quad  \text{ where } c_t^{KV} \in \R^{c} \quad \text { is down projection of keys }\\
    k_t^C = W^{UK} c_t^{KV} \quad \text {  up projection of keys} \\
    v_t^C = W^{UV} c_t^{KV}  \quad  \text {  up projection of values}
    $$

    $$ 
    c_t^{Q} = W^{DQ} X \quad \text{ where } c_t^{Q} \in \R^{c}  \\
    q_t^C = W^{UQ} c_t^{Q}
    $$



    $$ A_i = \text{Attention}(Q_i, K_i, V_i) = softmax(\frac{Q_i K_i^T}{\sqrt{d_k}})V_i $$
    $$ A = [A_1, A_2, ..., A_h] \space @ \space  Wo $$

    - **KVCache**: Each compressed vector is of size `c` per token per layer per group. Hence a total of $n*g*c*t$. Keys and values are inferred by decompressing this ($k_t^C, v_t^C$). A compression of `2*d/c` times compared to MHA. Note that in final implementation there's a nuance of additional heads (and hence keys and values) for RoPE. That adds a little more overhead. So the compression ratio essentially becomes $2*d/(c+r)$ where r is the RoPE key dimension.


This image from DeepSeek V2 paper gives a crisp view of the above mentioned architectures.

![MHA vs GQA vs MQA vs MLA](assets/img/blogs/transformer_showdown/attn_variants.png)
_MHA vs GQA vs MQA vs MLA_


## Other Transformer Variants:

- ### Differential Transformer
    Introduced in a [2024 paper from Microsoft](https://arxiv.org/abs/2410.05258). The main motivation is that attention scores have a lot of noise. So if we have two subnetworks calculating attention, subtracting one from the other would act as subtracting random noise from information induced with noise. This helps control the attention scores and logits (outliers are of lesser magnitude). This is said to improve convergence. We discussed this in great detail in one of our other blogs on substack, [check it out.](https://datta0.substack.com/i/150138108/differential-transformer)

    - Here owing to having two attention units, the number of paramters, activations and KVCache requirement goes up by a factor of 2 each as compared to GQA

- ### nGPT
    Introduced in a [2024 paper from NVIDIA](https://arxiv.org/abs/2410.01131). The main idea is, if normalisation layers are so important to the performance of deep networks and LLMs, why not make normalistion mathemtically implicit to the network. Given this assumption, at every step, we try to make sure we're interacting with normalized vectors and only normalised vectors are passed on after every step. This too is said to improve convergence. We discussed this in great detail in one of our other blogs on substack, [check it out.](https://datta0.substack.com/i/151875954/ngpt-normalized-transformer)

    - Apart from more normalisations there isn't much that would meaningfully contribute to parameters or activations or KVCache as compared to GQA.

So now that the introductions are out of the way, the burning question is do the changes contribute to any meaningful differences in the final performance of the models? 

Well the answer is nuanced. Let's see how they stack up.
