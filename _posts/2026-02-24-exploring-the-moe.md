---
title: Exploring the Mixture of Experts
description: An intuitive build up to Mixture of Experts
author: datta0
date: 2025-06-14T14:30:00+05:30
categories: [Mixture of Experts, Transformer, FFNN, Math]
tags: [Mixture of Experts, Transformer, FFNN, Math]
render_with_liquid: false
draft: true
math: true
image:
  path: /assets/img/blogs/mixture_of_experts/mixture_of_experts.jpg
  alt: Mixture of Experts imagined from the ground up
---

# Mixture of Experts

## Introduction

Previously we have talked about [Transformer and attention re-imagined](https://datta0.github.io/posts/transformer-imagined/) from the first principles. While that gives a very good understanding of how transformers work, specifically the attention mechanism, there has been a lot of progress ever since. One of the under-represented parts of the transformer is the FFNN. 
MoE or mixture of experts is a natural continuation of the same. Today we'll try to reimagine the same.

## FFNN
Let's start with the FFN. Attention is all cool but attention doesn't mix features with one another. It only brings you the information from the other tokens so get a context of what you're surrounded by thus giving you an idea of what you potentially are. But the model also needs to know which features of the token are important and which are potentially discardable given the context. This is pretty much the job of FFNN. Here we try to separate the features of interest from the non-interesting features. Given a token and its features, we project it from the model dimension to a higher dimension called the MLP dimension, or intermediate size, and apply a nonlinearity element-wise to achieve that.
In the recent past, we have Gated Linear unit as the buiding block which goes as follows:
$$
\begin{aligned}
X &\in \mathbb{R}^{n \times d} \\
W_{up}, W_{gate} &\in \mathbb{R}^{d \times d_{mlp}} \\
W_{down} &\in \mathbb{R}^{d_{mlp} \times d} \\
\\
g &= X @ W_{gate} \in \mathbb{R}^{n \times d_{mlp}} \\
u &= X @ W_{up} \in \mathbb{R}^{n \times d_{mlp}} \\
s &= \text{SiLU}(g) \\
gu &= s \odot u \\
\text{MLP}(X) &= gu @ W_{down} \in \mathbb{R}^{n \times d}
\end{aligned}
$$

For more information on how FFNN works, please refer to [this section](https://datta0.github.io/posts/transformer-imagined/#mlp-or-ffnn) in our previous blog.

## The motivation for MoE

If you notice closely, the non linearity used in general is SiLU or ReLU or GeLU. All of which pretty much set almost half the input values (negative ones) to zero.

![FFNN Activations](assets/img/blogs/moe_journey/activations.jpg)
_FFNN Activations_

After computing the gate proj we just basically throw away the corresponding features. Up proj being an element wise product with the output after the activation, wouldn't change the feature from 0. So we are potentially computing a lot of features which are not useful. What if we can avoid that?
Yeah we would like to solve exactly that problem. We would like to know which columns of the MLP matrix are useful for us and which are not. But how do we know which ones to avoid? Ideally it should depend on your hidden state just before the MLP. 


The simplest way to do it is to have a linear projection(yeah [like](https://datta0.github.io/posts/transformer-imagined/#the-transformations) [always](https://datta0.github.io/posts/transformer-imagined/#the-transformations:~:text=The%20simplest%20way%20to%20do%20that%20is%20to%20have%20a%20linear%20transformation%20%28yeah%20again%20%3A%29%29%2E)) that predicts the importance of a column for the given input. 
So for an input $X \in \R^{d}$ (single token), we need a matrix that takes `d` features and outputs us $d_mlp$ values. But that would be a $\R^{d \times d_{mlp}}$ matrix. So by doing this we're literally repeating the computation we'd have done for `gate_proj`. 

The input dimension cannot be compromised on. What can be changeed is the output dimension. What does it mean to the operations? Essentially we find a smaller dimension, say `n`, such that we predict `n` values per input and then use those values to pick `some` of the $d_{mlp}$ columns. But we should be careful as to not involve any more non simple, non trivial transformations on the said `n` that adds to the compute and parameter cost. When tasked with such constraints, always try to pick the simplest way out. So instead of having to take a decision for each of the columns, what if we take a single decision for a group of columns? This essentially is equivalent to saying, "Let us group $d_{mlp}$ columns into $k$ buckets, and for each bucket we decide whether to use it or not".  

Lets see how (much) that reduces our compute. If we multiply a matrix of shape `(m,k)` with another matrix of shape `(k,n)` to get an output of shape `(m,n)`, we're doing `O(mnk)` computations. You can visualise this by taking the perspective of the ouptut.
For each of the values in the output, we need to do a dot product between vectors of size `k`. So we do `k` multiplications and `k-1` additions (of `k` numbers). And we have `mn` such values. So the total computation is `O(mnk)`. If you consider addition and multiplication as 1 operation, we're looking at `~2mnk` ops. Please excuse the abuse of O notation here :).

So initially for single token, to perform full computation (MLP), we had
$$
\begin{aligned}
g      &= X @ W_{gate} &&\in \mathbb{R}^{n \times d_{mlp}} &&\quad | \quad \mathcal{O}(d \cdot d_{mlp}) \\
u      &= X @ W_{up}   &&\in \mathbb{R}^{n \times d_{mlp}} &&\quad | \quad \mathcal{O}(d \cdot d_{mlp}) \\
s      &= \text{SiLU}(g) &&                                  &&\quad | \quad \mathcal{O}(d \cdot d_{mlp}) \\
gu     &= s \odot u    &&\in \mathbb{R}^{n \times d_{mlp}} &&\quad | \quad \mathcal{O}(d \cdot d_{mlp}) \\
\text{out} &= gu @ W_{down} &&\in \mathbb{R}^{n \times d}  &&\quad | \quad \mathcal{O}(d \cdot d_{mlp}) \\
\end{aligned}
$$

$$
\text{total} = 5 \cdot \mathcal{O}(d \cdot d_{mlp})
$$

In the latter case, we have weights of the following shapes,
$$
\begin{aligned}
d_{e} &= \frac{d_{mlp}}{n} \\
W'_{up}   &\in \mathbb{R}^{n \times d \times d_{e}} &&\quad | \quad n \text{ buckets each output } d_{e} \text{ features} \\
W'_{gate} &\in \mathbb{R}^{n \times d \times d_{e}} &&\quad | \quad n \text{ buckets each output } d_{e} \text{ features} \\
W'_{down} &\in \mathbb{R}^{n \times d_{e} \times d} &&\quad | \quad n \text{ buckets each output } d \text{ features} \\
W'_{picker} &\in \mathbb{R}^{d \times n}            &&\quad | \quad \text{specifies the preference towards the } n \text{ buckets}
\end{aligned}
$$
First we compute the preference scores, find the top $k$, and pick the corresponding $k$ buckets to perform the operations:

$$
\begin{aligned}
\text{Scores} &= X @ W_{picker} &&\in \mathbb{R}^{n} &&\quad | \quad \mathcal{O}(d \cdot n) \\
\text{TopK} &= \text{top}_k(\text{Scores}) &&\in \mathbb{R}^{k} &&\quad | \quad \mathcal{O}(n \log k) \\
\\
g &= X @ W'_{gate}[\text{TopK}] &&\in \mathbb{R}^{n \times k \cdot d_e} &&\quad | \quad \mathcal{O}(d \cdot k \cdot d_e) \\
u &= X @ W'_{up}[\text{TopK}]   &&\in \mathbb{R}^{n \times k \cdot d_e} &&\quad | \quad \mathcal{O}(d \cdot k \cdot d_e) \\
s &= \text{SiLU}(g)             &&                                      &&\quad | \quad \mathcal{O}(d \cdot k \cdot d_e) \\
gu &= s \odot u                 &&\in \mathbb{R}^{n \times k \cdot d_e} &&\quad | \quad \mathcal{O}(d \cdot k \cdot d_e) \\
\text{out} &= gu @ W'_{down}[\text{TopK}] &&\in \mathbb{R}^{n \times d} &&\quad | \quad \mathcal{O}(d \cdot k \cdot d_e) \\
\end{aligned}
$$

$$
\text{total} = 5 \cdot \mathcal{O}(d \cdot k \cdot d_e) + \mathcal{O}(d \cdot n) + \mathcal{O}(n \log k)
$$

comparing the two, we have 
$$
\begin{aligned}
\text{Ops}_{\text{MLP}} &= 5 \cdot \mathcal{O}(d \cdot d_{mlp}) \\
\text{Ops}_{\text{MoE}} &= 5 \cdot \mathcal{O}(d \cdot k \cdot d_e) + \mathcal{O}(d \cdot n) + \mathcal{O}(n \log k) \\
&= 5 \cdot \mathcal{O}\left(d \cdot k \cdot \frac{d_{mlp}}{n}\right) + \mathcal{O}(d \cdot n) + \mathcal{O}(n \log k) \\
&\approx \mathcal{O} \left( d \cdot \left( 5 \cdot d_{mlp} \cdot \frac{k}{n} + n \right) \right) \quad (\text{ignoring } n \log k)
\end{aligned}
$$

I'd leave it to you to compare these for a typical MoE config like [Qwen3-30B-A3B](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507/blob/main/config.json) or [GPT OSS 20B](https://huggingface.co/unsloth/gpt-oss-20b/blob/main/config.json) and verify for yourself that this makes sense mathematically.
