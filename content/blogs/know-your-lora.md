---
title: "Know your LoRA"
date: 2024-06-07T03:06:01+00:00
draft: false
author: "Datta"
tags:
  - LoRA
  - Fine tuning
  - LLM
image: /images/lora.jpg
description: ""
toc: true
mathjax: true
---

## What is LoRA

LoRA has been a tremendous tool in the world of fine tuning, especially parameter efficient fine tuning. It is an easy way to fine tune your models with very little memory requirements. 
LoRA was first introduced in [this paper](https://arxiv.org/abs/2106.09685) by Hu et al. The premise of LoRA is, upon fine tuning, the change in weights of the matrices are of low rank in comparison with the original matrix
To exploit this, LoRA adds an adapter which we can train while having the initial model weights frozen. 
$$ W' = W + \Delta W  = W + AB \space\space where \space\space A=\mathbb{R}^{(m,r)}, \space\space B=\mathbb{R}^{(r,n)}, \space\space W=\mathbb{R}^{(m,n)}$$
Here W are the initial weights and ΔW is the change in weights upon fine tuning. The advantage with LoRA unlike other PEFT techniques is that LoRA weights can be merged back into the initial model and hence there will not be any performance loss at inference. Also, because this is just an adapter, one can dynamically switch between adapters and having no adapter aka using base model.
Such versatility and flexibility propelled LoRA to become the most used PEFT technique and the best part is, this is model agnostic. So any model that has linear layers, can use this. It has been very famous in both Image generation and NLP worlds off late.

Now comes the question. If we add another weight to existing weight matrix, wouldn't it put the model off? Yes, adding any random stuff does impact the model quality. But to ensure that at initialisation model doesn't suffer from such issues, we initialise matrices `A` and `B` such that the product `ΔW = AB = 0`.

But how do you do that? Initialising both to zero is a viable option but would inhibit the model from learning. So the original paper proposes to initialise A with [kaiming uniform](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_) (just uniform initialisation with differnt range parameter). So problem solved right? We now have a non zero A and a zero B such that `AB = 0`. Well technically yes and this has been working for long. So why change it huh?

Well I wasn't really satisified with this. I thought, why not try out some different initialisations? 
