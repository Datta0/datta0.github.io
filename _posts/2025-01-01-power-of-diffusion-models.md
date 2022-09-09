---
layout: post
title: 'Power of Diffusion Models'
date: 2022-01-01 11:00 +0800
categories: [Generative AI]
tags: [diffusion-models, jax, clip, dalle-2, midjourney, stable-diffusion]
math: true
enable_d3: true
published: true
---

> The purpose of this post is...

Sources:

- Papers:
	- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
	- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672.pdf) 
	- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)
	- [Generative Modeling by Estimating Gradients of the
Data Distribution](https://arxiv.org/pdf/1907.05600.pdf)
- Posts:
	- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models)
	- [Denoising Diffusion-based Generative Modeling: Foundations and Applications](https://drive.google.com/file/d/1DYHDbt1tSl9oqm3O333biRYzSCOtdtmn/view)
	- [The recent rise of diffusion-based models](https://maciejdomagala.github.io/generative_models/2022/06/06/The-recent-rise-of-diffusion-based-models.html#citation-14)

### Diffusion models

<script src="https://d3js.org/d3.v4.min.js"></script>
<link href="https://fonts.googleapis.com/css?family=Arvo" rel="stylesheet">

<div id="grph_chain" class="svg-container" align="center"></div> 

<script>

function draw_triangle(svg, x, y, rotate=0) {
	var triangleSize = 25;
	var triangle = d3.symbol()
	            .type(d3.symbolTriangle)
	            .size(triangleSize);
	
	svg.append("path")
	   .attr("d", triangle)
	   .attr("stroke", "black")
	   .attr("fill", "gray")
	   .attr("transform",
	   		function(d) { return "translate(" + x + "," + y + ") rotate(" + rotate  + ")"; });
}

function graph_chain() {

var svg = d3.select("#grph_chain")
			  .append("svg")
			  .attr("width", 600)
			  .attr("height", 130);

svg.append('circle')
  .attr('cx', 50)
  .attr('cy', 50)
  .attr('r', 20)
  .attr('stroke', 'black')
  .attr("opacity", 0.85)
  .attr('fill', '#348ABD');
  
svg.append('text')
  .attr('x', 42)
  .attr('y', 55)
  .text("x")
  .style("font-size", "21px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 55)
  .attr('y', 60)
  .text("0")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
 
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 25, y: 110}, {x: 27, y: 100}, {x: 29, y: 90}, {x: 30, y: 90}, 
           {x: 32, y: 90}, {x: 33, y: 100}, {x: 36, y: 90}, {x: 38, y: 80}, 
           {x: 42, y: 70}, {x: 45, y: 100}, {x: 50, y: 100}, {x: 55, y: 110},
           {x: 56, y: 100}, {x: 57, y: 90}, {x: 58, y: 85}, {x: 60, y: 80},
           {x: 65, y: 100}, {x: 68, y: 90}, {x: 72, y: 85}, {x: 75, y: 110},])
   .attr("fill", "#348ABD")
   .attr("opacity", "0.8")
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append('line')
  .attr('x1', 20)
  .attr('y1', 110)
  .attr('x2', 80)
  .attr('y2', 110)
  .style("stroke-width", 1)
  .attr('stroke', 'black');
  
svg.append('text')
  .attr('x', 35)
  .attr('y', 125)
  .text("q(x")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 61)
  .attr('y', 125)
  .text(")")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 56)
  .attr('y', 128)
  .text("0")
  .style("font-size", "8px")
  .attr("font-family", "Arvo");
       
svg.append('line')
  .attr('x1', 70)
  .attr('y1', 50)
  .attr('x2', 110)
  .attr('y2', 50)
  .style("stroke-width", 1)
  .attr('stroke', 'black');
  
draw_triangle(svg, 110, 50, 90);

svg.append('circle')
  .attr('cx', 130)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');
  
svg.append('circle')
  .attr('cx', 150)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');
  
svg.append('circle')
  .attr('cx', 170)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');

svg.append('line')
  .attr('x1', 185)
  .attr('y1', 50)
  .attr('x2', 225)
  .attr('y2', 50)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
draw_triangle(svg, 225, 50, 90);

svg.append('circle')
  .attr('cx', 250)
  .attr('cy', 50)
  .attr('r', 20)
  .attr('stroke', 'black')
  .attr("opacity", 0.75)
  .attr('fill', '#5286A5');
 
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 225, y: 110}, {x: 227, y: 110}, {x: 229, y: 90},
           {x: 230, y: 100}, {x: 232, y: 90}, {x: 233, y: 100},
           {x: 240, y: 95}, {x: 242, y: 75}, {x: 245, y: 80},
   			 {x: 250, y: 85}, {x: 257, y: 90}, {x: 258, y: 85},
   			 {x: 260, y: 88}, {x: 265, y: 100}, {x: 268, y: 90},
   			 {x: 270, y: 109}, {x: 272, y: 85}, {x: 275, y: 110}])
   .attr("fill", "#5286A5")
   .attr("opacity", "0.8")
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append('line')
  .attr('x1', 220)
  .attr('y1', 110)
  .attr('x2', 280)
  .attr('y2', 110)
  .style("stroke-width", 1)
  .attr('stroke', 'black');
  
svg.append('text')
  .attr('x', 240)
  .attr('y', 55)
  .text("x")
  .style("font-size", "21px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 253)
  .attr('y', 60)
  .text("t-1")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 234)
  .attr('y', 125)
  .text("q(x")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 266)
  .attr('y', 125)
  .text(")")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 255)
  .attr('y', 128)
  .text("t-1")
  .style("font-size", "8px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 270)
  .attr('y1', 50)
  .attr('x2', 330)
  .attr('y2', 50)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
draw_triangle(svg, 325, 50, 90);

svg.append('circle')
  .attr('cx', 350)
  .attr('cy', 50)
  .attr('r', 20)
  .attr('stroke', 'black')
  .attr("opacity", 0.7)
  .attr('fill', '#628498');
 
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 325, y: 110}, {x: 327, y: 110}, {x: 329, y: 100},
           {x: 330, y: 100}, {x: 332, y: 90}, {x: 333, y: 100},
           {x: 340, y: 95}, {x: 342, y: 70},
   			 {x: 350, y: 85}, {x: 357, y: 90}, {x: 358, y: 85},
   			 {x: 360, y: 88}, {x: 365, y: 100}, {x: 368, y: 90},
   			 {x: 370, y: 109}, {x: 375, y: 110}])
   .attr("fill", "#628498")
   .attr("opacity", "0.8")
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append('line')
  .attr('x1', 320)
  .attr('y1', 110)
  .attr('x2', 380)
  .attr('y2', 110)
  .style("stroke-width", 1)
  .attr('stroke', 'black');
  
svg.append('text')
  .attr('x', 342)
  .attr('y', 55)
  .text("x")
  .style("font-size", "21px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 355)
  .attr('y', 60)
  .text("t")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 335)
  .attr('y', 125)
  .text("q(x")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 360)
  .attr('y', 125)
  .text(")")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 356)
  .attr('y', 128)
  .text("t")
  .style("font-size", "8px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 370)
  .attr('y1', 50)
  .attr('x2', 410)
  .attr('y2', 50)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
draw_triangle(svg, 405, 50, 90);

svg.append('circle')
  .attr('cx', 430)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');
  
svg.append('circle')
  .attr('cx', 450)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');
  
svg.append('circle')
  .attr('cx', 470)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');
	  
svg.append('line')
  .attr('x1', 490)
  .attr('y1', 50)
  .attr('x2', 530)
  .attr('y2', 50)
  .style("stroke-width", 1)
  .attr("opacity", 0.95)
  .attr('stroke', 'black');
  
draw_triangle(svg, 525, 50, 90);

svg.append('circle')
  .attr('cx', 550)
  .attr('cy', 50)
  .attr('r', 20)
  .attr('stroke', 'black')
  .attr("opacity", 0.5)
  .attr('fill', '#808080');
 
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: 525, y: 110}, {x: 530, y: 109}, {x: 535, y: 103},
   			 {x: 540, y: 92}, {x: 550, y: 70},  {x: 560, y: 92},
   			 {x: 565, y: 103}, {x: 570, y: 109}, {x: 575, y: 110}])
   .attr("fill", "#808080")
   .attr("opacity", "0.8")
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
svg.append('line')
  .attr('x1', 520)
  .attr('y1', 110)
  .attr('x2', 580)
  .attr('y2', 110)
  .style("stroke-width", 1)
  .attr('stroke', 'black');
  
svg.append('text')
  .attr('x', 542)
  .attr('y', 55)
  .text("x")
  .style("font-size", "21px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 555)
  .attr('y', 60)
  .text("T")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 535)
  .attr('y', 125)
  .text("q(x")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 561)
  .attr('y', 125)
  .text(")")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 556)
  .attr('y', 128)
  .text("T")
  .style("font-size", "8px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 33)
  .attr('y', 15)
  .text("Data")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 529)
  .attr('y', 15)
  .text("Noise")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
}

graph_chain();

</script>

![](.)
*Forward diffusion process. Given a data point sampled from a real data distribution $\mathbf{x}_0 \sim q(x_0)$, we produce noisy latents $\mathbf{x}_1 \rightarrow \cdots \rightarrow \mathbf{x}_T$ by adding small amount of Gaussian noise at each timestep $t$. The latent $\mathbf{x}_t$ gradually loses its recognizable features as the step $t$ becomes larger and eventually with $T \rightarrow \infty$, $\mathbf{x}_T$ is nearly an isotropic Gaussian distribution.*

The step sizes are controlled by a variance schedule $\beta_t \in (0, 1)$:

$$\mathbf{x}_t = \sqrt{1-\beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, \mathbf{I})$$

Conditional distribution for the forward process is

$$q(\mathbf{x}_t \vert  \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} \mathbf{x}_t, \beta_t \mathbf{I}) \quad q(\mathbf{x}_{1:T} \vert  \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$$

Recall that Gaussian distribution has the following property: for $\epsilon_1 \sim \mathcal{N}(0, \sigma^2_1\mathbf{I})$ and $\epsilon_2 \sim \mathcal{N}(0, \sigma^2_2 \mathbf{I})$ we have

$$\epsilon_1 + \epsilon_2 \sim \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)\mathbf{I}).$$

Therefore for each latent $\mathbf{x}_t$ at arbitrary step $t$ we can sample it in a closed form. 

Using the notation $\alpha_t := 1 - \beta_t$ and $\overline\alpha_t := \prod_{s=1}^t \alpha_s$ we get

$$
\begin{aligned}
\mathbf{x}_t & = {\color{#5286A5} {\sqrt{\alpha_t} \mathbf{x}_{t-1}}} + { \sqrt{1-\alpha_t} \epsilon_{t-1}} \\ & = {\color{#5286A5} {\sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{\alpha_t (1-\alpha_{t-1})} \epsilon_{t-2}}} + { \sqrt{1-\alpha_t} \epsilon_{t-1}} \\ & = {\color{#5286A5} {\sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2}}} + \sqrt{1-\alpha_t \alpha_{t-1}} \bar\epsilon_{t-2} \qquad \color{Salmon}{\leftarrow \bar\epsilon_{t-2} \sim \mathcal{N}(0, \mathbf{I})} \\ & = \cdots \\ &= \sqrt{\overline\alpha_t} \mathbf{x}_0 + \sqrt{1-\overline\alpha_t} \epsilon
\end{aligned} $$

and 

$$q(\mathbf{x}_t \vert  \mathbf{x}_{0}) \sim \mathcal{N}\big(\sqrt{\overline\alpha_t}\mathbf{x}_{0}, \sqrt{1-\overline\alpha_t} \mathbf{I}\big).$$

If we were able to reverse diffusion process and sample from $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$, we could recreate samples from a true distribution $q(\mathbf{x}_0)$ with only a Gaussian noise input $\mathbf{x}_T$. 
In general $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ is intractable, since its calculation would require marginalization over the entire data distribution. However, it is worth to note that with $\beta_t$ small enough $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ is also Gaussian.

The core idea of diffusion algorithm is to train a model $p_\theta$ to approximate these conditional probabilities in order to run the reverse diffusion process:

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t}) = \mathcal{N}(\mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t)),$$

where $\mu_\theta(\mathbf{x}_t, t)$  and $\Sigma_\theta(\mathbf{x}_t, t)$ are trainable networks. Although, for simplicity we can decide for 

$$\Sigma_\theta(\mathbf{x}_t, t) = \sigma_t^2 \mathbf{I}.$$

<div id="grph_rvrs_chain" class="svg-container" align="center"></div> 

<script>

function draw_uroboros(svg, x) {
 svg.append("path")
   .attr("stroke", "black")
   .datum([{x: x + 33, y: 35}, {x: x + 20, y: 20}, {x: x, y: 15}, {x: x - 20, y: 20}, {x: x - 33, y: 35}])
   .attr("fill", "none")
   .attr("opacity", "0.8")
	.style("stroke-dasharray", ("4, 4"))
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
draw_triangle(svg, x - 31, 33, 220);
       
svg.append("path")
   .attr("stroke", "black")
   .datum([{x: x + 33, y: 65}, {x: x + 20, y: 80}, {x: x, y: 85}, {x: x - 20, y: 80}, {x: x - 33, y: 65}])
   .attr("fill", "none")
   .attr("opacity", "0.8")
   .attr("d",  d3.line()
       .curve(d3.curveBasis)
       .x(function(d) { return d.x; })
       .y(function(d) { return d.y; }));
       
draw_triangle(svg, x + 31, 67, 40);

}

function graph_reverse_chain() {

var svg = d3.select("#grph_rvrs_chain")
			  .append("svg")
			  .attr("width", 600)
			  .attr("height", 105);

svg.append('circle')
  .attr('cx', 50)
  .attr('cy', 50)
  .attr('r', 20)
  .attr('stroke', 'black')
  .attr("opacity", 0.85)
  .attr('fill', '#348ABD');
  
svg.append('text')
  .attr('x', 42)
  .attr('y', 55)
  .text("x")
  .style("font-size", "21px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 55)
  .attr('y', 60)
  .text("0")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
draw_uroboros(svg, 100);

svg.append('circle')
  .attr('cx', 130)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');
  
svg.append('circle')
  .attr('cx', 150)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');
  
svg.append('circle')
  .attr('cx', 170)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');

draw_uroboros(svg, 200);

svg.append('circle')
  .attr('cx', 250)
  .attr('cy', 50)
  .attr('r', 20)
  .attr('stroke', 'black')
  .attr("opacity", 0.75)
  .attr('fill', '#5286A5');
  
svg.append('text')
  .attr('x', 240)
  .attr('y', 55)
  .text("x")
  .style("font-size", "21px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 253)
  .attr('y', 60)
  .text("t-1")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");

svg.append('circle')
  .attr('cx', 350)
  .attr('cy', 50)
  .attr('r', 20)
  .attr('stroke', 'black')
  .attr("opacity", 0.7)
  .attr('fill', '#628498');
 
draw_uroboros(svg, 300);
       
svg.append('text')
  .attr('x', 268)
  .attr('y', 10)
  .text("p")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 275)
  .attr('y', 15)
  .text("θ")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 282)
  .attr('y', 10)
  .text("(x")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 295)
  .attr('y', 15)
  .text("t-1")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 309)
  .attr('y', 10)
  .text("| x")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 323)
  .attr('y', 15)
  .text("t")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 327)
  .attr('y', 10)
  .text(")")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 342)
  .attr('y', 55)
  .text("x")
  .style("font-size", "21px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 355)
  .attr('y', 60)
  .text("t")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");

svg.append('text')
  .attr('x', 272)
  .attr('y', 97)
  .text("q(x")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 293)
  .attr('y', 102)
  .text("t")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 298)
  .attr('y', 97)
  .text("| x")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 312)
  .attr('y', 102)
  .text("t-1")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 325)
  .attr('y', 97)
  .text(")")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 342)
  .attr('y', 55)
  .text("x")
  .style("font-size", "21px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 355)
  .attr('y', 60)
  .text("t")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
draw_uroboros(svg, 400);
  
svg.append('circle')
  .attr('cx', 430)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');
  
svg.append('circle')
  .attr('cx', 450)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');
  
svg.append('circle')
  .attr('cx', 470)
  .attr('cy', 50)
  .attr('r', 1)
  .attr('stroke', 'black')
  .attr("opacity", 1)
  .attr('fill', 'black');

svg.append('circle')
  .attr('cx', 550)
  .attr('cy', 50)
  .attr('r', 20)
  .attr('stroke', 'black')
  .attr("opacity", 0.5)
  .attr('fill', '#808080');
   
draw_uroboros(svg, 500);
   
svg.append('text')
  .attr('x', 542)
  .attr('y', 55)
  .text("x")
  .style("font-size", "21px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 555)
  .attr('y', 60)
  .text("T")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 33)
  .attr('y', 15)
  .text("Data")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 532)
  .attr('y', 15)
  .text("Prior")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
}

graph_reverse_chain();

</script>

![](.)
*Forward and reverse diffusion processes. Going backwards, we start from isotropic Gaussian noise $p(\mathbf{x}_T) \sim \mathcal{N}(0, \mathbf{I})$ and gradually sample from $p_\theta(\mathbf{x}_{t-1} \vert  \mathbf{x}_t)$ for $t=T, \dots, 1$ until we get a data point from approximated distribution.*

Note that reverse conditional probability is tractable when conditioned on $\mathbf{x}_0$:

$$q(\mathbf{x}_{t-1} \vert  \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}({\color{#5286A5}{\tilde \mu(\mathbf{x}_t, \mathbf{x}_0)}}, {\color{#C19454}{\tilde \beta_t \mathbf{I}}}).$$

Efficient training is therefore possible by minimizing Kullback-Leibler divergence between $p_\theta$ and $q$, or formally, evidence lower bound loss

$$
\begin{aligned}
L_{\operatorname{ELBO}} &= \mathbb{E}_q\bigg[\log\frac{q(\mathbf{x}_{1:T} \vert  \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \bigg]
\\ &= \mathbb{E}_q\bigg[\log\frac{\prod_{t=1}^T q(\mathbf{x}_t|\mathbf{x}_{t-1}) }{p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)} \bigg]
\\ &= \mathbb{E}_q\bigg[\sum_{t=1}^T \log \frac{ q(\mathbf{x}_t|\mathbf{x}_{t-1})} {p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)} -\log p_\theta(\mathbf{x}_T)\bigg]
\\ &= \mathbb{E}_q\bigg[\log \frac{q(\mathbf{x}_1|\mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0}|\mathbf{x}_1)} + \sum_{t=2}^T \log  \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{t}, \mathbf{x}_0) q(\mathbf{x}_t|\mathbf{x}_0)}{q(\mathbf{x}_{t-1}|\mathbf{x}_0)p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)} -\log p_\theta(\mathbf{x}_T)\bigg]
\\ &= \mathbb{E}_q\bigg[\log \frac{q(\mathbf{x}_1|\mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0}|\mathbf{x}_1)} + \sum_{t=2}^T \log  \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{t}, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)} + \log \frac{q(\mathbf{x}_T|\mathbf{x}_0)}{q(\mathbf{x}_1|\mathbf{x}_0)}-\log p_\theta(\mathbf{x}_T)\bigg]
\\ &= \mathbb{E}_q\bigg[-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)  + \sum_{t=2}^T \log  \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{t}, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}+ \log \frac{q(\mathbf{x}_T|\mathbf{x}_0)}{p_\theta(\mathbf{x}_T)}\bigg].
\end{aligned}$$

Labeling each term:

$$\begin{aligned}
L_0 &= \mathbb{E}_q[-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)], & \\
L_{t} &= D_{\operatorname{KL}}\big(q(\mathbf{x}_{t-1} \vert \mathbf{x}_{t}, \mathbf{x}_0) \big|\big| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)\big), &t = 1, \dots T-1, \\
L_T &= D_{\operatorname{KL}}\big(q(\mathbf{x}_T \vert  \mathbf{x}_0) \big|\big| p_\theta(\mathbf{x}_T)\big)\big],
\end{aligned}
$$

we get total objective

$$L_{\operatorname{ELBO}}= \sum_{t=0}^{T} L_t.$$

Last term $L_T$ can be ignored, as $q$ doesn't depend on $\theta$ and $p_\theta(\mathbf{x}_T)$ is isotropic Gaussian. All KL divergences in equation above are comparisons between Gaussians, so they can be calculated with closed form expressions instead of high variance Monte Carlo estimates. One can try to estimate $\color{#5286A5}{\tilde\mu(\mathbf{x}_t, \mathbf{x}_0)}$ directly with

$$ L_t = \mathbb{E}_q \Big[ \frac{1}{2\sigma_t^2}  \|{\color{#5286A5}{\tilde\mu(\mathbf{x}_t, \mathbf{x}_0)}} - \mu_\theta(\mathbf{x}_t, t)  \|^2 \Big] + C,$$

where $C$ is some constant independent of $\theta$. However [Ho et al.](https://arxiv.org/pdf/2006.11239.pdf) propose a different way - train neural network $\epsilon_\theta(\mathbf{x}_t, t)$ to predict the noise.

We can start from reformulation of $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$. First, note that

$$
\begin{aligned}
\log q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0) &\propto - {\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t}} = - {\frac{\mathbf{x}_t^2 - 2 \sqrt{\alpha_t} \mathbf{x}_t{\color{#5286A5}{\mathbf{x}_{t-1}}} + {\alpha_t} {\color{#C19454}{\mathbf{x}_{t-1}^2}}}{\beta_t}},
\\
\log q(\mathbf{x}_{t-1}|\mathbf{x}_0) &\propto -{\frac{(\mathbf{x}_{t-1} - \sqrt{\bar\alpha_{t-1}} \mathbf{x}_{0})^2}{1-\bar\alpha_{t-1}}} = - {\frac{ {\color{#C19454} {\mathbf{x}_{t-1}^2} } - 2\sqrt{\bar\alpha_{t-1}}{\color{#5286A5}{\mathbf{x}_{t-1}}} \mathbf{x}_{0} + \bar\alpha_{t-1}\mathbf{x}_{0}^2}{1-\bar\alpha_{t-1}}}.
\end{aligned}$$

Then, using Bayesian rule we have:

$$\begin{aligned}
\log q(\mathbf{x}_{t-1} \vert  \mathbf{x}_t, \mathbf{x}_0) & = \log q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0) + \log q(\mathbf{x}_{t-1}|\mathbf{x}_0) - \log q(\mathbf{x}_{t}|\mathbf{x}_0)
\\ & \propto {-\color{#C19454}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}) \mathbf{x}_{t-1}^2}} + {\color{#5286A5}{(\frac{2\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\mathbf{x}_0 )\mathbf{x}_{t-1}}} + f(\mathbf{x}_t, \mathbf{x}_0),
\end{aligned}
$$

where $f(\mathbf{x}_t, \mathbf{x}_0)$ is some function independent of $\mathbf{x}_{t-1}$. 

Now following the standard Gaussian density function, the mean and variance can be parameterized as follows (recall that $\alpha_t +\beta_t=1$):

$${\color{#C19454}{\tilde \beta_t}} = \Big(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}\Big)^{-1} = \Big(\frac{\alpha_t-\bar{\alpha}_{t}+\beta_t}{\beta_t (1-\bar{\alpha}_{t-1})}\Big)^{-1} = \beta_t \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_{t}}$$

and 

$$
\begin{aligned}
{\color{#5286A5}{\tilde\mu(\mathbf{x}_t, \mathbf{x}_0)}} &= \Big( \frac{\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\mathbf{x}_0 \Big) \cdot \color{#C19454}{\tilde \beta_t} 
\\ &= \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} \mathbf{x}_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}\mathbf{x}_0.
\end{aligned}
$$

Using representation $\mathbf{x}_0 = \frac{1}{\sqrt{\bar\alpha_t}}(\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\epsilon)$ we get

$$
\begin{aligned} L_t &= \mathbb{E}_q \Big[ \frac{1}{2\sigma_t^2}  \|{\color{#5286A5}{\tilde\mu(\mathbf{x}_t, \mathbf{x}_0)}} - \mu_\theta(\mathbf{x}_t, t)  \|^2 \Big]
\\ &= \mathbb{E}_{\mathbf{x}_0, \epsilon} \Big[ \frac{1}{2\sigma_t^2} \Big \|{\color{#5286A5}{\frac{1}{\sqrt{\bar\alpha_t}}\Big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\Big)}} - \frac{1}{\sqrt{\bar\alpha_t}}\Big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{x}_t, t)\Big)  \Big \|^2 \Big]
\\ &= \mathbb{E}_{\mathbf{x}_0, \epsilon} \Big[ \frac{\beta_t^2}{2\sigma_t^2 \bar\alpha_t (1-\bar\alpha_t)} \Big \|{\color{#5286A5}{\epsilon}} - \epsilon_\theta(\mathbf{x}_t, t)  \Big \|^2 \Big]
\end{aligned}$$

Empirically, [Ho et al.](https://arxiv.org/pdf/2006.11239.pdf) found that training the diffusion model works better with a simplified objective that ignores the weighting term:

$$L_t^{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, \epsilon} \big[  \|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)  \|^2 \big] = \mathbb{E}_{\mathbf{x}_0, \epsilon} \big[  \|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t} \epsilon, t)  \|^2 \big]$$

![Diffusion Model Architecture]({{'/assets/img/diffusion-u-net.png'|relative_url}})
*Diffusion models often use U-Net architectures with ResNet blocks and self-attention layers to represent $\epsilon_\theta(\mathbf{x}_t, t)$. Time features (usually sinusoidal positional embeddings or random Fourier features) are fed to the residual blocks using either simple spatial addition or using adaptive group normalization layers. [Image source](https://drive.google.com/file/d/1DYHDbt1tSl9oqm3O333biRYzSCOtdtmn/view).*


To summarize, our training process:

- Sample $\mathbf{x}_0 \sim q(\mathbf{x}_0)$
- Choose randomly a certain step in diffusion process: $t \sim \mathcal{U}(\lbrace 1,2, \dots T \rbrace)$
- Apply noising: $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t} \epsilon$ with $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
- Take a gradient step on
$$\nabla_\theta \| \epsilon - \epsilon_\theta(\mathbf{x}_t, t) \|^2$$
- Repeat until converge

```python
import jax.numpy as jnp
from jax import grad, jit, vmap, random

batch_size = 32
T = 100
key = random.PRNGKey(42)

# linear schedule
alphas = jnp.linspace(1, 0, T) 
alpha_bars = jnp.cumprod(alphas)

# initial model weights
dummy = sample_batch(key, batch_size)
params = model.init(key, dummy)

@jit
def loss(params, eps, x_t, t):
    return jnp.sum((eps - model.apply(params, x_t, t)) ** 2)

@jit
def apply_noising(a, img, noise):
    return jnp.sqrt(a) * img + jnp.sqrt(1 - a) * noise
	    
def train_on_batch():
    # sample from train data
    x_0 = sample_batch(key, batch_size)
    # choose random steps
    t = random.randint(key, shape=(batch_size,), minval=0, maxval=T)
    # add noise
    eps = random.normal(key, shape=x_0.shape)
    x_t = jit(vmap(apply_noising))(alpha_bars[t], x_0, eps)
    # calculate gradients
    grads = jit(grad(loss))(params, eps, x_t)
    # update parameters with gradients and your favourite optimizer
    ...
```
![](.)
*Diffusion model training in JAX*


Inference process consists of the following steps:

- Sample $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
- For $t = T, \dots, 1$ 

$$\mathbf{x}_{t-1} = \mu_\theta(\mathbf{x}_t, t) + \sigma_t \epsilon,$$

  where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ and

$$\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\bar\alpha_t}}\Big(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{x}_t, t) \Big).$$ 

- Return $\mathbf{x}_0$

```python
def get_x_tm1(params, x_t, t):
    eps = model.apply(params, x_t, t)
    mu_t = x_t - eps * (1 - alphas[t]) / jnp.sqrt(1 - alpha_bars[t])
    mu_t /= jnp.sqrt(alpha_bars[t])
    return mu_t + sigma_t * random.normal(key, shape=x_0.shape)

def sample_batch():
    x_t = random.normal(key, shape=x_0.shape)
    for t in range(T, 1, -1):
        x_t = get_x_tm1(params, x_t, t)
    return x_t
```
![](.)
*Diffusion model sampling in JAX*

### Score based generative modelling

Diffusion model is an example of discrete Markov chain. We can extend it to continuous stochastic process. Let's define **Wiener process (Brownian motion)** $\mathbf{w}_t$ - a random process, such that it starts with $0$, its samples are continuous paths and all of its increments are independent and normally distributed:

$$\frac{\mathbf{w}(t) - \mathbf{w}(s)}{\sqrt{t - s}} \sim \mathcal{N}(0, \mathbf{I}), \quad t > s.$$ 

Let also

$$\mathbf{x}\big(\frac{t}{T}\big) := \mathbf{x}_t \ \text{ and }  \ \beta\big(\frac{t}{T}\big) := \beta_t \cdot T,$$

then

$$\mathbf{x}\big(\frac{t + 1}{T}\big) = \sqrt{1-\frac{\beta(t/T)}{T}} \mathbf{x}(t/T) + \sqrt{\beta(t/T)} \Big( \mathbf{w}\big(\frac{t+1}{T}\big)-\mathbf{w}\big(\frac{t}{T}\big) \Big).$$

Rewriting equation above with $t:=\frac{t}{T}$ and $\Delta t = \frac{1}{T}$, we get

$$
\begin{aligned}
\mathbf{x}(t+\Delta t) &= \sqrt{1-\beta(t)\Delta t} \mathbf{x}(t) + \sqrt{\beta(t)} (\mathbf{w}(t + \Delta t)-\mathbf{w}(t)) \\
& \approx \Big(1 - \frac{\beta(t) \Delta t}{2} \Big) \mathbf{x}(t) + \sqrt{\beta(t)}(\mathbf{w}(t + \Delta t)-\mathbf{w}(t)). & \color{Salmon}{\leftarrow \text{Taylor expansion}}
\end{aligned}$$

With $\Delta t \rightarrow 0$ this converges to **stochastic differential equation (SDE)**:

$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}dt + \sqrt{\beta(t)} d\mathbf{w}.$$

The equation of type 

$$d\mathbf{x} = f(\mathbf{x}, t)dt + g(t)d\mathbf{w}$$

has a unique strong solution as long as the coefficients are globally Lipschitz in both state and time ([Øksendal (2003)](http://www.stat.ucla.edu/~ywu/research/documents/StochasticDifferentialEquations.pdf)). We hereafter denote by $q_t(\mathbf{x})$ probability density of $\mathbf{x}(t)$. 

By starting from samples of $\mathbf{x}_T \sim q_T(\mathbf{x})$ and reversing the process, we can obtain samples $\mathbf{x}_0 \sim q_0(\mathbf{x})$. It was proved in [Anderson (1982)](https://reader.elsevier.com/reader/sd/pii/0304414982900515?token=87C349DB9BEE275FFC8CA1B9E94F4EB84D25343F2FBCF9886B08402A7CE1C334B1ECBC2A7DB2805CD00A2BD720F9FBFF&originRegion=eu-west-1&originCreation=20220906054001) that the reverse of a diffusion process is also a diffusion process, running backwards in time and given by the reverse-time SDE:

$$
\begin{aligned}
d\mathbf{x} = [f(\mathbf{x}, t) - g(t)^2 &\underbrace{\nabla_{\mathbf{x}} \log q_t(\mathbf{x})}]dt + g(t) d\bar{\mathbf{w}}, &\\
&\color{Salmon}{\text{Score Function}} \\
\end{aligned}$$

where $\bar{\mathbf{w}}$ is a standard Wiener process when time flows backwards from $T$ to $0$. Once the score of each marginal distribution is known for all $t$, we can derive the reverse diffusion process from and simulate it to sample from $q_0$. In our case with

$$f(\mathbf{x},t) = -\frac{1}{2}\beta(t)\mathbf{x}(t) \ \text{ and } \ g(t) = \sqrt{\beta(t)}$$

we have reverse diffusion process

$$d\mathbf{x} = \big[-\frac{1}{2}\beta(t)\mathbf{x} - \beta(t) \nabla_{\mathbf{x}} \log q_t(\mathbf{x})\big] dt + \sqrt{\beta(t)}d\bar{\mathbf{w}}.$$
    

<div id="cntns_chain" class="svg-container" align="center"></div> 

<script>

d3.select("#cntns_chain")
  .style("position", "relative");
  
function continuous_chain() {

var svg = d3.select("#cntns_chain")
			  .append("svg")
			  .attr("width", 600)
			  .attr("height", 85);

svg.append('circle')
  .attr('cx', 50)
  .attr('cy', 50)
  .attr('r', 20)
  .attr('stroke', 'black')
  .attr("opacity", 0.85)
  .attr('fill', '#348ABD');
  
svg.append('text')
  .attr('x', 42)
  .attr('y', 55)
  .text("x")
  .style("font-size", "21px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 55)
  .attr('y', 60)
  .text("0")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
svg.append('line')
  .attr('x1', 70)
  .attr('y1', 40)
  .attr('x2', 530)
  .attr('y2', 40)
  .style("stroke-width", 1)
  .attr('stroke', 'black');
  
draw_triangle(svg, 525, 40, 90);
  
svg.append('line')
  .attr('x1', 70)
  .attr('y1', 60)
  .attr('x2', 530)
  .attr('y2', 60)
  .style("stroke-width", 1)
  .attr('stroke', 'black');
  
draw_triangle(svg, 75, 60, 270);
  
svg.append('circle')
  .attr('cx', 550)
  .attr('cy', 50)
  .attr('r', 20)
  .attr('stroke', 'black')
  .attr("opacity", 0.5)
  .attr('fill', '#808080');
  
svg.append('text')
  .attr('x', 542)
  .attr('y', 55)
  .text("x")
  .style("font-size", "21px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 555)
  .attr('y', 60)
  .text("T")
  .style("font-size", "11px")
  .attr("font-family", "Arvo");
  
d3.select("#cntns_chain")
  .append("span")
  .text("\\(d\\mathbf{x} = -\\frac{1}{2}\\beta(t)\\mathbf{x} dt + \\sqrt{\\beta(t)}d\\mathbf{w} \\)")
  .style("font-size", "14px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", "285px")
  .style("top", "10px");
  
d3.select("#cntns_chain")
  .append("span")
  .text("\\(d\\mathbf{x} = \\big[-\\frac{1}{2}\\beta(t)\\mathbf{x} - \\beta(t) \\nabla_{\\mathbf{x}} \\log q_t(\\mathbf{x})\\big] dt + \\sqrt{\\beta(t)}d\\bar{\\mathbf{w}} \\)")
  .style("font-size", "14px")
  .style("font-weight", "700")
  .attr("font-family", "Arvo")
  .style("position", "absolute")
  .style("left", "215px")
  .style("top", "70px");
  
svg.append('text')
  .attr('x', 33)
  .attr('y', 15)
  .text("Data")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
svg.append('text')
  .attr('x', 532)
  .attr('y', 15)
  .text("Prior")
  .style("font-size", "14px")
  .attr("font-family", "Arvo");
  
}

continuous_chain();

</script>
![](.)
*Forward and reverse generative diffusion SDEs.*

In order to estimate $\nabla_{\mathbf{x}} \log q_t(\mathbf{x})$ we can train a time-dependent score-based model $\mathbf{s}_\theta(\mathbf{x}, t)$, such that

$$\mathbf{s}_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log q_t(\mathbf{x}).$$

The marginal diffused density $q_t(\mathbf{x}(t))$ is not tractable, however, 

$$q_t(\mathbf{x}(t) \vert \mathbf{x}(0)) \sim \mathcal{N}(\sqrt{\bar{\alpha}(t)} \mathbf{x}(0), (1 - \bar{\alpha}(t)) \mathbf{I})$$

with $\bar{\alpha}(t) = e^{\int_0^t \beta(s) ds}$. Therefore we can minimize

$$\mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}(0, t)} \mathbb{E}_{\mathbf{x}(0) \sim q_0(\mathbf{x})} \mathbb{E}_{\mathbf{x}(t) \sim q_t(\mathbf{x}(t) \vert \mathbf{x}(0))}[ \| \mathbf{s}_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)} \log q_t(\mathbf{x}(t) \vert \mathbf{x}(0)) \|^2 ],$$

and after expectations, $\mathbf{s}_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log q_t(\mathbf{x}).$

//TODO: sampling and training Jax code

#### Connection to diffusion model

Given a Gaussian distribution

$$\mathbf{x}(t) = \sqrt{\bar{\alpha}(t)} \mathbf{x}(0) + \sqrt{1 - \bar{\alpha}(t)} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}),$$

we can write the derivative of the logarithm of its density function as

$$ 
\begin{aligned}
\nabla_{\mathbf{x}(t)} \log q_t(\mathbf{x}(t) \vert \mathbf{x}(0)) &= -\nabla_{\mathbf{x}(t)} \frac{(\mathbf{x}(t) - \sqrt{\bar{\alpha}(t)} \mathbf{x}(0))^2}{2 (1 - \bar{\alpha}(t))} \\
&= -\frac{\mathbf{x}(t) - \sqrt{\bar{\alpha}(t)} \mathbf{x}(0)}{1 - \bar{\alpha}(t)} \\
&= \frac{\epsilon}{\sqrt{1 - \bar{\alpha}(t)}}.
\end{aligned}
$$

Also,

$$\mathbf{s}_\theta(\mathbf{x}, t) = -\frac{\epsilon_\theta(\mathbf{x}, t)}{\sqrt{1 - \bar{\alpha}(t)}}.$$

 
### Guided diffusion

Once the model $\epsilon_\theta(\mathbf{x}_t, t)$ is trained, we can use it to run the isotropic Gaussian distribution $\mathbf{x}_T$ back to $\mathbf{x}_0$ and generate limitless image variations. Now there is the question: how can we guide the class-conditional model $\epsilon_\theta(\mathbf{x}_t,t \vert y)$ to generate specific images by feeding additional information about class $y$ during the training process?

If we have a differentiable discriminative model $f_\phi(y \vert \mathbf{x}_t)$, trained to classify noisy images $\mathbf{x}_t$, we can use its gradients to guide the diffusion sampling process toward the conditioning information $y$  by altering the noise prediction. 

We can write the score function for the joint distribution $q(\mathbf{x}, y)$ as following,

$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t, y) &= \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log q(y \vert \mathbf{x}_t) \\
& \approx  -\frac{\epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}} + \nabla_{\mathbf{x}_t} \log f_\phi (y \vert \mathbf{x}_t).
\end{aligned}
$$

At each step of denoising, the classifier checks whether the image is denoised in the right direction and contributes its own gradient of loss function into the overall loss of diffusion model. To control the strength of the classifier guidance, we can add a weight $\omega$, called the **guidance scale**, and here is our new classifier-guided model $\epsilon_\theta$:

$$\epsilon_\theta(\mathbf{x}_t, t \vert y) = \epsilon_\theta(\mathbf{x}_t, t) + \omega \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi (y \vert \mathbf{x}_t).$$

If we revert the gradient and the logarithm operations, we observe that we are raising the conditional part of the distribution to a power:

$$q(\mathbf{x}_t, y) \propto q(\mathbf{x}) \cdot q(y \vert \mathbf{x})^\omega.$$

This corresponds to tuning the temperature of distribution, taking $\omega > 1$ we sharpen the distribution, which usually leads to better individual sample quality, but less sample diversity. 

Unfortunately, there are a few problems, which make this approach impractical...

//TODO: draw diagrams on guidance

#### Classifier-free guidance

A downside of classifier guidance is that it requires an additional classifier model and thus complicates the training pipeline. You can't plug in a standard pre-trained classifier, because this model has to be trained on noisy data $\mathbf{x}_t$. [Ho & Salimans](https://openreview.net/pdf?id=qw8AKxfYbI) proposed an alternative method, **a classifier-free guidance**, which doesn't require training a separate classifier. Instead, one trains a conditional diffusion model $p_\theta(\mathbf{x} \vert y)$ parameterized through $\epsilon(\mathbf{x}_t, t, \vert y)$ with conditioning dropout: 10-20% of the time, the conditioning information $y$ is removed. This way model knows how to generate images unconditionally as well, i.e.

$$\epsilon(\mathbf{x}_t, t) = \epsilon(\mathbf{x}_t, t \vert \emptyset).$$

### DALL·E 2

#### CLIP

![CLIP]({{'/assets/img/clip-arch.png'|relative_url}})
*CLIP architecture*

![](.)
*JAX-like pseudocode for the core of an implementation of CLIP:*

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(jnp.dot(I_f, W_i), axis=1)
T_e = l2_normalize(jnp.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = jnp.dot(I_e, T_e.T) * jnp.exp(t)

# symmetric loss function
labels = jnp.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t) / 2
```

![Bear in mind]({{'/assets/img/bear-in-mind.jpg'|relative_url}})
*Bear in mind, digital art. Image source: DALL·E 2 by OpenAI Instagram account.*

![Outpainting]({{'/assets/img/outpainting.jpeg'|relative_url}})

### Disco diffusion

### Midjourney

### Stable diffusion