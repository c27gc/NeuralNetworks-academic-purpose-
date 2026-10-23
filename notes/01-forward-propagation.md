# 01 · Forward propagation

A batch is a matrix with one sample per row. A dense layer maps $d$ input
features to $u$ outputs:

$$X \in \mathbb{R}^{B \times d},\quad W \in \mathbb{R}^{d \times u},\quad b \in \mathbb{R}^{u}$$

$$Z = XW + b,\qquad A = f(Z).$$

The bias is broadcast over samples. `weights[i][j]` connects input feature $i$ to
output unit $j$. A network composes these transformations; a layer's input width
must match the previous layer's output width. Both APIs validate this at compile
time in the model lifecycle (`compile`), before training begins.

## Why nonlinearities matter

Composing linear layers still produces an affine function. XOR cannot be
separated by one affine boundary, so a hidden nonlinear layer is necessary.

| Activation | Function | Derivative from its output $a$ |
| :--- | :--- | :--- |
| Linear | $z$ | $1$ |
| Sigmoid | $1/(1+e^{-z})$ | $a(1-a)$ |
| Tanh | $\tanh z$ | $1-a^2$ |
| ReLU | $\max(0,z)$ | $1$ when $a>0$, otherwise $0$ |

The ReLU derivative at zero is chosen as zero. Avoid that nondifferentiable point
when checking derivatives with finite differences.

## Breaking symmetry

Zero initialization gives hidden units identical outputs and identical updates.
Each layer instead draws weights uniformly from

$$W_{ij} \sim U\left(-\sqrt{\frac{6}{d+u}},\sqrt{\frac{6}{d+u}}\right),\qquad b_j=0.$$

This is Xavier uniform initialization. A seed makes experiments repeatable
within one engine. Explicit `set_weights` parameters are preserved by `compile`.

## Two objectives

For $N=Bu$ target values, mean squared error is

$$L_{\mathrm{MSE}}=\frac{1}{N}\sum_{n,j}(A_{nj}-Y_{nj})^2.$$

For binary targets with a sigmoid output, binary cross-entropy is evaluated
directly from logits:

$$L_{\mathrm{BCE}}=\frac{1}{N}\sum_{n,j}\left[\max(Z_{nj},0)-Z_{nj}Y_{nj}+\log(1+e^{-|Z_{nj}|})\right].$$

This softplus expression stays meaningful for a confidently wrong output:
$z=1000,y=0$ has loss approximately $1000$, even when the stored sigmoid output
has rounded to one. Clipping probabilities before taking logarithms would hide
that distinction.

Continue with [the backward pass](02-backpropagation.md).
