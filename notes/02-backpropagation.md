# 02 · Deriving the backward pass

Cache the inputs and outputs of every layer during the forward pass. Write
$\Delta^{(l)} = \partial L/\partial Z^{(l)}$ for the derivative at layer $l$.

For mean squared error, the output delta is

$$\Delta^{(L)}=\frac{2}{Bu}(A^{(L)}-Y)\odot f'(Z^{(L)}).$$

For sigmoid combined with binary cross-entropy, the sigmoid derivative cancels
the corresponding denominator in the loss derivative:

$$\Delta^{(L)}=\frac{A^{(L)}-Y}{Bu}.$$

The fused expression also avoids division by a saturated probability. It is used
only for this specific loss/activation pair; `compile` rejects BCE with a
non-sigmoid output.

## Layer gradients

For $Z^{(l)}=A^{(l-1)}W^{(l)}+b^{(l)}$,

$$\frac{\partial L}{\partial W^{(l)}}=(A^{(l-1)})^T\Delta^{(l)},\qquad
\frac{\partial L}{\partial b^{(l)}}=\sum_n\Delta_n^{(l)}.$$

To continue backwards,

$$\Delta^{(l-1)}=\left(\Delta^{(l)}(W^{(l)})^T\right)\odot f'(Z^{(l-1)}).$$

The factor $1/(Bu)$ is already in the output delta: do not divide again in each
layer. Bias-free layers produce zero bias gradients. `gradients(x, y)` exposes
these matrices without updating the model.

## Check the derivative independently

For any scalar parameter $\theta$, central finite differences give

$$g_{\mathrm{numeric}}=\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}.$$

Compare this with the analytical derivative. Python tests perturb every weight
and bias in a multilayer fixture across activations and both losses. C++ checks
hidden-layer weights and biases with a larger epsilon for float precision.
Cross-language tests additionally compare inference and multi-step optimization.

## SGD and Adam

SGD applies $\theta_t=\theta_{t-1}-\eta g_t$ to each mini-batch gradient.
Adam keeps two exponentially weighted moments:

$$m_t=0.9m_{t-1}+0.1g_t,\qquad v_t=0.999v_{t-1}+0.001g_t^2.$$

After bias correction,

$$\hat m_t=\frac{m_t}{1-0.9^t},\quad \hat v_t=\frac{v_t}{1-0.999^t},\quad
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+10^{-8}}.$$

The step counter advances once per mini-batch. Moments persist between `fit`
calls, allowing the laboratory to train in short steps. Calling `compile` or
loading a model resets optimizer state. Model files are parameter snapshots,
not exact training-resumption checkpoints.

Continue with [experiment design](03-experiments.md).
