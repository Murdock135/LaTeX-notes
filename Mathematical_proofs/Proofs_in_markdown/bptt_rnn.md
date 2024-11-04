# Backpropagation through time for RNN
Author: Qazi Zarif Ul Islam

![Unfolded Recurrent Neural Net](rnn.png)  
*Credit: [murat_bptt]*

In this document, we shall derive the gradient of the loss of RNN w.r.t. the *hidden weights* \(W_{hh}\). In truth, we shall derive the gradient aka differentiate the **loss at a particular time instant wrt *the concatenated weight matrix $[W_{xh}\;W_{hh}]^T$.** This concatenation does not make any difference from a computational perspective. [d2l_bptt, murat_bptt].

The empirical loss of the neural network is,

$$
\hat{L} = \frac{1}{T} \sum_{t=1}^{T} \ell(y_t, d_t) = \frac{1}{T} (l_1 + l_2 + ... + l_t ... + l_T)
$$

For an RNN, the system is defined by,

$$
h_{t} = f (X_{t}, h_{t-1}) = \phi_{h}(W_{xh} \cdot X_{t} + W_{hh}\cdot h_{t-1} +b_{h}) \\
\hat{o}_{t} = f_{o}(h_{t}) = \phi_{o}(W_{hy}\cdot h_{t} + b_{y})
$$

Let \(w_h = [W_{xh}\;W_{hh}]^T\).

Thus,

$$
\frac{\partial \hat{L}}{\partial w_h} = \frac{1}{T} \left(\frac{\partial l_1}{\partial w_h} + \frac{\partial l_2}{\partial w_h} + ... + \frac{\partial l_t}{\partial w_h} ... + \frac{\partial l_T}{\partial w_h}\right)
$$

Now the loss at a particular time instant \(t\) follows the chain rule of derivatives.

$$
\frac{\partial l_t}{\partial w_h} = \frac{\partial l_t}{\partial o_t}\frac{\partial o_t}{\partial h_t}\frac{\partial h_t}{\partial w_h}
$$

But \(h_t\) is a function of \(h_{t-1}\) and \(w_h\) as well (besides \(X_t\)).
Furthermore, \(h_{t-1}\) is again a function of \(w_h\).
Thus, by the multivariable chain rule,

If \(h_t = f_1(h_{t-1}, w_h)\), \(h_{t-1}= g_1(h_{t-2}, w_h)\), \(w_h = h_1(w_h)\)

$$
\frac{\partial h_t}{\partial w_h} = \frac{\partial f_1}{\partial w_h} + \frac{\partial f_1}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial w_h}
$$

Similarly, for \(h_{t-1}, h_{t-2}\), and so on, leading to a summation of multiplications for the final form of the derivative.

### References
- [d2l_bptt]
- [murat_bptt]
