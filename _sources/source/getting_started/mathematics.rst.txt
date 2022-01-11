.. _mathematics:

Mathematical Background
=========================


Tensor Product Expansions 
--------------------------

For a point $\mathbf{x}\in\mathbb{R}^d$ and coefficients $\mathbf{w}$, we consider expansions of the form 

.. math::

    g(\mathbf{x}; \mathbf{w}) = \sum_{\alpha\in \mathcal{A}} w_\alpha \Phi_\alpha(\mathbf{x}),

where $\alpha\in\mathbb{N}^d$ is a multiindex, $\mathcal{A}$ is a multiindex set, and $\Phi_{\mathbf{\alpha}}$ is a
multivariate function defined as a tensor product of one-dimensional functions $\phi_{\alpha_i} : \mathbb{R}\rightarrow \mathbb{R}$
through

.. math::

    \Phi_\mathbf{\alpha}(\mathbf{x}) = \prod_{\alpha_i \in \mathbf{\alpha}} \phi_{\alpha_i}(x_i).


Monotone Parameterizations
--------------------------

.. math::
    T_d(\mathbf{x}; \mathbf{w}) = g(x_1,\ldots, x_{d-1},0; \mathbf{w}) + \int_0^{x_d} p( g(x_1,\ldots, x_{d-1},t; \mathbf{w}) ) dt
