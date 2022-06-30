.. _mathematics:

Mathematical Background
=========================

Tensor Product Expansions 
--------------------------

For a point :math:`\mathbf{x}\in\mathbb{R}^d` and coefficients :math:`\mathbf{w}`, we consider expansions of the form 

.. math::

    f(\mathbf{x}; \mathbf{w}) = \sum_{\alpha\in \mathcal{A}} w_\alpha \Phi_\alpha(\mathbf{x}),

where :math:`\alpha\in\mathbb{N}^d` is a multi-index, :math:`\mathcal{A}` is a multiindex set, and :math:`\Phi_{\mathbf{\alpha}}` is a
multivariate function defined as a tensor product of one-dimensional functions :math:`\phi_{\alpha_i}\colon  \mathbb{R}\rightarrow \mathbb{R}`
through

.. math::

    \Phi_\mathbf{\alpha}(\mathbf{x}) = \prod_{\alpha_i \in \mathbf{\alpha}} \phi_{\alpha_i}(x_i).


Monotone Parameterizations
--------------------------

We represent monotone functions as the smooth transformation of an unconstrained function :math:`f\colon\mathbb{R}^{d} \rightarrow \mathbb{R}`. Let :math:`g\colon\mathbb{R}\rightarrow \mathbb{R}_{>0}` be a strictly positive function, such as the softplus :math:`g(x) = \log(1 + \exp(x))`. Then, the d-th map component :math:`T_{d}` is given by

.. math::
    :label: cont_map 

    T_d(\mathbf{x}_{1:d}; \mathbf{w}) = f(x_1,\ldots, x_{d-1},0; \mathbf{w}) + \int_0^{x_d} g( \partial_d f(x_1,\ldots, x_{d-1},t; \mathbf{w}) ) dt.

Other choices for the :math:`g` include the squared and exponential functions. These choices, however, have implications for the identifiability of the coefficients. If :math:`g` is bijective, then we can recover :math:`f` from :math:`T_d` as 

.. math::
    f(\mathbf{x}_{1:d}; \mathbf{w}) = T_d(x_1,\ldots, x_{d-1},0; \mathbf{w}) + \int_0^{x_d} g^{-1}( \partial_d T_d(x_1,\ldots, x_{d-1},t; \mathbf{w}) ) dt.

Using the representation for monotone functions with a bijective :math:`g`, we can approximate :math:`T_d` by finding :math:`f`.

Numerical Integration
^^^^^^^^^^^^^^^^^^^^^^^^

Computationally, we approximate the integral in the definition of :math:`T_d(\mathbf{x}_{1:d}; \mathbf{w})` using a quadrature rule with :math:`N` points :math:`\{t^{(1)}, \ldots, t^{(N)}\}` and corresponding weights :math:`\{c^{(1)}, \ldots, c^{(N)}\}` designed to approximate integrals over :math:`[0,1]`.  Note that these points and weights will often be chosen adaptively.    The quadrature rule yields an approximation of the map component in :eq:`cont_map` with the form

.. math::
    :label: discr_map 

    \tilde{T}_d(\mathbf{x}_{1:d}; \mathbf{w}) = f(x_1,\ldots, x_{d-1},0; \mathbf{w}) + x_d \sum_{i=1}^N c^{(i)} g( \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) ),

where the :math:`x_d` term outside the summation comes from a change of integration domains from :math:`[0,1]` to :math:`[0,x_d]`. 

.. _diag_deriv_section:

Diagonal Derivatives
^^^^^^^^^^^^^^^^^^^^^^^^^^

We will often require derivatives of :math:`T_d` with respect to an input :math:`x_i` or the parameters :math:`\mathbf{w}`.  When computing these derivatives however, we have a choice of whether to differentiate the continuous map form in :eq:`cont_map` or the discretized map in :eq:`discr_map`.  This is similar to the "discretize-then-optimize" or "optimize-then-discretize" choice in PDE-constrained optimization.  When the quadrature rule is accurate, there might not be a large practical difference in these approaches.  For approximate rules however, using the continuous derivative may cause issues during optimization because the derivative will not be consistent with the discreteized map: a finite difference approximation will not converge to the continuous derivative.   In these cases, it is preferrable to differentiate the discrete map in :eq:`discr_map`.   

The derivative :math:`\partial T_d / \partial x_d` is particularly important when using the monotone function :math:`T_d` in a measure transformation.   The continuous version of this derivative is simply 

.. math::
    :label: cont_deriv 

    \frac{\partial T_d}{\partial x_d}(\mathbf{x}_{1:d}; \mathbf{w}) = g(\, \partial_d f(\mathbf{x}_{1:d}; \mathbf{w})\, ).

The discrete derivative on the other hand is more complicated: 

.. math::
    :label: discr_deriv 

    \frac{\partial \tilde{T}_d}{\partial x_d}(\mathbf{x}_{1:d}; \mathbf{w}) &= \frac{\partial}{\partial x_d} \left[x_d \sum_{i=1}^N c^{(i)} g( \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) )\right]\\
    & = \sum_{i=1}^N c^{(i)} g( \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) ) \\
    &+ x_d \sum_{i=1}^N c^{(i)} t^{(i)} \partial g( \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) ) \partial^2_{dd}f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) .


Coefficient Derivatives 
^^^^^^^^^^^^^^^^^^^^^^^^
In addition to computing :math:`\partial T_d/\partial d`, we will also need the gradient of the monotone function :math:`T_d` with respect to the parameters :math:`\mathbf{w}`, denoted by :math:`\nabla_{\mathbf{w}}T_d`. 

.. math::
    :label: coeff_deriv 

    \nabla_{\mathbf{w}} T_d(\mathbf{x}_{1:d}; \mathbf{w}) &= \nabla_{\mathbf{w}} f(x_1,\ldots, x_{d-1},0; \mathbf{w})\\
     &+ \int_0^{x_d} \partial g( \partial_d f(x_1,\ldots, x_{d-1},t; \mathbf{w}) ) \nabla_{\mathbf{w}}\left[\partial_d f(x_1,\ldots, x_{d-1},t; \mathbf{w})\right] dt \\
     &\approx \nabla_{\mathbf{w}} f(x_1,\ldots, x_{d-1},0; \mathbf{w})\\
    & + x_d \sum_{i=1}^N c^{(i)} \partial g( \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) ) \nabla_{\mathbf{w}}\left[\partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w})\right]


If is also possible to compute the gradient of the diagonal derivative :math:`\nabla_{\mathbf{w}}\left( \partial T_d/\partial d\right)` with respect to the parameters, but like before, there is a question of whether the derivative of the exact map or the derivative of the quadrature-based approximate map should be used.    In the case of the exact map, the mixed coefficient gradient has the simple form

.. math::

    \nabla_{\mathbf{w}}\left[ \frac{\partial T_d}{\partial d}\right] & = \nabla_{\mathbf{w}}\left[ g(\, \partial_d f(\mathbf{x}_{1:d}; \mathbf{w})\, ) \right] \\
    & = \partial g(\, \partial_d f(\mathbf{x}_{1:d}; \mathbf{w})\, )  \nabla_{\mathbf{w}}\left[\partial_d f(\mathbf{x}_{1:d}; \mathbf{w})\right].


The gradient of the discrete derivative is more expansive and takes the form

.. math::

    \nabla_{\mathbf{w}}\left[ \frac{\partial \tilde{T}_d}{\partial d}\right] &= \sum_{i=1}^N c^{(i)} \nabla_{\mathbf{w}}\left[g( \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) ) \right] \\
    & + x_d \sum_{i=1}^N c^{(i)} t^{(i)} \nabla_{\mathbf{w}}\left[\partial g( \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) ) \partial^2_{dd}f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) \right] \\
    &= \sum_{i=1}^N c^{(i)} \partial g( \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w})) \nabla_{\mathbf{w}}\left[ \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) \right] \\
    &+ x_d \sum_{i=1}^N c^{(i)} t^{(i)} \partial^2 g( \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) ) \partial^2_{dd}f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) \nabla_{\mathbf{w}}\left[ \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) \right]  \\
    &  + x_d \sum_{i=1}^N c^{(i)} t^{(i)} \partial g( \partial_d f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w}) )  \nabla_{\mathbf{w}}\left[\partial^2_{dd}f(x_1,\ldots, x_{d-1},x_d t^{(i)}; \mathbf{w})\right].


Triangular Transport Maps
--------------------------

Let :math:`\pi` and :math:`\eta` be two densities on :math:`\mathbb{R}^d`. In measure transport, our goal is to find a multivariate transformation :math:`T` that pushes forward :math:`\eta` to :math:`\pi`, meaning that if :math:`\mathbf{X} \sim \eta`, then :math:`T(\mathbf{X}) \sim \pi`. Given such a map, we can easily generate samples from :math:`\eta` by pushing samples :math:`\mathbf{x}^i \sim \eta` through the map :math:`T(\mathbf{x}^i) \sim \pi`. Furthermore, we can express the push-forward density of a diffeomorphic map by :math:`T_{\sharp}\eta(\mathbf{x}) := \eta(T^{-1}(\mathbf{x}))|\nabla T^{-1}(\mathbf{x})|`.

While there are infinitely many transformations that couple densities, if :math:`\pi` is absolutely continuous with respect to :math:`\eta`, there exists a unique lower triangular and monotone function :math:`T\colon \mathbb{R}^d \rightarrow \mathbb{R}^d` that pushes forward :math:`\pi` to :math:`\eta` of the form

.. math::
    T(\mathbf{x}) = \begin{bmatrix} T_1(x_1) \\ T_2(x_1,x_2) \\ \vdots \\ T_d(x_1,\dots,x_d) \end{bmatrix}.

