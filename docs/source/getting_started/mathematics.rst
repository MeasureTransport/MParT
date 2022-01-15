.. _mathematics:

Mathematical Background
=========================


Tensor Product Expansions 
--------------------------

For a point :math:`\mathbf{x}\in\mathbb{R}^d` and coefficients :math:`\\mathbf{w}`, we consider expansions of the form 

.. math::

    f(\mathbf{x}; \mathbf{w}) = \sum_{\alpha\in \mathcal{A}} w_\alpha \Phi_\alpha(\mathbf{x}),

where :math:`\\alpha\\in\\mathbb{N}^d` is a multi-index, :math:`\\mathcal{A}` is a multiindex set, and :math:`\\Phi_{\\mathbf{\\alpha}}` is a
multivariate function defined as a tensor product of one-dimensional functions :math:`\\phi_{\alpha_i} : \mathbb{R}\rightarrow \mathbb{R}`
through

.. math::

    \Phi_\mathbf{\alpha}(\mathbf{x}) = \prod_{\alpha_i \in \mathbf{\alpha}} \phi_{\alpha_i}(x_i).


Monotone Parameterizations
--------------------------

We represent monotone functions as the smooth transformation of an unconstrained function :math:`f\\colon\\mathbb{R}^{d} \\rightarrow \\mathbb{R}`. Let :math:`g\\colon\\mathbb{R}\\rightarrow \\mathbb{R}_{>0}` be a strictly positive function, such as the softplus :math:`g(x) = \\log(1 + \\exp(x))`. Then, the d-th map component :math:`T_{d}` is given by

.. math::
    T_d(\mathbf{x}_{1:d}; \mathbf{w}) = f(x_1,\ldots, x_{d-1},0; \mathbf{w}) + \int_0^{x_d} g( \partial_d f(x_1,\ldots, x_{d-1},t; \mathbf{w}) ) dt

Other choices for the :math:`g` include the squared and exponential functions. These choices, however, have implications for the identifiability of the coefficients. If :math:`g` is bijective, then we can recover :math:`f` from :math:`T_d` as. 

.. math::
    f(\mathbf{x}_{1:d}; \mathbf{w}) = T_d(x_1,\ldots, x_{d-1},0; \mathbf{w}) + \int_0^{x_d} g^{-1}( \partial_d T_d(x_1,\ldots, x_{d-1},t; \mathbf{w}) ) dt

Using the representation for monotone functions with a bijective :math:`g`, we can approximate :math:`T_d` by finding :math:`f`.


Triangular Transport Maps
--------------------------

Let :math:`\\pi` and :math:`\\eta` be two densities on :math:`\\mathbb{R}^d`. In measure transport, our goal is to find a multivariate transformation :math:`T` that pushes forward :math:`\\eta` to :math:`\\pi`, meaning that if :math:`\\mathbf{X} \\sim \\eta`, then :math:`T(\\mathbf{X}) \\sim \\pi`. Given such a map, we can easily generate samples from :math:`\\eta` by pushing samples :math:`\\mathbf{x}^i \\sim \\eta` through the map :math:`T(\\mathbf{x}^i) \\sim \\pi`. Furthermore, we can express the push-forward density of a diffeomorphic map by :math:`T_{\sharp}\\eta(\mathbf{x}) \\coloneqq \\eta(T^{-1}(\\mathbf{x}))|\\nabla T^{-1}(\\mathbf{x})|`.

While there are infinintely many transformations that couple densities, if :math:`\pi` is absolutely continuous with respect to :math:`\eta`, there exists a unique lower triangular and monotone function :math:`T\colon \\mathbf{R}^d \\rightarrow \\mathbf{R}^d`that pushes forward :math:`\\pi` to :math:`\\eta` of the form

.. math::
    T(\mathbf{x}) = \begin{bmatrix} T_1(x_1) \\ T_2(x_1,x_2) \\ \vdots \\ T_d(x_1,\dots,x_d) \end{bmatrix}.


