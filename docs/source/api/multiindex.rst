===================
MultiIndices
===================

A multiindex is simply a length :math:`D` vector of nonnegative integers :math:`\mathbf{p}=[p_1,p_2,\dots,p_D]`.
Multiindices are commonly employed for defining multivariate polynomial expansions and other function parameterizations.
In these cases, sets of multiindices define the form of the expansion.   

**Example: Multivariate Polynomial**

A multivariate polynomial :math:`\Phi_{\mathbf{p}}(\mathbf{x}) : \mathbb{R}^D\rightarrow R` can be defined as the 
product of :math:`D` univariate polynomials.   Using monomials for example,

.. math::

    \Phi_{\mathbf{p}}(\mathbf{x}) = \prod_{i=1}^D x_i^{p_i}

A multivariate polynomial expansion can then be written succinctly as 

.. math::

    f(\mathbf{x}) = \sum_{\mathbf{p}\in\mathcal{S}} c_{\mathbf{p}} \Phi_{\mathbf{p}}(\mathbf{x})

where :math:`\mathcal{S}` is a set of multiindices and :math:`c_{\mathbf{p}}` are scalar coefficients.


**Example: Wavelets**

Multivariate polynomials are constructed from a tensor product of one-dimensional functions and each
one-dimensional function depends on a single integer: the degree of the one-dimensional polynomial.   This is a common 
way to define multivariate functions from indexed families of one-dimensional basis functions.  In a general 
setting, however, the one-dimensional family does not need to be index by a single integer.  Families of 
one-dimensional functions indexed with multiple integers can also be "tensorized" into multivariate functions.
Wavelets are a prime example of this.

A one dimensional wavelet basis contains functions of the form

.. math::

    \psi_{j,k}(x) = 2^{j/2}\psi(2^jx -k)

where :math:`j` and :math:`k` are integers and :math:`\psi :\mathbb{R}\rightarrow \mathbb{R}` is an orthogonal wavelet. 
Unlike polynomials, two integers are required to index the one-dimensional family. Nevertheless, a multivariate wavelet 
basis can be defined through the tensor product of components in this family:

.. math::

    \Psi_{\mathbf{p}}(\mathbf{x}) = \prod_{i=1}^{D/2} \psi_{p_{2i},p_{2i+1}}(x_i)

where :math:`\Psi_{\mathbf{p}} : \mathbb{R}^{D/2}\rightarrow \mathbb{R}` is a multivariate wavelet basis function in 
:math:`D/2` dimensions and :math:`\mathbf{p}` is a multiindex with :math:`D` components. 

C++ Objects
-------------------

.. toctree::

   multiindices/multiindex
   multiindices/multiindexset
   multiindices/fixedmultiindexset
   multiindices/multiindexlimiter


Definitions
-------------------

.. topic:: Limiting Set

    In general, any length :math:`D` vector in :math:`\mathbb{N}^D\` is a multiindex.  In many adaptive approaches, however,
    it is useful to only consider multiindices in some subset :math:`\mathcal{G}\subseteq \mathbb{N}^D`.   We call this subset 
    the limiting set or, more loosely, the "limiter".   In MParT, limiting sets are defined by functors that accept a 
    MultiIndex and return `true` if the multiindex is in :math:`\mathcal{G}` and `false` otherwise.  Some predefined limiting 
    sets are defined in the MPart::MultiIndexLimiter namespace, but custom functors can also be employed.

.. topic:: Neighbors

    Let :math:`\mathbf{j}=[j_1,j_2,\dots,j_D]` be a :math:`D`-dimensional multiindex in the set :math:`\mathcal{S}` 
    of multiindices.  In 
    
    For polynomials, the of :math:`\mbox{j}` are the multiindices that only differ from :math:`\mbox{j}` in one component,
    and in that component, the difference is -1. For example, \f$[j_1-1, j_2,\dots,j_D]\f$ and \f$[j_1, j_2-1,\dots,j_D]\f$ are
    backwards neighbors of \f$\mbox{j}\f$, but \f$[j_1-1,
    j_2-1,\dots,j_D]\f$ and \f$[j_1, j_2-2,\dots,j_D]\f$ are not.
    
.. topic:: Downward Closed

    A multiindex set \f$\mathcal{S}\f$ is downward closed if for any multiindex \f$\mathbf{j}\in\mathcal{S}\f$, all 
    backward neighbors of \f$\mathbf{j}\f$ are also in \f$\mathcal{S}\f$.

.. topic:: Margin

    For a multindex set \f$\mathcal{S}\f\subseteq\mathcal{G}$ and limiting set \f$\mathcal{G}\f$, 
    the margin of \f$\mathcal{S}\f$ is the set \f$\mathcal{M}\f\in \mathcal{G}\\ \mathcal{S}\f$ containing 
    indices that have at least one backward neighbor in \f$\mathcal{S}\f$.

.. topic:: Reduced Margin
    The reduced margin \f$\mathcal{M}_r\subset\mathcal{M}\f$ is a subset of the margin \$\mathcal{M}\f$ containing
    indices that could be added to the set \f$\mathcal{S}\f$ while preserving the downward closed property of \f$\mathcal{S}\f$.
    Multiindices in the reduced margin are often called admissible.

.. topic:: Active/Inactive Indices

    The MParT::MultiIndexSet class stores both the set \f$\mathcal{S}\f$ and the margin \f$\mathcal{M}\f$.   In MParT,
    multiindices in \f$\mathcal{S}\f$ are sometimes called active, while multiindices in \f$\mathcal{M}\f$ are called inactive.
    Only active indices are included in the linear indexing.  Inactive
    indices are hidden (i.e. not even considered) in all the public members
    of this class.  Inactive multiindices are used behind the scenes to check admissability, to define the margin, and 
    are added to the \f$\mathcal{S}\f$ during adaptation.


