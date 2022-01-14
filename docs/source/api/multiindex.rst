===================
MultiIndices
===================

Background
-------------------

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
   multiindices/multiindexneighborhood


Definitions
-------------------

.. topic:: Limiting Set

    In general, any length :math:`D` vector in :math:`\mathbb{N}^D\` is a multiindex.  In many adaptive approaches, however,
    it is useful to only consider multiindices in some subset :math:`\mathcal{G}\subseteq \mathbb{N}^D`.   We call this subset 
    the limiting set or, more loosely, the "limiter".   In MParT, limiting sets are defined by functors that accept a 
    MultiIndex and return `true` if the multiindex is in :math:`\mathcal{G}` and `false` otherwise.  Some predefined limiting 
    sets are defined in the MultiIndexLimiter namespace, but custom functors can also be employed.

.. topic:: Neighbors

    In most parameterization applications, there is additional structure in the multiindex set that can be represented 
    through edges on a graph.   Each multiindex can be treated as a node in directed graph.  Outgoing edges connect to 
    other multiindices, which we call "forward neighbors", that represent higher order basis functions.   Incoming edges
    come from multiindices, which we call "backward neighbors", that represent lower order basis functions.   
    
    In the multivariate polynomial example above, the backward neighbors of a multiindex :math:`\mathbf{j}` are the multiindices
    that only differ from :math:`\mbox{j}` in one component, and in that component, the difference is -1.  More precisely, 
    the set of backward neighbors is 
    
    .. math::
    
        \mathcal{B}_{\mathbf{j}} = \{\mathbf{i} : \|\mathbf{i}-\mathbf{j}\|_1=1 \text{ and } \mathbf{i}_k \leq \mathbf{j}_k \text{ for all } k \}.
    
    The set of forward neighbors is similarly defined as 

    .. math::

        \mathcal{F}_{\mathbf{j}} = \{\mathbf{i} : \|\mathbf{i}-\mathbf{j}\|_1=1 \text{ and } \mathbf{i}_k \geq \mathbf{j}_k \text{ for all } k \}

    The neighborhood structure of a multindex set is defined through a child of the abstract MultiIndexNeighborhood class.

.. topic:: Downward Closed

    A multiindex set :math:`\mathcal{S}` is downward closed if for any multiindex :math:`\mathbf{j}\in\mathcal{S}`, all 
    backward neighbors of :math:`\mathbf{j}` are also in :math:`\mathcal{S}`.

.. topic:: Margin

    For a multindex set :math:`\mathcal{S}\subseteq\mathcal{G}`, the margin of :math:`\mathcal{S}` is the set
    :math:`\mathcal{M} \in \mathcal{G}\\ \mathcal{S}` containing indices that have at least one backward neighbor
    in :math:`\mathcal{S}`.

.. topic:: Reduced Margin

    The reduced margin :math:`\mathcal{M}_r\subset\mathcal{M}` is a subset of the margin :math:`\mathcal{M}` containing
    indices that could be added to the set :math:`\mathcal{S}` while preserving the downward closed property of :math:`\mathcal{S}`.
    Multiindices in the reduced margin are often called admissible.

.. topic:: Active/Inactive Indices

    The MultiIndexSet class stores both the set :math:`\mathcal{S}` and the margin :math:`\mathcal{M}`.   In MParT,
    multiindices in :math:`\mathcal{S}` are sometimes called active, while multiindices in :math:`\mathcal{M}` are called inactive.
    Only active indices are included in the linear indexing.  Inactive
    indices are hidden (i.e. not even considered) in all the public members
    of this class.  Inactive multiindices are used behind the scenes to check admissability, to define the margin, and 
    are added to the :math:`\mathcal{S}` during adaptation.

.. topic:: Frontier

    The Frontier is similar but contains multiindices in :math:`\mathcal{S}` that have at least one forward neighbor in 
    the margin :math:`\mathcal{M}`.   We use the term "strict frontier" to describe the set of multiindices in :math:`\mathcal{S}` that have all of their forward neighbors in the margin.   
 



