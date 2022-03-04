================================
Cached Parameterization Concept
================================
This concept provides the interface expected by the :code:`MonotoneComponent` class for the definition of the generally non-monotone function :math:`f(x) : \mathbb{R}^d \rightarrow \mathbb{R}`.  This concept defines the :code:`ExpansionType` template argument in the :code:`MonotoneComponent` definition.

The :code:`MonotoneComponent` class will evaluate :math:`f(x)` at many points :math:`x^{(k)}` that differ only in the :math:`d^{th}` component.  That is, for any :math:`k_1` and :math:`k_2`, :math:`x_j^{(k_1)} = x_j^{(k_2)}` for :math:`j<d`.   In many cases, it is possible to take advantage of this fact to precompute some quantities related to :math:`x_{1:d-1}` that we know will not change as the :code:`MonotoneComponent` class varies :math:`x_d`.   

To enable this, :code:`ExpansionType` needs to be able to interact with a preallocated block of memory called the cache below.  The cache will be allocated outside the :code:`ExpansionType` class, and then passed to :code:`ExpansionType::FillCache1` to do any necessary precomputations.    

When the value of the last component :math:`x_d` is altered, the :code:`ExpansionType::FillCache2` function will then be called, before calling :code:`ExpansionType::Evaluate` to actually evaluate :math:`f(x)`.   The :code:`ExpansionType::Evaluate` function only has access to the memory in the cache, so its important to perform all necessary computations in either the :code:`ExpansionType::FillCache1` or :code:`ExpansionType::FillCache2` functions.    

The details of these functions, as well as all the functions that :code:`MonotoneComponent` expects :code:`ExpansionType` to have, are provided in the `Specific Requirements`_ section below. 

Potential Usage 
-----------------

Below is an example of how a class implementing the cached parameterization concept could be used to evaluate :math:`f(x;c)` at a randomly chosen point :math:`x` and randomly chosen coefficients :math:`c`.

.. code:: c++ 

    ExpansionType expansion;
    // ... Define the expansion

    // Evaluate the 
    std::vector<double> cache( expansion.CacheSize() );

    Eigen::VectorXd coeffs = Eigen::VectorXd::Random( expansion.NumCoeffs() );

    unsigned int dim = expansion.InputSize();
    Eigen::VectorXd evalPt = Eigen::VectorXd::Random( dim );

    expansion.FillCache1(&cache[0], evalPt, DerivativeFlags::None);
    expansion.FillCache2(&cache[0], evalPt, evalPt(dim-1), DerivativeFlags::None);

    double f = expansion.Evaluate(&cache[0], coeffs);

    // Now perturb x_d and re-evaluate 
    double xd = evalPt(dim-1) + 1e-2;
    expansion.FillCache2(&cache[0], evalPt, xd, DerivativeFlags::None);

    double f2 = expansion.Evaluate(&cache[0], coeffs);


Specific Requirements
----------------------

.. panels::
    :container: container-lg pb-3
    :column: col-lg-12 p-2 border-0

    Cache Size 
    ^^^^^^^^^^^^^^
    .. code:: c++

        unsigned int CacheSize() const;

    Returns the number of doubles required in the cache memory allocation.
    ---

    Number of Coefficients 
    ^^^^^^^^^^^^^^
    .. code:: c++

        unsigned int NumCoeffs() const;

    Returns the number of coefficients in the parameterization.
    ---

    Dimension
    ^^^^^^^^^^^^^^
    .. code:: c++
        
        unsigned int InputSize() const;
    
    Returns the dimension (i.e., number of components) in the input vector  :math:`x`

    ---

    Fill Cache 1
    ^^^^^^^^^^^^^^
    .. code:: c++ 

        template<typename PointType>
        void FillCache1(double*                           cache, 
                        PointType                  const& pt, 
                        DerivativeFlags::DerivativeType   derivType) const;

    Precomputes parts of the cache using all but the last component of the point, i.e., using only :math:`x_1,x_2,\ldots,x_{d-1}`, not :math:`x_d`. Can be an empty function if no precomputations are possible without using the :math:`d^{th}` component of the input :math:`x_d`.

    ---

    Fill Cache 2
    ^^^^^^^^^^^^^^
    .. code:: c++ 

        template<typename PointType>
        void FillCache2(double*                         cache, 
                        PointType const&                pt,
                        double                          xd,
                        DerivativeFlags::DerivativeType derivType) const;

    This function is called just before calling :code:`Evaluate`.   After this call, the :code:`cache` variable and a vector of coefficients should be sufficient for evaluating the function :math:`f(x)` or its derivatives.   If :code:`derivType==None`, then the cache will not be used for evaluating any derivatives.  If :code:`derivType==Diagonal`, then the cache may be used for evaluating the diagonal derivative :math:`\partial_d f`.   If :code:`derivType==Diagonal2`, then the cache may be used for evaluating :math:`\partial_d^2 f` or the mixed derivatives :math:`\nabla_c [\partial_d f]`.

    Note that the :math:`d^{th}` component of the :code:`pt` argument should not be used.  The value of the :math:`x_d` should be taken from the :code:`xd` argument only.  This is a subtle choice that makes it possible for :code:`MonotoneComponent` to integrate over :math:`x_d` without copying the entire point pt.

    ---

    Evaluate :math:`f(x)`
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: c++

        template<typename CoeffVecType>
        double Evaluate(const double*       cache,
                        CoeffVecType const& coeffs) const;

    Uses the cache (after calls to :code:`FillCache1` and :code:`FillCache2`) and a vector of coefficients to evaluate :math:`f(x)`.  The :code:`CoeffVecType` type is guaranteed to have a paranthetical access operator `double operator()(unsigned int)` and be at least the length returned by the :code:`NumCoeffs()` function.
                                  
    ---

    Derivative :math:`\partial_d f`
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: c++

        template<typename CoeffVecType>
        double DiagonalDerivative(const double*       cache, 
                                  CoeffVecType const& coeffs, 
                                  unsigned int        derivOrder) const;

    Uses the cache (after calls to :code:`FillCache1` and :code:`FillCache2`) and a vector of coefficients to evaluate :math:`\partial_d f(x)` or :math:`\partial^2_{dd} f(x)`.  The :code:`CoeffVecType` type is guaranteed to have a paranthetical access operator `double operator()(unsigned int)` and be at least the length returned by the :code:`NumCoeffs()` function.   :code:`derivOrder` will be either :math:`1` or :math:`2` to specify if :math:`\partial_d f(x)` or :math:`\partial^2_{dd} f(x)` should be computed.

    ---

    Coefficient Gradient :math:`\nabla_c f`
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: c++ 

        template<typename CoeffVecType, typename GradVecType>
        double CoeffDerivative(const double*       cache, 
                               CoeffVecType const& coeffs,
                               GradVecType&        grad) const;

    Computes the gradient :math:`\nabla_c f` at a single point.  The gradient is stored in the preallocated :code:`grad` vector.  The :code:`double` returned by this function should return the value of :math:`f(x)`.   Both the :code:`CoeffVecType` and :code:`GradVecType` types will provide the parantheses operator :code:`()` for accessing and setting values.   The :code:`grad` will be preallocated, but might not be initialized; values of grad should therefore be set even if they are :math:`0`.

    ---

    Mixed Gradient :math:`\nabla_c \partial_d f`
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: c++ 

        template<typename CoeffVecType, typename GradVecType>
        double MixedDerivative(const double*       cache,
                               CoeffVecType const& coeffs,
                               unsigned int        derivOrder,
                               GradVecType&        grad) const;

    Computes the mixed gradient :math:`\nabla_c[ \partial_d f]` or :math:`\nabla_c[ \partial^2_{dd} f]` at a single point.  The gradient should be stored in the preallocated :code:`grad` vector.  The :code:`double` returned by this function should contain :math:`\partial_d` if :code:`derivOrder==1` or :math:`\partial^2_{dd}` if :code:`derivOrder==2`.   Both the :code:`CoeffVecType` and :code:`GradVecType` types will provide the parantheses operator :code:`()` for accessing and setting values.   The :code:`grad` will be preallocated, but might not be initialized; values of grad should therefore be set even if they are :math:`0`.


Implementations
-----------------
The following classes currently implement this concept.

.. toctree::
   :maxdepth: 1

   ../multivariateexpansion
   ../tensorproductfunction