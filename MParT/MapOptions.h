#ifndef MPART_MAPOPTIONS_H
#define MPART_MAPOPTIONS_H

namespace mpart{

    enum class BasisTypes
    {
        ProbabilistHermite,
        PhysicistHermite,
        HermiteFunctions
    };

    enum class PosFuncTypes
    {
        Exp,
        SoftPlus
    };

    enum class QuadTypes
    {
        ClenshawCurtis,
        AdaptiveSimpson,
        AdaptiveClenshawCurtis
    };

    /** @brief struct to hold options used by the MapFactory methods to construct monotone components and triangular maps.
        @details 

        ### Usage:
        Note that all options are optional.

        @code{.cpp}
        MapOptions opts;
        opts.basisType = BasisTypes::ProbabilistHermite;
        opts.posFuncType = PosFuncTypes::SoftPlus;
        opts.quadType = QuadTypes::AdaptiveSimpson;
        opts.quadAbsTol = 1e-6;
        opts.quadRelTol = 1e-6;
        opts.quadMaxSub = 20;
        opts.quadPts = 5;
        @endcode
    */
    struct MapOptions 
    {   
        /** The type of 1d basis function used to define the multivariate expansion. */
        BasisTypes basisType = BasisTypes::ProbabilistHermite;

        /** The type of positive bijector used inside the monotonicity-guaranteeing integral formulation. */
        PosFuncTypes posFuncType = PosFuncTypes::SoftPlus;

        /** The type of quadrature rule to use. */
        QuadTypes quadType = QuadTypes::AdaptiveSimpson;

        /** The absolute tolerance used by adaptive quadrature rules like AdaptiveSimpson and AdaptiveClenshawCurtis. */
        double quadAbsTol = 1e-6;

        /** The relative tolerance used by adaptive quadrature rules like AdaptiveSimpson and AdaptiveClenshawCurtis. */
        double quadRelTol = 1e-6;

        /** The maximum number of subdivisions used in the adaptive quadrature rules. */
        unsigned int quadMaxSub = 30;

        /** The number of quadrature points used in fixed rules like the Clenshaw Curtis rule.  Also defines the base level used in the adaptive Clenshaw-Curtis rule. */
        unsigned int quadPts = 5;
    }; 
};


#endif 