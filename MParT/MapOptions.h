#ifndef MPART_MAPOPTIONS_H
#define MPART_MAPOPTIONS_H

#include <limits>

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

        /** Linearization bounds for the 1d basis function. The basis function is linearized outside [lb,ub] */
        double basisLB = -std::numeric_limits<double>::infinity();
        double basisUB =  std::numeric_limits<double>::infinity();


        /** The type of positive bijector used inside the monotonicity-guaranteeing integral
            formulation.
        */
        PosFuncTypes posFuncType = PosFuncTypes::SoftPlus;

        /** The type of quadrature rule to use. */
        QuadTypes quadType = QuadTypes::AdaptiveSimpson;


        /** The absolute tolerance used by adaptive quadrature rules like AdaptiveSimpson and
            AdaptiveClenshawCurtis.
        */
        double quadAbsTol = 1e-6;

        /** The relative tolerance used by adaptive quadrature rules like AdaptiveSimpson
            and AdaptiveClenshawCurtis.
        */
        double quadRelTol = 1e-6;

        /** The maximum number of subdivisions used in the adaptive quadrature rules. */
        unsigned int quadMaxSub = 30;

        /** The minimum number of subdivisions used in the adaptive quadrature rules. */
        unsigned int quadMinSub = 0;


        /** The number of quadrature points used in fixed rules like the Clenshaw Curtis rule.
            Also defines the base level used in the adaptive Clenshaw-Curtis rule.
        */
        unsigned int quadPts = 5;

        /** Specifies whether the derivative of the integral defining a monotone component
            is used or if the derivative of the quadrature-based approximation of the integral
            is used.  See :ref:`diag_deriv_section` for more details.  If true, the integral
            is differentiated directly.  If false, the numerical approximation to the integral
            is differentiated.
        */
        bool contDeriv = true;

        /** If orthogonal polynomial basis functions should be normalized. */
        bool basisNorm = true;

        #if defined(MPART_HAS_CEREAL)
        template<class Archive>
        void serialize(Archive &archive)
        {
            archive( basisType, basisLB, basisUB, posFuncType, quadType, quadAbsTol, quadRelTol, quadMaxSub, quadMinSub, quadPts, contDeriv, basisNorm );
        }
        #endif // MPART_HAS_CEREAL

        bool operator==(MapOptions opts2) const {
            bool ret = true;
            ret &= (basisType   == opts2.basisType);
            ret &= (basisLB     == opts2.basisLB);
            ret &= (basisUB     == opts2.basisUB);
            ret &= (posFuncType == opts2.posFuncType);
            ret &= (quadType    == opts2.quadType);
            ret &= (quadAbsTol  == opts2.quadAbsTol);
            ret &= (quadRelTol  == opts2.quadRelTol);
            ret &= (quadMaxSub  == opts2.quadMaxSub);
            ret &= (quadMinSub  == opts2.quadMinSub);
            ret &= (quadPts     == opts2.quadPts);
            ret &= (contDeriv   == opts2.contDeriv);
            ret &= (basisNorm   == opts2.basisNorm);
            return ret;
        }
    };
};


#endif