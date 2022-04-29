#ifndef MPART_CONDITIONALMAPBASE_H
#define MPART_CONDITIONALMAPBASE_H

#include <Kokkos_Core.hpp>

namespace mpart {

    /**
     * @brief Provides an abstract base class for conditional transport maps where the input dimension might be larger than output dimension.
    
     @details 
      This class provides an interface for functions \f$T:\mathbb{R}^{N+M}\rightarrow \mathbb{R}^M\f$, where $N\geq 0$. Let 
      \f$x_1\in \mathbb{R}^N\f$ denote the first block of inputs and \f$x_2\in\mathbb{R}^M\f$ denote the second block of inputs.
      Let $r\in\mathbb{R}^M\f$ denote the map output, \f$r=T(x_2; x_1)\f$.  The conditional maps implemented by children of this 
      class guarantee that for fixed \f$x_1\f$, the conditional mapping from \f$x_1 \rightarrow r\f$ is invertible and the 
      Jacobian matrix with respect to \f$x_2\f$, \f$\nabla_{x_2} T\f$\, is positive definite.
     */
    class ConditionalMapBase {
    
    public:

        /**
         @brief Construct a new Conditional Map Base object
          
         @param inDim The dimension \f$N\f$ of the input to this map. 
         @param outDim The dimension \f$M\f$ of the output from this map.
         */
        ConditionalMapBase(unsigned int inDim, unsigned int outDim) : inputDim(inDim), outputDim(outDim){};

        virtual ~ConditionalMapBase() = default;

        /** Returns a view containing the coefficients for this conditional map.  This function returns a reference 
            and can therefore be used to to update the coefficients or even set them to be a subview of a larger view.
            When the values of the larger view are updated, the subview stored by this class will also be updated. This 
            is particularly convenient when simultaneously optimizing the coefficients over many conditional maps because 
            each map can just use a slice into the larger vector of all coefficients that is updated by the optimizer.
        */
        virtual Kokkos::View<double*, Kokkos::HostSpace>& Coeffs(){return this->savedCoeffs;};

        /** Const version of the Coeffs() function. */
        virtual Kokkos::View<const double*, Kokkos::HostSpace> Coeffs() const{return this->savedCoeffs;};


        virtual Kokkos::View<double**, Kokkos::HostSpace> Evaluate(Kokkos::View<double**, Kokkos::HostSpace> const& pts);


        virtual void Evaluate(Kokkos::View<double**, Kokkos::HostSpace> const& pts,
                              Kokkos::View<double**, Kokkos::HostSpace>      & output) = 0;



        /** Returns the value of \f$x_2\f$ given \f$x_1\f$ and \f$r\f$.   Note that the \f$x1\f$ view may contain more 
            than \f$N\f$ rows, but only the first \f$N\f$ will be used in this function. 
        */
        virtual Kokkos::View<double**, Kokkos::HostSpace> Inverse(Kokkos::View<double**, Kokkos::HostSpace> const& x1, 
                                                                  Kokkos::View<double**, Kokkos::HostSpace> const& r);

        virtual void Inverse(Kokkos::View<double**, Kokkos::HostSpace> const& x1, 
                             Kokkos::View<double**, Kokkos::HostSpace> const& r,
                             Kokkos::View<double**, Kokkos::HostSpace>      & output) = 0;


        const unsigned int inputDim; // The total dimension of the input N+M
        const unsigned int outputDim; // The output dimension M

    protected:

        Kokkos::View<double*, Kokkos::HostSpace> savedCoeffs;

    }; // class ConditionalMapBase
}

#endif 