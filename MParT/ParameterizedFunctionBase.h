#ifndef MPART_ParameterizedFunctionBase_H
#define MPART_ParameterizedFunctionBase_H

#include "MParT/Utilities/EigenTypes.h"
#include "MParT/Utilities/ArrayConversions.h"

#include <Eigen/Core>

namespace mpart {

    
    template<typename MemorySpace>
    class ParameterizedFunctionBase {

    public:

        /**
         @brief Construct a new parameterized function object

         @param inDim The dimension \f$N\f$ of the input to this map.
         @param outDim The dimension \f$M\f$ of the output from this map.
         @param nCoeffs The number of coefficients in the parameterization.
         */
        ParameterizedFunctionBase(unsigned int inDim, unsigned int outDim, unsigned int nCoeffs) : inputDim(inDim), outputDim(outDim), numCoeffs(nCoeffs){};

        virtual ~ParameterizedFunctionBase() = default;

        /** Returns a view containing the coefficients for this conditional map.  This function returns a reference
            and can therefore be used to to update the coefficients or even set them to be a subview of a larger view.
            When the values of the larger view are updated, the subview stored by this class will also be updated. This
            is particularly convenient when simultaneously optimizing the coefficients over many conditional maps because
            each map can just use a slice into the larger vector of all coefficients that is updated by the optimizer.
        */
        virtual Kokkos::View<double*, MemorySpace>& Coeffs(){return this->savedCoeffs;};

        /** @briefs Set the internally stored view of coefficients.
            @detail Performs a shallow copy of the input coefficients to the internally stored coefficients.
            If values in the view passed to this function are changed, the values will also change in the
            internally stored view.

            @param[in] coeffs A view to save internally.
        */
        virtual void SetCoeffs(Kokkos::View<double*, MemorySpace> coeffs);
        virtual void SetCoeffs(Eigen::Ref<Eigen::VectorXd> coeffs);

        /** Returns an eigen map wrapping around the coefficient vector, which is stored in a Kokkos::View.  Updating the
            components of this map should also update the view.
        */
        virtual Eigen::Map<Eigen::VectorXd> CoeffMap();

        /** Const version of the Coeffs() function. */
        virtual Kokkos::View<const double*, MemorySpace> Coeffs() const{return this->savedCoeffs;};

        

        virtual Kokkos::View<double**, MemorySpace> Evaluate(Kokkos::View<const double**, MemorySpace> const& pts);

        virtual Eigen::RowMatrixXd Evaluate(Eigen::Ref<const Eigen::RowMatrixXd> const& pts);

        virtual void EvaluateImpl(Kokkos::View<const double**, MemorySpace> const& pts,
                                  Kokkos::View<double**, MemorySpace>            & output) = 0;


        /** @brief Computes the gradient of the map output with respect to the map coefficients.
        @details Consider a map \f$T(x; w) : \mathbb{R}^N \rightarrow \mathbb{R}^M\f$ parameterized by coefficients \f$w\in\mathbb{R}^K\f$.
                 This function computes 
                 \f[
                    g_i = s_i^T\nabla_w T(x_i; w),
                 \f] 
                 where \f$\nabla_w T(x_i; w)\in\mathbb{R}^{M\times K}\f$ denotes the 
                 Jacobian of the map  \f$T(x_i; w)\f$ with respect to the coefficients at the single point \f$x_i\f$ and \f$s_i\in \mathbb{R}^M\f$ 
                 is a vector of sensitivities.  Often the sensititivities \f$s_i\f$ represent the gradient of some objective function with 
                 respect to the map output, i.e. \f$s_i = \nabla_{y_i} c(y_i)\f$, where \f$c:\mathbb{R}^M\rightarrow \mathbb{R}\f$ is a scalar-valued 
                 objective function and \f$y_i=T(x_i;w)\f$ is the output of the map.   In this setting, the vector \f$g_i\f$ computed by this 
                 function represents \f$g_i = \nabla_{w} c(T(x_i; w))\f$; and this function essentially computes a single step in the chain rule. 

        @param pts A collection of points \f$x_i\f$ where we want to compute the Jacobian.  Each column contains a single point.
        @param sens A collection of sensitivity vectors \f$s_i\f$ for each point.   Each column is a single \f$s_i\f$ vector and 
                    this view should therefore have the same number of columns as `pts`.  It should also have \f$M\f$ rows.   
        @return A collection of vectors \f$g_i\f$.  Will have the same number of columns as pts with \f$K\f$ rows.
        */
        virtual Kokkos::View<double**, MemorySpace> CoeffGrad(Kokkos::View<const double**, MemorySpace> const& pts, 
                                                              Kokkos::View<const double**, MemorySpace> const& sens);

        virtual Eigen::RowMatrixXd CoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts,
                                             Eigen::Ref<const Eigen::RowMatrixXd> const& sens);

        virtual void CoeffGradImpl(Kokkos::View<const double**, MemorySpace> const& pts,  
                                   Kokkos::View<const double**, MemorySpace> const& sens,
                                   Kokkos::View<double**, MemorySpace> &output) = 0;

        
        const unsigned int inputDim; /// The total dimension of the input N+M
        const unsigned int outputDim; /// The output dimension M
        const unsigned int numCoeffs; /// The number of coefficients used to parameterize this map.

    protected:
        
        /** Throws an exception if a host-only function was called when the MemorySpace is on the device.

        @details 
        @param functionName The name of the host-only function (e.g., "Evaluate(Eigen::RowMatrixXd const& pts)")
         */ 
        void CheckDeviceMismatch(std::string functionName) const;

        /** Checks to see if the coefficients have been initialized yet. If not, an exception is thrown. */
        void CheckCoefficients(std::string const& functionName) const;

        Kokkos::View<double*, MemorySpace> savedCoeffs;

    }; // class ConditionalMapBase<MemorySpace>
}

#endif