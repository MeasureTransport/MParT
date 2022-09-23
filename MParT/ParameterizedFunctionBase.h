#ifndef MPART_ParameterizedFunctionBase_H
#define MPART_ParameterizedFunctionBase_H

#include "MParT/Utilities/EigenTypes.h"
#include "MParT/Utilities/ArrayConversions.h"

#include "MParT/Utilities/GPUtils.h"

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

            @return A reference to the Kokkos::View containing the coefficients.
        */
        virtual Kokkos::View<double*, MemorySpace>& Coeffs(){return this->savedCoeffs;};

        /** @brief Set the internally stored view of coefficients.
            @detail Performs a deep copy of the input coefficients to the internally stored coefficients.
            @param coeffs A view containing the coefficients to copy.
        */
        virtual void SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs);

        /** @brief Wrap the internal coefficient view around another view.
            @detail Performs a shallow copy of the input coefficients to the internally stored coefficients.
            If values in the view passed to this function are changed, the values will also change in the
            internally stored view.
            @param coeffs A view containing the coefficients we want to wrap.
        */
        virtual void WrapCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs);

        #if defined(MPART_ENABLE_GPU)
        virtual void SetCoeffs(Kokkos::View<double*, mpart::DeviceSpace> coeffs);
        virtual void WrapCoeffs(Kokkos::View<double*, mpart::DeviceSpace> coeffs);
        #endif 

        /** SetCoeffs function with conversion from Eigen to Kokkos vector types.*/
        virtual void SetCoeffs(Eigen::Ref<Eigen::VectorXd> coeffs);
        virtual void WrapCoeffs(Eigen::Ref<Eigen::VectorXd> coeffs);

        /** Returns an eigen map wrapping around the coefficient vector, which is stored in a Kokkos::View.  Updating the
            components of this map should also update the view.
        */
        virtual Eigen::Map<Eigen::VectorXd> CoeffMap();

        /** Const version of the Coeffs() function. */
        virtual Kokkos::View<const double*, MemorySpace> Coeffs() const{return this->savedCoeffs;};

        /** Evaluate function with conversion between default view layout and const strided matrix. */
        template<typename ViewType>
        StridedMatrix<double, typename ViewType::memory_space> Evaluate(ViewType pts){StridedMatrix<const double, typename ViewType::memory_space> newpts(pts); return this->Evaluate(newpts);}

        /** Evaluate function with conversion from Eigen to Kokkos (and possibly copy to/from device.) */
        Eigen::RowMatrixXd Evaluate(Eigen::Ref<const Eigen::RowMatrixXd> const& pts);

        /** @brief Evaluate the function at multiple points.
        @details Computes \f$\mathbf{y}^{(i)}=T(\mathbf{x}^{(i)}; \mathbf{w})\f$ for \f$N\f$ points \f$\{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(N)}\}\f$.  The parameters \f$\mathbf{w}\f$ are defined by the :code:`SetCoeffs` function.
        @param pts A \f$d_{in}\times N\f$ matrix containining \f$N\f$ points in \f$\mathbb{R}^d\f$ where this function be evaluated.  Each column is a point.
        @return A \f$d_{out}\times N\f$ matrix containing evaluations of this function.  
        */
        template<typename AnyMemorySpace>
        StridedMatrix<double, AnyMemorySpace> Evaluate(StridedMatrix<const double, AnyMemorySpace> const& pts);

        /** Pure virtual EvaluateImpl function that is overridden by children of this class.
        @param pts A \f$d_{in}\times N\f$ matrix containining \f$N\f$ points in \f$\mathbb{R}^d\f$ where this function be evaluated.  Each column is a point.
        @param output A preallocated \f$d_{out}\times N\f$ that will be filled with the evaluations.  This matrix must be sized correctly before being passed into this function.
        */
        virtual void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                  StridedMatrix<double, MemorySpace>              output) = 0;


        /** Evaluate the gradient of the function with conversion between default view layout and const strided matrix. */
        template<typename ViewType1, typename ViewType2>
        StridedMatrix<double, typename ViewType1::memory_space> Gradient(ViewType1 pts, ViewType2 sens){
            StridedMatrix<const double, typename ViewType1::memory_space> newpts(pts);
            StridedMatrix<const double, typename ViewType1::memory_space> newSens(sens);
            return this->Gradient(newpts, newSens);
        }

        /** Evaluate the gradient of the function with conversion from Eigen to Kokkos (and possibly copy to/from device.) */
        Eigen::RowMatrixXd Gradient(Eigen::Ref<const Eigen::RowMatrixXd> const& pts,
                                    Eigen::Ref<const Eigen::RowMatrixXd> const& sens);

        /** @brief Evaluate the gradient of the function at multiple points.
        @details For input points \f$x^{(i)}\f$ and sensitivity vectors \f$s^{(i)}\f$, this function computes 
                 \f[
                    g^{(i)} = \left[s^{(i)}\right]^T\nabla_x T(x^{(i)}; w),
                 \f] 
        where \f$\nabla_x T\f$ is the Jacobian of the function \f$T\f$ with respect to the input \f$x\f$.  Note that this function can
        be used to evaluate one step of the chain rule (e.g., one backpropagation step).  Given a scalar-valued function \f$g\f$, the gradient of 
        \f$g(T(x))\f$ with respect to \f$x\f$ is given by \f$\left(\nabla g\right)^T \left(\nabla_x T\right)\f$.  Passing \f$\nabla g\f$ as the 
        sensitivity input to this function then allows you to compute this product and thus the gradient of the composed function \f$g(T(x))\f$.
        
        @param pts A \f$d_{in}\times N\f$ matrix containining \f$N\f$ points in \f$\mathbb{R}^{d_{in}}\f$ where this function be evaluated.  Each column is a point.
        @param sens A \f$d_{out}\times N\f$ matrix containing \f$N\f$ sensitivity vectors in \f$\mathbb{R}^{d_{out}}\f$.  
        @return A \f$d_{in}\times N\f$ matrix containing the gradient of vectors.  Each column corresponds to the gradient at a particular point and sensitivity.  
        */
        template<typename AnyMemorySpace>
        StridedMatrix<double, AnyMemorySpace> Gradient(StridedMatrix<const double, AnyMemorySpace> const& pts,
                                                       StridedMatrix<const double, AnyMemorySpace> const& sens);


        virtual void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                  StridedMatrix<const double, MemorySpace> const& sens,
                                  StridedMatrix<double, MemorySpace>              output) = 0;


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
        template<typename AnyMemorySpace>
        StridedMatrix<double, AnyMemorySpace> CoeffGrad(StridedMatrix<const double, AnyMemorySpace> const& pts, 
                                                        StridedMatrix<const double, AnyMemorySpace> const& sens);

        /** CoeffGrad function with conversion between general view type and const strided matrix. */
        template<typename PtsViewType, typename SensViewType>
        StridedMatrix<double, typename PtsViewType::memory_space> CoeffGrad(PtsViewType pts,  SensViewType sens){
            StridedMatrix<const double, typename PtsViewType::memory_space> newpts(pts); 
            StridedMatrix<const double, typename SensViewType::memory_space> newsens(sens); 
            return this->CoeffGrad(newpts,newsens);
        }

        /** Coeff grad function with conversion between Eigen and Kokkos matrix types. */
        Eigen::RowMatrixXd CoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts,
                                     Eigen::Ref<const Eigen::RowMatrixXd> const& sens);

        /** @brief Pure virtual function overridden by child classes for computing the gradient of the function output with respect to the parameter vector \f$\mathbf{w}\f$.  See the non-virtual CoeffGrad function for more details.
        @details Evaluates the gradient with respect to the coefficients and stores the results in a preallocated matrix.
        @param pts A \f$d_{in}\times N\f$ matrix containining \f$N\f$ points in \f$\mathbb{R}^d\f$ where this function be evaluated.  Each column is a point.
        @param sens A matrix of sensitivity vectors.  Each column contains one sensitivity.
        @param output A preallocated matrix to store the results.
        */
        virtual void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                   StridedMatrix<const double, MemorySpace> const& sens,
                                   StridedMatrix<double, MemorySpace>              output) = 0;

        
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

    }; // class ParameterizedFunctionBase
}

#endif