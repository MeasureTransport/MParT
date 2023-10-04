#ifndef MPART_DIAGONALTENSORPRODUCTFUNCTION_H
#define MPART_DIAGONALTENSORPRODUCTFUNCTION_H

#include <utility>

#include <Kokkos_Core.hpp>
#include <Eigen/Core>

namespace mpart {

/**
 * We assume that OffDiagFunction satisfies the Cached Paramterization Concept
 * and DiagFunction has function FillCache (since it's univariate, no need for 1 and 2)
*/
template<class OffDiagFunction, class DiagFunction>
class DiagonalTensorProductFunction {
public:

    template<typename MemorySpace>
    DiagonalTensorProductFunction(OffDiagFunction const& f1,
        DiagFunction const& f2) : _f1(f1),
                                  _f2(f2),
                                  _dim(f1.InputSize()+1),
                                  _coeff_f1(f1.NumCoeffs()),
                                  _coeff_f2(f2.NumCoeffs())
    {

    };


    KOKKOS_INLINE_FUNCTION unsigned int CacheSize() const {return _f1.CacheSize() + _f2.CacheSize();};

    KOKKOS_INLINE_FUNCTION unsigned int NumCoeffs() const {return _coeff_f1 + _coeff_f2;};

    KOKKOS_INLINE_FUNCTION unsigned int InputSize() const {return _dim1 + _dim2;};

    // Fills all the cache entries independent of dimension \f$d\f$,
    // which is exactly all of the dimensions dependent on _f1
    template<typename PointType>
    KOKKOS_FUNCTION void FillCache1(double* cache,
                    PointType const&        pt,
                    DerivativeFlags::DerivativeType derivType) const
    {
        auto pt2 = Kokkos::subview(pt, std::make_pair(_dim1, _dim1+_dim2));

        _f1.FillCache1(cache, pt, derivType);
        _f1.FillCache2(cache, pt, pt(_dim-2), derivType);
    }

    template<class PointType>
    KOKKOS_FUNCTION void FillCache2(double* cache,
                    PointType const&,
                    double                  xd,
                    DerivativeFlags::DerivativeType derivType) const
    {
        _f2.FillCache(&cache[_f1.CacheSize()], xd, derivType);
    }


    template<typename CoeffVecType>
    KOKKOS_FUNCTION double Evaluate(const double* cache,
                    CoeffVecType const& coeffs) const
    {
        double f;
        auto coeffs1 = Kokkos::subview(coeffs, std::make_pair(int(0), int(_coeff_f1)));
        f = _f1.Evaluate(cache, coeffs1);

        auto coeffs2 = Kokkos::subview(coeffs, std::make_pair(int(_coeff_f1), int(_coeff_f1+_coeff_f2)));
        f *= _f2.Evaluate(&cache[_f1.CacheSize()], coeffs2);

        return f;
    }


    template<typename CoeffVecType>
    KOKKOS_FUNCTION double DiagonalDerivative(const double* cache,
                              CoeffVecType const&           coeffs,
                              unsigned int                  derivOrder) const
    {
        double df;

        auto coeffs1 = Kokkos::subview(coeffs, std::make_pair(int(0), int(_coeff_f1)));
        df = _f1.Evaluate(cache, coeffs1);

        auto coeffs2 = Kokkos::subview(coeffs, std::make_pair(int(_coeff_f1), int(_coeff_f1+_coeff_f2)));
        df *= _f2.InputDerivative(&cache[_f1.CacheSize()], coeffs2, derivOrder);

        return df;
    }


    template<typename CoeffVecType, typename GradVecType>
    KOKKOS_FUNCTION double CoeffDerivative(const double* cache,
                           CoeffVecType const& coeffs,
                           GradVecType grad) const
    {
        double f1, f2;

        auto coeffs1 = Kokkos::subview(coeffs, std::make_pair(int(0), int(_dim1)));
        Eigen::Ref<Eigen::VectorXd> gradHead = grad.head(_f1.NumCoeffs());
        f1 = _f1.CoeffDerivative(cache, coeffs1, gradHead);

        auto coeffs2 = Kokkos::subview(coeffs, std::make_pair(_dim1, _dim1+_dim2));
        Eigen::Ref<Eigen::VectorXd> gradTail = grad.tail(_f2.NumCoeffs());
        f2 = _f2.CoeffDerivative(&cache[_f1.CacheSize()], coeffs2, gradTail);

        gradHead *= f2;
        gradTail *= f1;

        return f1*f2;
    }

    template<typename CoeffVecType, typename GradVecType>
    KOKKOS_FUNCTION double MixedDerivative(const double* cache,
                           CoeffVecType const& coeffs,
                           unsigned int derivOrder,
                           GradVecType grad) const
    {
        double f1, df2;

        auto coeffs1 = Kokkos::subview(coeffs, std::make_pair(int(0), int(_dim1)));
        Eigen::Ref<Eigen::VectorXd> gradHead = grad.head(_f1.NumCoeffs());
        f1 = _f1.CoeffDerivative(cache, coeffs1, gradHead);

        auto coeffs2 = Kokkos::subview(coeffs, std::make_pair(_dim1, _dim1+_dim2));
        Eigen::Ref<Eigen::VectorXd> gradTail = grad.tail(_f2.NumCoeffs());
        df2 = _f2.MixedDerivative(&cache[_f1.CacheSize()], coeffs2, derivOrder, gradTail);

        gradHead *= df2;
        gradTail *= f1;

        return f1*df2;
    }



private:
    OffDiagFunction _f1;
    DiagFunction _f2;
    unsigned int _dim; // (number of inputs to f1) + 1
    unsigned int _coeff_f1; // Number of coefficients for f1
    unsigned int _coeff_f2; // Number of coefficients for f2

}; // class DiagonalTensorProductFunction

} // namespace mpart

#endif // #ifndef MPART_DIAGONALTENSORPRODUCTFUNCTION_H