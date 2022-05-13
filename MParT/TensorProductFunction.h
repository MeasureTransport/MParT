#ifndef MPART_TENSORPRODUCTFUNCTION_H
#define MPART_TENSORPRODUCTFUNCTION_H

#include <utility>

#include <Kokkos_Core.hpp>
#include <Eigen/Core>

namespace mpart {


template<class FunctionType1, class FunctionType2>
class TensorProductFunction {
public:

    TensorProductFunction(FunctionType1 const& f1,
                          FunctionType2 const& f2) : _f1(f1),
                                                     _dim1(f1.InputSize()),
                                                     _f2(f2),
                                                     _dim2(f2.InputSize())
    {

    };


    unsigned int CacheSize() const{ return _f1.CacheSize() + _f2.CacheSize();};

    unsigned int NumCoeffs() const{return _f1.NumCoeffs() + _f2.NumCoeffs();};

    unsigned int InputSize() const {return _dim1 + _dim2;};

    template<class... KokkosProperties>
    void FillCache1(double*                                           cache,
                    Kokkos::View<double*, KokkosProperties...> const& pt,
                    DerivativeFlags::DerivativeType                   derivType) const
    {
        auto pt2 = Kokkos::subview(pt, std::make_pair(_dim1, _dim1+_dim2));

        _f1.FillCache1(cache, pt, derivType);
        _f1.FillCache2(cache, pt, pt(_dim1-1), derivType);

        _f2.FillCache1(&cache[_f1.CacheSize()], pt2, derivType);
    }

    template<class... KokkosProperties>
    void FillCache2(double*                                           cache,
                    Kokkos::View<double*, KokkosProperties...> const& pt,
                    double                                            xd,
                    DerivativeFlags::DerivativeType                   derivType) const
    {
        auto pt2 = Kokkos::subview(pt, std::make_pair(int(_dim1), int(pt.extent(0))));

        _f2.FillCache2(&cache[_f1.CacheSize()], pt2, xd, derivType);
    }


    template<class... KokkosProperties>
    double Evaluate(const double* cache,
                    Kokkos::View<double*, KokkosProperties...> const& coeffs) const
    {
        double f;

        auto coeffs1 = Kokkos::subview(coeffs, std::make_pair(int(0), int(_dim1)));
        f = _f1.Evaluate(cache, coeffs1);

        auto coeffs2 = Kokkos::subview(coeffs, std::make_pair(_dim1, _dim1+_dim2));
        f *= _f2.Evaluate(&cache[_f1.CacheSize()], coeffs2);

        return f;
    }


    template<class... KokkosProperties>
    double DiagonalDerivative(const double*                                     cache,
                              Kokkos::View<double*, KokkosProperties...> const& coeffs,
                              unsigned int                                      derivOrder) const
    {
        double df;

        auto coeffs1 = Kokkos::subview(coeffs, std::make_pair(int(0), int(_dim1)));
        df = _f1.Evaluate(cache, coeffs1);

        auto coeffs2 = Kokkos::subview(coeffs, std::make_pair(_dim1, _dim1+_dim2));
        df *= _f2.DiagonalDerivative(&cache[_f1.CacheSize()], coeffs2, derivOrder);

        return df;
    }


    template<class... KokkosProperties>
    double CoeffDerivative(const double* cache,
                           Kokkos::View<double*, KokkosProperties...> const& coeffs,
                           Eigen::Ref<Eigen::VectorXd> grad) const
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

    template<class... KokkosProperties>
    double MixedDerivative(const double* cache,
                           Kokkos::View<double*, KokkosProperties...> const& coeffs,
                           unsigned int derivOrder,
                           Eigen::Ref<Eigen::VectorXd> grad) const
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
    FunctionType1 _f1;
    unsigned int _dim1; //<- The number of inputs to f1

    FunctionType2 _f2;
    unsigned int _dim2; //<- The number of inputs to f2

}; // class TensorProductFunction

} // namespace mpart

#endif // #ifndef MPART_TENSORPRODUCTFUNCTION_H