#ifndef MPART_DERIVATIVEFLAGS_H
#define MPART_DERIVATIVEFLAGS_H

namespace mpart{
namespace DerivativeFlags{

    enum DerivativeType {
        None,       //<- No derivatives
        Parameters, //<- Deriv wrt coeffs
        Diagonal,   //<- first derivative wrt diagonal
        Diagonal2,  //<- second derivative wrt diagonal
        MixedCoeff, //<- gradient wrt coeffs of first derivative wrt x_d
        Input,      //<- gradient wrt map input
        MixedInput  //<- gradient of diagonal wrt map input
    };

}
}
#endif // #ifndef MPART_DERIVATIVEFLAGS_H