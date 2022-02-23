#ifndef MPART_DERIVATIVEFLAGS_H
#define MPART_DERIVATIVEFLAGS_H

namespace mpart{
namespace DerivativeFlags{

    enum DerivativeType {
        None,       //<- No derivatives
        Parameters, //<- Deriv wrt coeffs
        Diagonal,   //<- first derivative wrt diagonal
        Diagonal2   //<- second derivative wrt diagonal
    };

}
}
#endif // #ifndef MPART_DERIVATIVEFLAGS_H