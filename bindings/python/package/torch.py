import torch
from .torch_helpers import ExtractTorchTensorData, MpartTorchAutograd
        

class TorchParameterizedFunctionBase(torch.nn.Module):
    """ Defines a wrapper around the MParT ParameterizedFunctionBase class that
        can be used with pytorch.  
    """

    def __init__(self, f=None, store_coeffs=True, dtype=torch.double):
        super().__init__()

        self.f = f 
        self.dtype = dtype 

        if store_coeffs:
            coeff_tensor = torch.tensor( self.f.CoeffMap(), dtype=dtype)
            self.coeffs = torch.nn.Parameter(coeff_tensor)
        else:
            self.coeffs = None 
    
    def forward(self, x, coeffs=None):
        
        if coeffs is None:
            if self.coeffs is None:
                raise RuntimeError('Must either set store_coeffs=True in constructor or pass coeffs to forward function.')
            else:
                coeffs = self.coeffs

        return MpartTorchAutograd.apply(x.T, coeffs, self.f, False).T


class TorchConditionalMapBase(torch.nn.Module):
    """ Defines a wrapper around the MParT ConditionalMapBase class that
        can be used with pytorch.  The log determinant of the function's
        Jacobian can also be return by setting the return_logdet attribute.
        This can be done either in the constructor or afterwards.
    """

    def __init__(self, f=None, store_coeffs=True, return_logdet=False, dtype=torch.double):
        super().__init__()

        self.return_logdet = return_logdet
        self.f = f 

        if store_coeffs:
            coeff_tensor = torch.tensor( self.f.CoeffMap(), dtype=dtype)
            self.coeffs = torch.nn.Parameter(coeff_tensor)
        else:
            self.coeffs = None 
    
    def forward(self, x, coeffs=None):
        
        if coeffs is None:
            if self.coeffs is None:
                raise RuntimeError('Must either set store_coeffs=True in constructor or pass coeffs to forward function.')
            else:
                coeffs = self.coeffs
        
        self._check_shapes(x, None, coeffs)

        if self.return_logdet:
            y, logdet = MpartTorchAutograd.apply(x.T, coeffs, self.f, self.return_logdet)
            return y.T, logdet
        else:
            y = MpartTorchAutograd.apply(x.T, coeffs, self.f, self.return_logdet)
            return y.T

    def inverse(self, x, r, coeffs=None):

        
        if coeffs is None:
            if self.coeffs is None:
                raise RuntimeError('Must either set store_coeffs=True in constructor or pass coeffs to inverse function.')
            else:
                coeffs_dbl = self.coeffs.double()
        else:
            coeffs_dbl = coeffs.double()
        self.f.WrapCoeffs(ExtractTorchTensorData(coeffs_dbl))

        x_dbl = x.double()
        r_dbl = r.double()

        if (x.shape[1] != self.f.inputDim) & ((x.shape[1]+r.shape[1]) == self.f.inputDim):
            x_dbl = torch.hstack([x_dbl,r_dbl])

        self._check_shapes(x_dbl, r_dbl, coeffs_dbl)

        output = torch.zeros((self.f.outputDim, x_dbl.shape[0]), dtype=torch.double)
        self.f.InverseImpl(ExtractTorchTensorData(x_dbl.T), ExtractTorchTensorData(r_dbl.T), ExtractTorchTensorData(output))

        return output.T.type(x.dtype)


    def _check_shapes(self, x, r, coeffs):

        if x is not None:
            if (x.shape[1] != self.f.inputDim):
                raise ValueError(f'Input to map has wrong shape.  x has {x.shape[1]} columns but map expects input dimension of {self.f.inputDim}.')

        if r is not None:
            if (r.shape[1] != self.f.outputDim):
                raise ValueError(f'Reference input to map inverse has wrong shape.  r has {r.shape[1]} columns but map expects input dimension of {self.f.outputDim}.')

        if coeffs is not None:
            if (coeffs.shape[0] != self.f.numCoeffs):
                raise ValueError(f'Specified coefficients have wrong shape. coeffs input has length {coeffs.shape[0]} but map has {self.f.numCoeffs} coefficients.')