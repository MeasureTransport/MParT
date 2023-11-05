import torch

def ExtractTorchTensorData(tensor):
    """ Extracts the pointer, shape, and stride from a pytorch tensor and returns a tuple
        that can be passed to MParT functions that have been overloaded to accept
        (double*, std::tuple<int,int>, std::tuple<int,int>) instead of a Kokkos::View.

        Arguments:
        ------------
        tensor: pytorch.Tensor
            The pytorch tensor we want to eventually wrap with a Kokkos view.
        
        Returns:
        ------------
        Tuple[int, Tuple[int,int], Tuple[int,int]]
            A python tuple that contains all information needed to construct a Kokkos::View.
            After casting to c++ types using pybind, this output can be passed to the 
            mpart::ConstructViewFromPointer function.
    """

    # Make sure the tensor has double data type
    if tensor.dtype != torch.float64:
        raise ValueError(f'Currently only tensors with float64 datatype can be converted.  Current dtype is {tensor.dtype}')
    
    if len(tensor.shape)==1:
        return tensor.data_ptr(), tensor.shape[0], tensor.stride()[0]
    elif len(tensor.shape)==2:
        return tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride())
    else:
        raise ValueError(f'Currently only 1d and 2d tensors can be converted.')

class MpartTorchAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, coeffs, f, return_logdet):
        ctx.save_for_backward(input, coeffs)
        ctx.f = f

        if coeffs is not None:
            f.WrapCoeffs(ExtractTorchTensorData(coeffs))

        output = torch.zeros(f.outputDim, input.shape[1], dtype=input.dtype) 
        f.EvaluateImpl(ExtractTorchTensorData(input), ExtractTorchTensorData(output))

        if return_logdet:
            logdet = torch.zeros(input.shape[1], dtype=input.dtype)
            f.LogDeterminantImpl(ExtractTorchTensorData(input), ExtractTorchTensorData(logdet))
            return output, logdet
        else:
            return output

    @staticmethod
    def backward(ctx, output_sens, logdet_sens=None):
        input, coeffs = ctx.saved_tensors
        f = ctx.f
        
        if coeffs is not None:
            f.WrapCoeffs(ExtractTorchTensorData(coeffs))

        # Get the gradient wrt input 
        grad = None 
        if input.requires_grad:          
            grad = torch.zeros(f.inputDim, input.shape[1], dtype=input.dtype)

            f.GradientImpl(ExtractTorchTensorData(input), 
                          ExtractTorchTensorData(output_sens),
                          ExtractTorchTensorData(grad))

            if logdet_sens is not None:
                grad2 = torch.zeros(f.inputDim, input.shape[1], dtype=input.dtype)

                f.LogDeterminantInputGradImpl(ExtractTorchTensorData(input), 
                                              ExtractTorchTensorData(grad2))
                grad += grad2*logdet_sens[None,:]

        coeff_grad = None
        if coeffs is not None:
            if coeffs.requires_grad:
                coeff_grad = torch.zeros(f.numCoeffs, input.shape[1], dtype=input.dtype)
                f.CoeffGradImpl(ExtractTorchTensorData(input),
                                ExtractTorchTensorData(output_sens),
                                ExtractTorchTensorData(coeff_grad))

                coeff_grad = coeff_grad.sum(axis=1) # pytorch expects total gradient not per-sample gradient

                if logdet_sens is not None:
                    grad2 = torch.zeros(f.numCoeffs, input.shape[1], dtype=input.dtype)

                    f.LogDeterminantCoeffGradImpl(ExtractTorchTensorData(input), 
                                                 ExtractTorchTensorData(grad2))
                    
                    coeff_grad += torch.sum(grad2*logdet_sens[None,:],axis=1)
                    
        return grad, coeff_grad, None, None  



class TorchParameterizedFunctionBase(torch.nn.Module):
    """ Defines a wrapper around the MParT ParameterizedFunctionBase class that
        can be used with pytorch.  
    """

    def __init__(self, f):
        super().__init__()

        self.f = f 

        coeff_tensor = torch.tensor( self.f.CoeffMap(), dtype=torch.double)
        self.coeffs = torch.nn.Parameter(coeff_tensor)

    def forward(self, x):
        
        return MpartTorchAutograd.apply(x.T, self.coeffs, self.f, False).T


class TorchConditionalMapBase(torch.nn.Module):
    """ Defines a wrapper around the MParT ConditionalMapBase class that
        can be used with pytorch.  The log determinant of the function's
        Jacobian can also be return by setting the return_logdet attribute.
        This can be done either in the constructor or afterwards.
    """

    def __init__(self, f, return_logdet=False):
        super().__init__()

        self.return_logdet = return_logdet
        self.f = f 

        coeff_tensor = torch.tensor( self.f.CoeffMap(), dtype=torch.double)
        self.coeffs = torch.nn.Parameter(coeff_tensor)

    def forward(self, x):
        
        if self.return_logdet:
            y, logdet = MpartTorchAutograd.apply(x.T, self.coeffs, self.f, self.return_logdet)
            return y.T, logdet
        else:
            y = MpartTorchAutograd.apply(x.T, self.coeffs, self.f, self.return_logdet)
            return y.T

    def inverse(self, x, r):
        if (x.shape[1] != self.f.inputDim) & ((x.shape[1]+r.shape[1]) == self.f.inputDim):
            x = torch.hstack([x,r])

        output = torch.zeros((self.f.outputDim, x.shape[0]), dtype=torch.double)
        self.f.InverseImpl(ExtractTorchTensorData(x.T), ExtractTorchTensorData(r.T), ExtractTorchTensorData(output))

        return output.T



    
