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

    def __reduce__(self):
        return (self.__class__, (None,))
    
    @staticmethod
    def forward(ctx, input, coeffs, f, return_logdet):
        ctx.save_for_backward(input, coeffs)
        ctx.f = f

        coeffs_dbl = None 
        if coeffs is not None:
            coeffs_dbl = coeffs.double()
            f.WrapCoeffs(ExtractTorchTensorData(coeffs_dbl))
        input_dbl = input.double()

        output = torch.zeros(f.outputDim, input.shape[1], dtype=torch.double) 
        f.EvaluateImpl(ExtractTorchTensorData(input_dbl), ExtractTorchTensorData(output))

        if return_logdet:
            logdet = torch.zeros(input.shape[1], dtype=torch.double)
            f.LogDeterminantImpl(ExtractTorchTensorData(input_dbl), ExtractTorchTensorData(logdet))
            return output.type(input.dtype), logdet.type(input.dtype)
        else:
            return output.type(input.dtype)

    @staticmethod
    def backward(ctx, output_sens, logdet_sens=None):
        input, coeffs = ctx.saved_tensors
        f = ctx.f
        
        coeffs_dbl = None 
        if coeffs is not None:
            coeffs_dbl = coeffs.double()
            f.WrapCoeffs(ExtractTorchTensorData(coeffs_dbl))
        input_dbl = input.double()
        output_sens_dbl = output_sens.double()

        logdet_sens_dbl = None 
        if logdet_sens is not None:
            logdet_sens_dbl = logdet_sens.double()

        # Get the gradient wrt input 
        grad = None 
        if input.requires_grad:          
            grad = torch.zeros(f.inputDim, input.shape[1], dtype=torch.double)

            f.GradientImpl(ExtractTorchTensorData(input_dbl), 
                          ExtractTorchTensorData(output_sens_dbl),
                          ExtractTorchTensorData(grad))

            if logdet_sens is not None:
                grad2 = torch.zeros(f.inputDim, input.shape[1], dtype=torch.double)

                f.LogDeterminantInputGradImpl(ExtractTorchTensorData(input_dbl), 
                                              ExtractTorchTensorData(grad2))
                grad += grad2*logdet_sens_dbl[None,:]

        coeff_grad = None
        if coeffs is not None:
            if coeffs.requires_grad:
                coeff_grad = torch.zeros(f.numCoeffs, input.shape[1], dtype=torch.double)
                f.CoeffGradImpl(ExtractTorchTensorData(input_dbl),
                                ExtractTorchTensorData(output_sens_dbl),
                                ExtractTorchTensorData(coeff_grad))

                coeff_grad = coeff_grad.sum(axis=1) # pytorch expects total gradient not per-sample gradient

                if logdet_sens is not None:
                    grad2 = torch.zeros(f.numCoeffs, input.shape[1], dtype=torch.double)

                    f.LogDeterminantCoeffGradImpl(ExtractTorchTensorData(input_dbl), 
                                                 ExtractTorchTensorData(grad2))
                    
                    coeff_grad += torch.sum(grad2*logdet_sens[None,:],axis=1)
                    
        if coeff_grad is not None:
            coeff_grad = coeff_grad.type(input.dtype)
        
        if grad is not None:
            grad = grad.type(input.dtype)
        
        return grad, coeff_grad, None, None  