try:
    import torch
    haveTorch = True
except:
    print('Could not import torch in test_TorchWrapper.py.  PyTorch interface will not be tested.')
    haveTorch = False

import mpart as mt 
import numpy as np

if haveTorch:

    numSamps = 5
    dim = 2
    x = torch.randn(dim, numSamps, dtype=torch.double)
    out = torch.zeros(dim,numSamps, dtype=torch.double)
    grad = torch.zeros(dim, numSamps, dtype=torch.double)
    sens = torch.ones(dim,numSamps, dtype=torch.double)

    f = mt.IdentityMap(dim, dim)

    def test_Evaluation():
        f.EvaluateImpl(mt.ExtractTorchTensorData(x), mt.ExtractTorchTensorData(out))
        assert torch.all(x==out)
    
    def test_GradientEvaluation():
        f.GradientImpl(mt.ExtractTorchTensorData(x), mt.ExtractTorchTensorData(sens), mt.ExtractTorchTensorData(grad))
        assert torch.all(grad==sens)
    
    def test_Autograd():
        x.requires_grad = True 
        fx = mt.TorchWrapper_ParameterizedFunction.apply(x,None,f)
        assert torch.all(fx==x)

        loss = fx.sum()
        loss.backward()
        assert torch.all(x.grad==sens)

    def test_AutogradCoeffs():
        x.requires_grad = False 

        opts = mt.MapOptions()
        tmap = mt.CreateTriangular(dim,dim,3,opts) # Simple third order map

        coeffs = torch.randn(tmap.numCoeffs, dtype=torch.double)
        coeffs.requires_grad = True 

        # Check the autograd gradient with finite differences
        assert torch.autograd.gradcheck(lambda c: mt.TorchWrapper_ParameterizedFunction.apply(x,c,tmap), coeffs)


if __name__=='__main__':
    test_Evaluation()
    test_GradientEvaluation()
    test_Autograd()
    test_AutogradCoeffs()