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
        fx = mt.MpartTorchAutograd.apply(x,None,f,False)
        assert torch.all(fx==x)

        loss = fx.sum()
        loss.backward()
        assert torch.all(x.grad==sens)


    def test_AutogradCoeffs():

        opts = mt.MapOptions()
        tmap = mt.CreateTriangular(dim,dim,3,opts) # Simple third order map

        x.requires_grad = True
        assert torch.autograd.gradcheck(lambda xx: mt.MpartTorchAutograd.apply(x,None,tmap,False), x)
        assert torch.autograd.gradcheck(lambda xx: mt.MpartTorchAutograd.apply(x,None,tmap,True)[1], x)
        assert torch.autograd.gradcheck(lambda xx: mt.MpartTorchAutograd.apply(x,None,tmap,True)[0], x)

        x.requires_grad = False        
        coeffs = torch.randn(tmap.numCoeffs, dtype=torch.double)
        coeffs.requires_grad = True 

        # Check the autograd gradient with finite differences
        assert torch.autograd.gradcheck(lambda c: mt.MpartTorchAutograd.apply(x,c,tmap,False), coeffs)
        assert torch.autograd.gradcheck(lambda c: mt.MpartTorchAutograd.apply(x,c,tmap,True)[1], coeffs)
        assert torch.autograd.gradcheck(lambda c: mt.MpartTorchAutograd.apply(x,c,tmap,True)[0], coeffs)


        # Use the TorchParameterizedFunctionBase module to compute gradients wrt to the coeffs
        tmap2 = mt.TorchParameterizedFunctionBase(tmap)
        assert tmap2.coeffs.grad is None

        loss = tmap2.forward(x.T).sum()
        loss.backward()
        assert tmap2.coeffs.grad is not None

        # Use the TorchConditionalMapBase module to compute gradients wrt to the coeffs
        tmap2 = mt.TorchConditionalMapBase(tmap)
        assert not tmap2.return_logdet
        assert tmap2.coeffs.grad is None

        loss = tmap2.forward(x).sum()
        loss.backward()
        assert tmap2.coeffs.grad is not None
        
        tmap2.return_logdet = True 
        y, logdet = tmap2.forward(x)
        loss = -0.5*(y*y).sum() + logdet.sum()

        loss.backward()
        assert tmap2.coeffs.grad is not None
    
    def test_TorchMethod():
        opts = mt.MapOptions()
        tmap = mt.CreateTriangular(dim,dim,3,opts) # Simple third order map

        x = torch.randn(numSamps, dim, dtype=torch.double)
        tmap2 = tmap.torch()
        y = tmap2.forward(x)
        assert np.all(y.detach().numpy() == tmap.Evaluate(x.T.detach().numpy()).T)

        tmap2 = tmap.torch(return_logdet=True)
        y, logdet = tmap2.forward(x)
        
        assert np.all(y.detach().numpy() == tmap.Evaluate(x.T.detach().numpy()).T)
        assert np.all(logdet.detach().numpy() == tmap.LogDeterminant(x.T.detach().numpy()))


if __name__=='__main__':
    test_Evaluation()
    test_GradientEvaluation()
    test_Autograd()
    test_AutogradCoeffs()
    test_TorchMethod()