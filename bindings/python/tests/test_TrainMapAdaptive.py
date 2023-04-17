# Test methods of TriangularMap object
import mpart
import numpy as np
import math

def KS_statistic(map, test_samples):
    pullback_samples = map.Evaluate(test_samples)
    print(pullback_samples.shape)
    sorted_samps = np.sort(pullback_samples.flatten())
    erf = np.vectorize(math.erf)
    samps_cdf = (1 + erf(sorted_samps/np.sqrt(2)))/2
    samps_ecdf = (np.arange(pullback_samples.size) + 1)/pullback_samples.size
    KS_stat = np.abs(samps_cdf - samps_ecdf).max()
    return KS_stat

# Create samples from banana
dim=2
seed = 43
numPts = 5000
testPts = numPts//5
np.random.seed(seed)
samples = np.random.randn(dim,numPts)
target_samples = np.vstack([samples[0,:], samples[1,:] + samples[0,:]**2])

# Separate into testing and training samples
test_samples = target_samples[:,:testPts]
train_samples = target_samples[:,testPts:]

# Create training objective
obj = mpart.CreateGaussianKLObjective(np.asfortranarray(train_samples),np.asfortranarray(test_samples))

# Use ATM to build a map
opts = mpart.ATMOptions()
msets = [mpart.MultiIndexSet.CreateTotalOrder(d+1,1) for d in range(2)]
map = mpart.TrainMapAdaptive(msets, obj, opts)

def test_Msets():
    # Make sure the multiindex set has changed
    assert msets[1].Size()>3

def test_TestError():
    assert obj.TestError(map) < 5.

def test_Normality():
    print("Testing map1...")
    KS_stat = KS_statistic(map, test_samples)
    assert KS_stat < 0.1