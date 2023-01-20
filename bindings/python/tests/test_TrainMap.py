# Test methods of TriangularMap object
import mpart
import numpy as np
import math

# Create samples from banana
dim=2
seed = 43
numPts = 20000
testPts = numPts//5
np.random.seed(seed)
samples = np.random.randn(dim,numPts)
target_samples = np.vstack([samples[0,:], samples[1,:] + samples[0,:]**2])

# Separate into testing and training samples
test_samples = target_samples[:,:testPts]
train_samples = target_samples[:,testPts:]

# Create training objective
obj = mpart.GaussianKLObjective(np.asfortranarray(train_samples),np.asfortranarray(test_samples))

# Create untrained map
map_options = mpart.MapOptions()
map = mpart.CreateTriangular(dim,dim,2,map_options)

# Train map
train_options = mpart.TrainOptions()
mpart.TrainMap(map, obj, train_options)

def test_TestError():
    assert obj.TestError(map) < 5.

def test_Normality():
    pullback_samples = map.Evaluate(test_samples)
    sorted_samps = np.sort(pullback_samples[:])
    erf = np.vectorize(math.erf)
    samps_cdf = (1 + erf(sorted_samps/np.sqrt(2)))/2
    samps_ecdf = (np.arange(testPts) + 1)/testPts
    KS_stat = np.abs(samps_cdf - samps_ecdf).max()
    assert KS_stat < 0.1