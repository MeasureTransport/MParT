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
numPts = 20000
testPts = numPts//5
np.random.seed(seed)
samples = np.random.randn(dim,numPts)
target_samples = np.vstack([samples[0,:], samples[1,:] + samples[0,:]**2])

# Separate into testing and training samples
test_samples = target_samples[:,:testPts]
train_samples = target_samples[:,testPts:]

# Create training objective
obj2 = mpart.CreateGaussianKLObjective(np.asfortranarray(train_samples),np.asfortranarray(test_samples))
obj1 = mpart.CreateGaussianKLObjective(np.asfortranarray(train_samples),np.asfortranarray(test_samples),1)

# Create untrained maps
map_options = mpart.MapOptions()
map2 = mpart.CreateTriangular(dim,dim,2,map_options) # Triangular map
map1 = mpart.CreateComponent(mpart.FixedMultiIndexSet(2,2),map_options) # Singular Component

# Train map
train_options = mpart.TrainOptions()
mpart.TrainMap(map2, obj2, train_options)
mpart.TrainMap(map1, obj1, train_options)

def test_TestError():
    assert obj1.TestError(map1) < 5.
    assert obj2.TestError(map2) < 5.

def test_Normality():
    print("Testing map1...")
    KS_stat1 = KS_statistic(map1, test_samples)
    print("Testing map2...")
    KS_stat2 = KS_statistic(map2, test_samples)
    assert KS_stat1 < 0.1
    assert KS_stat2 < 0.1