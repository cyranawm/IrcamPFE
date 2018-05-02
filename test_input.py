import sys
sys.path.append('./aciditools/')
from aciditools.drumLearning import importDataset
import numpy as np
import librosa
from nsgt.cq import NSGT
from nsgt.fscale import OctScale, LogScale, MelScale
from skimage.transform import resize
from outils.nsgt_inversion import regenerateAudio
from matplotlib import pyplot as plt


#%%
sig, sr = librosa.load('./datasets/DummyDrumsRaw/data/Kicks/VEC2 Bassdrums Breakbeat 03.wav')
print(sr)
sig_pad1 = np.pad(sig, (0,int(np.round(1 * sr)) - len(sig)), 'constant');
sig_pad2 = np.pad(sig, (0,int(np.round(2 * sr)) - len(sig)), 'constant');
print(sig.shape)
print(sig_pad1.shape)
print(sig_pad2.shape)

mFreq = 30
maFreq = 11000
bins = 48

# Create a frequency scale
scl = OctScale(mFreq, maFreq, bins)
# Create a NSGT object
nsgt = NSGT(scl, sr, sig.size, real=True, matrixform=True, reducedform=1)
nsgt1s = NSGT(scl, sr, 1*sr, real=True, matrixform=True, reducedform=1)
nsgt2s = NSGT(scl, sr, 2*sr, real=True, matrixform=True, reducedform=1)

# Compute and turn into Numpy array
NSGT_sigpad1 = np.abs(list(nsgt1s.forward(sig_pad1)))
NSGT_sigpad2 = np.abs(list(nsgt2s.forward(sig_pad2)))
rawNSGT = np.abs(list(nsgt.forward(sig)))

#padded_rawNSGT = np.pad(rawNSGT, ((0,0), (0,rawNSGT.shape[1]//10)), 'linear_ramp')
#padded_rawNSGT = np.pad(padded_rawNSGT, ((0,0), (0,padNSGT.shape[1]-padded_rawNSGT.shape[1])), 'linear_ramp')

plt.figure()
plt.subplot(311)
plt.plot(sig)
plt.subplot(312)
plt.plot(sig_pad1)
plt.subplot(313)
plt.plot(sig_pad2)

plt.figure()
plt.subplot(131)
plt.imshow(rawNSGT, aspect = 'auto')
plt.subplot(132)
plt.imshow(NSGT_sigpad1, aspect = 'auto')
plt.subplot(133)
plt.imshow(NSGT_sigpad2, aspect = 'auto')

regenerateAudio(NSGT_sigpad1, sr=22050, targetLen = 1, iterations=50, nsgtBins=bins, minFreq=mFreq, maxFreq=maFreq, curName='./tests/iterations/sigpad1_50')
regenerateAudio(NSGT_sigpad1, sr=22050, targetLen = 1, iterations=100, nsgtBins=bins, minFreq=mFreq, maxFreq=maFreq, curName='./tests/iterations/sigpad1_100')
regenerateAudio(NSGT_sigpad1, sr=22050, targetLen = 1, iterations=500, nsgtBins=bins, minFreq=mFreq, maxFreq=maFreq, curName='./tests/iterations/sigpad1_500')
regenerateAudio(NSGT_sigpad1, sr=22050, targetLen = 1, iterations=1000, nsgtBins=bins, minFreq=mFreq, maxFreq=maFreq, curName='./tests/iterations/sigpad1_1000')
regenerateAudio(NSGT_sigpad1, sr=22050, targetLen = 1, iterations=2000, nsgtBins=bins, minFreq=mFreq, maxFreq=maFreq, curName='./tests/iterations/sigpad1_2000')
#regenerateAudio(NSGT_sigpad2, sr=22050, targetDur = 2, iterations=50, nsgtBins=bins, minFreq=mFreq, maxFreq=maFreq, curName='./sigpad2', crop = 1)
