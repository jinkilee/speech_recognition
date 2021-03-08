import scipy.io.wavfile as wav
import python_speech_features as features
import numpy as np

def create_mfcc(filename):
    """Perform standard preprocessing, as described by Alex Graves (2012)
    http://www.cs.toronto.edu/~graves/preprint.pdf
    Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
    [1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)
    """

    (rate,sample) = wav.read(filename)

    mfcc = features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep = 13, nfilt=26,
    preemph=0.97, appendEnergy=True)
    d_mfcc = features.delta(mfcc, 2)
    a_mfcc = features.delta(d_mfcc, 2)

    out = np.concatenate([mfcc, d_mfcc, a_mfcc], axis=1)

    return out, out.shape[0]

