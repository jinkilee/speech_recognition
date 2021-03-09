import numpy as np
import contextlib
import librosa
import struct
#import soundfile

import scipy.io.wavfile as wav
import python_speech_features as features

def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return  float2pcm(sig, dtype='int16').tobytes()

def byte_to_float(byte):
    # byte -> int16(PCM_16) -> float32
    return pcm2float(np.frombuffer(byte,dtype=np.int16), dtype='float32')

def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def create_mfcc(input_filename, sr=16000):
    # read pcm file
    if input_filename.endswith('wav'):
        rate, sample = wav.read(input_filename)
        assert sr == rate, 'sample rate of {} is not allowed, only {}'.format(rate, sr)

    # read pcm file
    if input_filename.endswith('pcm'):
        with open(input_filename, 'rb') as f:
            byte = f.read()
        sample = np.frombuffer(byte, dtype=np.int16)

    mfcc = features.mfcc(sample, sr, winlen=0.025, winstep=0.01, numcep = 13, nfilt=26,
            preemph=0.97, appendEnergy=True)
    d_mfcc = features.delta(mfcc, 2)
    a_mfcc = features.delta(d_mfcc, 2)
    out = np.concatenate([mfcc, d_mfcc, a_mfcc], axis=1)
    return out

out = create_mfcc('1.pcm')
print(out.shape)

out = create_mfcc('2.pcm')
print(out.shape)
