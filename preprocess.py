import re
import scipy.io.wavfile as wav
import python_speech_features as features
import numpy as np

INITIALS = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
"char list: Hangul initials (초성)"

MEDIALS = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
"char list: Hangul medials (중성)"

FINALS = list("∅ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
"char list: Hangul finals (종성)."

SPACE_TOKEN = " "
LABELS = sorted({SPACE_TOKEN}.union(INITIALS).union(MEDIALS).union(FINALS))
"char list: All CTC labels."

def check_syllable(char):
	return 0xAC00 <= ord(char) <= 0xD7A3

def split_syllable(char):
	assert check_syllable(char)
	diff = ord(char) - 0xAC00
	_m = diff % 28
	_d = (diff - _m) // 28
	return (INITIALS[_d // 21], MEDIALS[_d % 21], FINALS[_m])

'''
- b/ 숨소리
- l/ 웃음소리
- o/ 다른사람말소리
- n/ 주변잡음
- 간투어
'''
def preprocess_text(txt_filename):
	with open(txt_filename, 'r', encoding='cp949') as f:
		txt = f.readline()
		result = ''
		for char in re.sub('\\s+', SPACE_TOKEN, txt.strip()):
			if char == SPACE_TOKEN:
				result += SPACE_TOKEN
			elif check_syllable(char):
				result += ''.join(split_syllable(char))
	return result

def preprocess_audio(filename):
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

