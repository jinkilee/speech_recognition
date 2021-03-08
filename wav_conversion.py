import glob
import re
from subprocess import call
from tqdm import tqdm

cmdstr = 'ffmpeg -f s16le -ar 16k -ac 1 -i {} {}'

for pcm in tqdm(glob.glob('/data_hdd/stt/KsponSpeech_0?/*/**.pcm'), desc='conversion'):
	wav = re.sub('pcm', 'wav', pcm)
	cmd = cmdstr.format(pcm, wav).split()
	call(cmd)
	#print('conversion {} -> {}'.format(pcm, wav))
