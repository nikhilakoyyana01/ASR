import librosa , librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
fs, data = wavfile.read('voiced.wav')
fs = 8000
start=1
stop=2
duration = len(data)/fs
time = np.arange(0,duration,1/fs)
plt.xlabel('Time [s]')
plt.ylabel('Amp ')
plt.plot(time[start:stop], data[start:stop])
plt.title('Unvoiced Region Plot')
plt.show()
zcr =np.asarray([int(x) for x in librosa.zero_crossings(data[start:stop])])
print("Total zero crossings in the frame is = ",sum(zcr))