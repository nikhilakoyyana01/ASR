import librosa
import math
#for calculating zero crossing rate for voiced region
voiced_file='voiced.wav'
s,fs=librosa.load(voiced_file)
zcr=librosa.feature.zero_crossing_rate(s,frame_length=1024,hop_length=512)[0]
print("zero crossing rate for voiced region=",math.ceil(sum(zcr*100)))
#for calculating zero crossing rate for unvoiced region
unvoiced_file='unvoiced.wav'
s1,fs1=librosa.load(unvoiced_file)
zcr=librosa.feature.zero_crossing_rate(s1,frame_length=1024,hop_length=512)[0]
print("zero crossing rate for unvoiced region=",math.ceil(sum(zcr*100)))
#for calculating energy for voiced region
energy1=librosa.feature.rms(y=s,frame_length=1024,hop_length=512)[0]
print('energy for voiced signal=',sum(energy1))
#for calculating energy for unvoiced region
energy2=librosa.feature.rms(y=s1,frame_length=1024,hop_length=512)[0]
print('energy for unvoiced signal=',sum(energy2))
#for calculating autocorrelation for voiced signal
a1=librosa.autocorrelate(s)
print('autocorrelation for voiced signal=',sum(a1))
#for calculating autocorrelation for unvoiced signal
a2=librosa.autocorrelate(s1)
print('autocorrelation for unvoiced signal=',sum(a2))
