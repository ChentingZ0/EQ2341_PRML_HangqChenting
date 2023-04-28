from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
import GetMusicFeatures
import Feature_Extractor
import os
import simpleaudio as sa
import wave


# check WAV file
path1 = 'Songs/melody_1.wav'
path2 = 'Songs/melody_2.wav'
path3 = 'Songs/melody_3.wav'
sample_rate1, data1 = wavfile.read(path1)
sample_rate2, data2 = wavfile.read(path2)
sample_rate3, data3 = wavfile.read(path3)
print("Sample rate:", sample_rate1)
print("Number of samples:", data1.shape[0])
print("Duration (seconds):", data1.shape[0] / sample_rate1)

# listen to wav file
wav1 = wave.open(path1, 'rb')
data = wav1.readframes(wav1.getnframes())
play_obj = sa.play_buffer(data, wav1.getnchannels(), wav1.getsampwidth(), wav1.getframerate())
play_obj.wait_done()
wav1.close()

# print time plot of three wav

wav1 = wave.open(path1, 'rb')
wav2 = wave.open(path2, 'rb')
wav3 = wave.open(path3, 'rb')


# close all wav files
def close_all(wav1, wav2, wav3):
    wav1.close()
    wav2.close()
    wav3.close()

def data_wav(wav):
    data = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i%d' % wav.getsampwidth())
    data = np.reshape(data, (-1, wav.getnchannels()))
    time = np.arange(wav.getnframes()) / float(wav.getframerate())
    return time, data[:,0]

time1, data1 = data_wav(wav1)
time2, data2 = data_wav(wav2)
time3, data3 = data_wav(wav3)
plt.figure(1, figsize=[10, 15])
plt.subplot(3,1,1)
plt.plot(time1, data1)
# plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('melody 1')

plt.subplot(3,1,2)
plt.plot(time2, data2)
# plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('melody 2')

plt.subplot(3,1,3)
plt.plot(time3, data3)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('melody 3')
plt.savefig('figures/time_plot.png')
plt.show()

# close_all(wav1, wav2, wav3)

## check what is frIsequence
# first row: an estimated pitch ft
# second row: estimated correlation coefficient rt between adjacent pitch periods
# third row: frame root-mean-square intensity It
# sample_rate1, data1 = wavfile.read(path1)
frIsequence1 = GetMusicFeatures.GetMusicFeatures(data1, sample_rate1)
frIsequence2 = GetMusicFeatures.GetMusicFeatures(data2, sample_rate2)
frIsequence3 = GetMusicFeatures.GetMusicFeatures(data3, sample_rate3)
frame_num1 = frIsequence1.shape[1]
frame_num2 = frIsequence2.shape[1]
frame_num3 = frIsequence3.shape[1]

frequence1 = np.arange(0,frame_num1)
frequence2 = np.arange(0,frame_num2)
frequence3 = np.arange(0,frame_num3)


print(frIsequence1.shape,type(frIsequence1),'dimension of frIsequence')
print(frIsequence1[:,0:5],'print the first five frames')

# plot the pitch
plt.figure(2, figsize=[10,15])
plt.subplot(3, 1, 1)
plt.plot(frequence1, frIsequence1[0, :])
plt.ylabel('Frequency/Hz')
plt.title('melody 1')

plt.subplot(3, 1, 2)
plt.plot(frequence2, frIsequence2[0, :])
plt.ylabel('Frequency/Hz')
plt.title('melody 2')

plt.subplot(3, 1, 3)
plt.plot(frequence3, frIsequence3[0, :])
plt.xlabel('Time frame')
plt.ylabel('Frequency/Hz')
plt.title('melody 3')
plt.savefig('figures/pitch_plot.png')
plt.show()

# plot the correlation coefficient
plt.figure(3, figsize=[10,15])
plt.subplot(3, 1, 1)
plt.plot(frequence1, frIsequence1[1, :])
plt.ylabel('Frequency/Hz')
plt.title('melody 1')

plt.subplot(3, 1, 2)
plt.plot(frequence2, frIsequence2[1, :])
plt.ylabel('Frequency/Hz')
plt.title('melody 2')

plt.subplot(3, 1, 3)
plt.plot(frequence3, frIsequence3[1, :])
plt.xlabel('Correlation coefficient')
plt.ylabel('Frequency/Hz')
plt.title('melody 3')
plt.savefig('figures/correlation_plot.png')
plt.show()

# plot the intensity
plt.figure(4, figsize=[10,15])
plt.subplot(3, 1, 1)
plt.plot(frequence1, frIsequence1[2, :])
plt.ylabel('Frequency/Hz')
plt.title('melody 1')

plt.subplot(3, 1, 2)
plt.plot(frequence2, frIsequence2[2, :])
plt.ylabel('Frequency/Hz')
plt.title('melody 2')

plt.subplot(3, 1, 3)
plt.plot(frequence3, frIsequence3[2, :])
plt.xlabel('Intensity')
plt.ylabel('Frequency/Hz')
plt.title('melody 3')
plt.savefig('figures/intensity_plot.png')
plt.show()








