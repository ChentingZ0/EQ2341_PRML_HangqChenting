from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
import GetMusicFeatures
from Feature_Extractor import normalized_intensity, filter_pitch, analysis_note
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
# wav1 = wave.open(path1, 'rb')
# data = wav1.readframes(wav1.getnframes())
# play_obj = sa.play_buffer(data, wav1.getnchannels(), wav1.getsampwidth(), wav1.getframerate())
# play_obj.wait_done()
# wav1.close()

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
# ----------------------------------------------------

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
# ----------------------------------------------------

# plot the pitch
# \todo look at the frequency range 100–300 Hz especially
plt.figure(2, figsize=[10,15])
plt.subplot(3, 1, 1)
plt.plot(frequence1, frIsequence1[0, :])
plt.ylabel('pitch/Hz')
plt.title('melody 1')

plt.subplot(3, 1, 2)
plt.plot(frequence2, frIsequence2[0, :])
plt.ylabel('pitch/Hz')
plt.title('melody 2')

plt.subplot(3, 1, 3)
plt.plot(frequence3, frIsequence3[0, :])
plt.xlabel('frame index')
plt.ylabel('pitch/Hz')
plt.title('melody 3')
plt.savefig('figures/pitch_plot.png')
plt.show()
# ----------------------------------------------------

# plot the correlation coefficient
plt.figure(3, figsize=[10,15])
plt.subplot(3, 1, 1)
plt.plot(frequence1, frIsequence1[1, :])
plt.ylabel('Correlation coefficient')
plt.title('melody 1')

plt.subplot(3, 1, 2)
plt.plot(frequence2, frIsequence2[1, :])
plt.ylabel('Correlation coefficient')
plt.title('melody 2')

plt.subplot(3, 1, 3)
plt.plot(frequence3, frIsequence3[1, :])
plt.xlabel('frame index')
plt.ylabel('Correlation coefficient')
plt.title('melody 3')
plt.savefig('figures/correlation_plot.png')
plt.show()
# ----------------------------------------------------

# plot the intensity
plt.figure(4, figsize=[10,15])
plt.subplot(3, 1, 1)
plt.plot(frequence1, frIsequence1[2, :])
plt.ylabel('Intensity')
plt.title('melody 1')

plt.subplot(3, 1, 2)
plt.plot(frequence2, frIsequence2[2, :])
plt.ylabel('Intensity')
plt.title('melody 2')

plt.subplot(3, 1, 3)
plt.plot(frequence3, frIsequence3[2, :])
plt.xlabel('frame index')
plt.ylabel('Intensity')
plt.title('melody 3')
plt.savefig('figures/intensity_plot.png')
plt.show()

pitch_log1, intensity_nor1 = normalized_intensity(frIsequence1)
pitch_log2, intensity_nor2 = normalized_intensity(frIsequence2)
pitch_log3, intensity_nor3 = normalized_intensity(frIsequence3)
# ----------------------------------------------------

plt.figure(5, figsize=[10,15])
plt.subplot(3, 1, 1)
plt.plot(frequence1, pitch_log1)
plt.ylabel('pitch_log_scale')
plt.title('melody 1')

plt.subplot(3, 1, 2)
plt.plot(frequence2, pitch_log2)
plt.ylabel('pitch_log_scale')
plt.title('melody 2')

plt.subplot(3, 1, 3)
plt.plot(frequence3, pitch_log3)
plt.xlabel('frame index')
plt.ylabel('pitch_log_scale')
plt.title('melody 3')
plt.savefig('figures/pitch_log_scale.png')
plt.show()
# ----------------------------------------------------
plt.figure(6, figsize=[10,15])
plt.subplot(3, 1, 1)
plt.plot(frequence1, intensity_nor1)
plt.ylabel('Normalized intensity')
plt.title('melody 1')

plt.subplot(3, 1, 2)
plt.plot(frequence2, intensity_nor2)
plt.ylabel('Normalized intensity')
plt.title('melody 2')

plt.subplot(3, 1, 3)
plt.plot(frequence3, intensity_nor3)
plt.xlabel('frame index')
plt.ylabel('Normalized intensity')
plt.title('melody 3')
plt.savefig('figures/normalized intensity.png')
plt.show()


filter_pitch1, filter_pitch1_log = filter_pitch(frIsequence1)
filter_pitch2, filter_pitch2_log = filter_pitch(frIsequence2)
filter_pitch3, filter_pitch3_log = filter_pitch(frIsequence3)

plt.figure(7, figsize=[10,15])
plt.subplot(3, 1, 1)
plt.plot(frequence1, filter_pitch1_log)
plt.ylabel('filter_pitch')
plt.title('melody 1')

plt.subplot(3, 1, 2)
plt.plot(frequence2, filter_pitch2_log)
plt.ylabel('filter_pitch')
plt.title('melody 2')

plt.subplot(3, 1, 3)
plt.plot(frequence3, filter_pitch3_log)
plt.xlabel('frame index')
plt.ylabel('filter_pitch')
plt.title('melody 3')
plt.savefig('figures/filter_pitch_log.png')
plt.show()
# ----------------------------------------------------

# analysis the semitones of melody1



# filter_pitch1
# extract the voiced frame(exclude the duration of silence)
# print(filter_pitch1[26])

# semi_tone_relative = np.zeros(frame_num1)
# octave = np.zeros(frame_num1)
# note = np.zeros(frame_num1)
# for i in range(frame_num1):
#     if filter_pitch1[i] != 0:
#         octave[i], semi_tone_relative[i], _ = analysis_note(filter_pitch1[i])
#
# print(semi_tone_relative)
#
# plt.figure(8)
# plt.subplot(2, 1, 1)
# plt.plot(frequence1, filter_pitch1)
# plt.subplot(2, 1, 2)
# plt.plot(frequence1, semi_tone_relative)
# plt.xlabel('frame index')
# plt.ylabel('pitch1')
# plt.title('melody 1')
# plt.savefig('figures/filter_pitch1.png')
# plt.show()

