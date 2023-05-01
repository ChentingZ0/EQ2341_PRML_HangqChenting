from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
import GetMusicFeatures
from Feature_Extractor import normalized_intensity, filter_pitch, semi_adjust, analysis_note
import os
import simpleaudio as sa
import wave
# #
# #
# # load data
path1 = 'Songs/melody_1.wav'
path2 = 'Songs/melody_2.wav'
path3 = 'Songs/melody_3.wav'
sample_rate1, data1 = wavfile.read(path1)
sample_rate2, data2 = wavfile.read(path2)
sample_rate3, data3 = wavfile.read(path3)
print("Sample rate:", sample_rate1)
print("Number of samples:", data1.shape[0])
print("Duration (seconds):", data1.shape[0] / sample_rate1)

frIsequence1 = GetMusicFeatures.GetMusicFeatures(data1, sample_rate1)
frIsequence2 = GetMusicFeatures.GetMusicFeatures(data2, sample_rate2)
frIsequence3 = GetMusicFeatures.GetMusicFeatures(data3, sample_rate3)
frame_num1 = frIsequence1.shape[1]
frame_num2 = frIsequence2.shape[1]
frame_num3 = frIsequence3.shape[1]
frequence1 = np.arange(0, frame_num1)
frequence2 = np.arange(0, frame_num2)
frequence3 = np.arange(0, frame_num3)
pitch1 = frIsequence1[0, :]

# test if transposition invariant
frIsequence1_trans = np.copy(frIsequence1)
shift = 7
for i in range(0, frame_num1):
    # f2 = f1 * (2 ^ (1 / 12)) ^ n
    frIsequence1_trans[0, i] = frIsequence1[0, i] * 2**(shift/12)

pitch1_trans = frIsequence1_trans[0, :]
# print(type(pitch1_trans),'what pitch1 trans type')
# check where zero occurs
# if np.any(pitch1 == 0):
#     print("Zero exists in the pitch1.")
# else:
#     print("Zero does not exist in pitch1.")
#
# if np.any(pitch1_trans == 0):
#     print("Zero exists in the pitch1_trans.")
# else:
#     print("Zero does not exist in pitch1_trans.")
#



# Feature
pitch_log1, intensity_nor1 = normalized_intensity(frIsequence1)
filter_pitch1, filter_pitch1_log = filter_pitch(frIsequence1)

print(filter_pitch1, '\n', 'filter_pitch value')
# print(set(filter_pitch1), '\n', 'filter_pitch value')

voice_frame_num1 = len(filter_pitch1)
frequence1_temp = np.arange(0, voice_frame_num1)

semi_absolute1, semi_drag1 = semi_adjust(filter_pitch1)
print(semi_absolute1, '\n', 'absolute tone original')
notes = semi_absolute1
# for i in notes:
#     _,_,_,note = analysis_note(i)
#     print(note, end="")
# print(semi_drag1, '\n', 'original drag')



pitch_log1_trans, intensity_nor1_trans = normalized_intensity(frIsequence1_trans)
filter_pitch1_trans, filter_pitch1_trans_log = filter_pitch(frIsequence1_trans)
print(set(filter_pitch1_trans),'\n','filter_pitch_trans value')
semi_absolute1_trans, semi_drag1_trans = semi_adjust(filter_pitch1_trans)

voice_frame_num1_trans = len(filter_pitch1_trans)
frequence1_temp_trans = np.arange(0, voice_frame_num1_trans)


print(semi_absolute1_trans, '\n', 'absolute tone trans')
# print(semi_drag1_trans, '\n', 'trans drag')

#
# gap1 = np.max(pitch_log1)-np.min(pitch_log1)
# gap1_trans = np.max(pitch_log1_trans)-np.min(pitch_log1_trans)
# print(gap1, gap1_trans,'gap')

plt.figure(figsize=[10,15])
plt.plot(frequence1_temp, filter_pitch1)
plt.plot(frequence1_temp_trans, filter_pitch1_trans)
plt.xlabel('frame number')
plt.ylabel('filter_pitch')
plt.title('melody 1 and trans')
plt.savefig('figures/filter_pitch_test.png')
plt.show()
#

plt.figure(figsize=[10,15])
plt.plot(frequence1_temp, filter_pitch1_log)
plt.plot(frequence1_temp_trans, filter_pitch1_trans_log)
plt.xlabel('frame number')
plt.ylabel('filter_pitch_log')
plt.title('melody 1 and trans')
plt.savefig('figures/filter_pitch_log_test.png')
plt.show()
#


plt.figure(figsize=[10,15])
plt.plot(frequence1_temp, semi_absolute1)
plt.plot(frequence1_temp, semi_absolute1_trans)
plt.xlabel('frame index')
plt.ylabel('semi-drag')
plt.title('melody 1 and trans')
plt.savefig('figures/semitones_absolute_test.png')
plt.show()


plt.figure(figsize=[10,15])
plt.plot(frequence1_temp, semi_drag1)
plt.plot(frequence1_temp, semi_drag1_trans)
plt.xlabel('frame index')
plt.ylabel('semi-drag')
plt.title('melody 1 and trans')
plt.savefig('figures/semitones_drag_test.png')
plt.show()

# #
# octave_C = [32.703, 65.406, 130.813, 261.626, 523.251, 1046.502]  # C1-C6 frequency in HZ, C4->middle C
# octave_C_log = np.log2(octave_C)
# x = np.linspace(5.0313135, 6.03135108, 12)
# print(x)
# print(octave_C_log)

# test_note = 2**7.948
# test_note2 = 2**8
# print(test_note)
# print(test_note2)
octave, semi_tone_relative, semitone_absolute, note = analysis_note(126)
print(octave, semi_tone_relative, semitone_absolute, note)

#
# octave2, semi_tone_relative2, semitone_absolute2, note2 = analysis_note(test_note2)
# print(octave2, semi_tone_relative2, semitone_absolute2, note2)
