import statistics
import numpy as np
from Feature_Extractor import normalized_intensity, filter_pitch, semi_adjust, analysis_note
# list comprehension 优雅的算法
# input_list = [0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 5, 8, 9, 0, 0, 0, 0, 2, 9, 8]
#
# sublists = []
# sublist = []
#
# # len = 3
#     # hyperparameter len(assigned duration of each note and silence)
#
# for num in input_list:
#     if num != 0:
#         sublist.append(num)
#     elif sublist:
#         templist = [statistics.median(sublist)] * len(sublist)
#         # sublists.append([0] * len)
#         sublists.append(templist)
#         sublist = []
#
# if sublist:
#     templist_last = [statistics.median(sublist)] * len(sublist)
#     # sublists.append([0] * len)
#     sublists.append(templist_last)
#     # sublists.append([0] * len)
#
# sublists = [elem for row in sublists for elem in row]
#     # filter out the silence, and use the median value of non-zeros, make it a new list
# print(sublists)
# # filter out the silence, and use the median value

# test
# semi_drag = np.array([1,-15,9,10,25,2])
#
# for i in range(len(semi_drag)):
#     if semi_drag[i] > 12 or semi_drag[i] < -12:  # limitation, how to perceive octave jump? worth thinking
#         if semi_drag[i] > 12:
#            semi_drag[i] = semi_drag[i] % 12
#         else:
#             semi_drag[i] = semi_drag[i] % 12 - 12
#
# print(semi_drag)

# print(np.log2(32.703), np.log2(1046.502))

#
octave_C = [32.703, 65.406, 130.813, 261.626, 523.251, 1046.502]  # C1-C6 frequency in HZ, C4->middle C
octave_C_log = np.log2(octave_C)
x = np.linspace(5.0313135, 6.03135108, 12)
print(x)
print(octave_C_log)

# test_note = 2**7.948
# # test_note2 = 2**8
# print(test_note)
# print(test_note2)
octave, semi_tone_relative, semitone_absolute, note = analysis_note(168.43556171751226)
print(octave, semi_tone_relative, semitone_absolute, note)

#
# octave2, semi_tone_relative2, semitone_absolute2, note2 = analysis_note(test_note2)
# print(octave2, semi_tone_relative2, semitone_absolute2, note2)
