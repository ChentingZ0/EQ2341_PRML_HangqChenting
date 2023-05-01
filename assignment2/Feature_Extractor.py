import numpy as np
import statistics
from collections import Counter


def mean_nonzero(input_list):
    sublists = []
    sublist = []
    for num in input_list:
        if num != 0:
            sublist.append(num)
        elif sublist:
            templist = [statistics.mean(sublist)] * len(sublist)
            sublists.append(templist)
            sublist = []

    if sublist:
        templist_last = [statistics.mean(sublist)] * len(sublist)
        sublists.append(templist_last)

    sublists = [elem for row in sublists for elem in row]
    # filter out the silence, and use the median value of non-zeros, make it a new list
    return sublists


def normalized_intensity(frIsequence):
    pitch=frIsequence[0, :]
    # correlation=frIsequence[1, :]
    intensity=frIsequence[2, :]

    frame_num = frIsequence.shape[1]
    pitch_log = np.zeros(frame_num)
    intensity_log = np.zeros(frame_num)
    intensity_nor = np.zeros(frame_num)
    # for each frame
    # change pitch and intensity to log2-scale
    for i in range(0, frame_num):
        pitch_log[i] = np.log2(pitch[i])
        intensity_log[i] = np.log2(intensity[i])
    # change intensity to normalized intensity
    for i in range(0, frame_num):
        intensity_nor[i] = (intensity_log[i] - np.min(intensity_log)) / (np.max(intensity_log) - np.min(intensity_log))

    return pitch_log, intensity_nor

    # test code in main(test pass)

def filter_pitch(frIsequence):
    frame_num = frIsequence.shape[1]
    correlation = frIsequence[1, :]
    pitch = frIsequence[0, :]
    pitch_log, intensity_nor = normalized_intensity(frIsequence)
    threshold_cor = 0.75
    threshold_intensity_nor = np.mean(intensity_nor)

    for i in range(0, frame_num):
        if pitch_log[i] > np.log2(1046.502) or pitch_log[i] < np.log2(32.703) or intensity_nor[i] < threshold_intensity_nor or correlation[i] < threshold_cor:
            pitch_log[i] = 0 # define as unvoiced
            pitch[i] = 0

    pitch = mean_nonzero(pitch)
    pitch_log = np.log2(pitch)
    return pitch, pitch_log

# list comprehension. elegant algorithm
def note_name(semi_tone_relative):
    semi_tone_index = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11
    }
    for name, index in semi_tone_index.items():
        if index == semi_tone_relative:
            # print("the corresponding note name is {}".format(name))
            note_represent = name
    return note_represent

def analysis_note(test_tone):
    # analysis each note
    octave_C = [32.703, 65.406, 130.813, 261.626, 523.251, 1046.502]  # C1-C6 frequency in HZ, C4->middle C
    octave_C_log = np.log2(octave_C)
    semi_tones = []  # (5,12)


    for i in range(len(octave_C) - 1):
        semi_tones.append(np.linspace(octave_C_log[i], octave_C_log[i + 1], 13)[:-1].tolist())

    octave = 0
    for i, c in enumerate(octave_C):
        if c <= test_tone < octave_C[i + 1]:
            octave = i + 1
            break

    semitones_specific = semi_tones[octave - 1]
    test_tone_log = np.log2(test_tone)
    semi_tone_relative = 0

    for i, c in enumerate(semitones_specific):
        if i == len(semitones_specific) - 1:
            semi_tone_relative = 11
            if test_tone_log - c > octave_C_log[octave] - test_tone_log:
                octave = octave + 1
                semi_tone_relative = 0
        else:
            if c <= test_tone_log < semitones_specific[i + 1]:  # find the specific interval
                if test_tone_log - c > semitones_specific[i + 1] - test_tone_log:
                    semi_tone_relative = i + 1
                else:
                    semi_tone_relative = i  # approximate to the nearest semitone
                break


    semitone_absolute = octave * 12 + semi_tone_relative
    note = note_name(semi_tone_relative)
    return octave, semi_tone_relative, semitone_absolute, note


octave, semi_tone_relative, semitone_absolute, note = analysis_note(125.71896068623788)
print(octave, semi_tone_relative, semitone_absolute, note)


def semi_adjust(filter_pitch):
    semi_tone_absolute = np.zeros(len(filter_pitch))

    for i in range(len(filter_pitch)):
        if filter_pitch[i] != 0:
            _, _, semi_tone_absolute[i], _ = analysis_note(filter_pitch[i])

    threshold = 5  # window length = (threshold-1)*2
    for i in range(len(semi_tone_absolute)):

        if i - threshold + 1 < 0:
            temp_list = list(semi_tone_absolute[:i + threshold - 1])
        elif i + threshold - 1 > len(semi_tone_absolute) - 1:
            temp_list = list(semi_tone_absolute[i-threshold+1:])
        else:
            temp_list = list(semi_tone_absolute[i-threshold+1:i+threshold-1])

        count_dict = dict(Counter(temp_list))
        if len(list(count_dict.values())) > 1:
            min_element = list(count_dict.keys())[list(count_dict.values()).index(min(count_dict.values()))]
            max_element = list(count_dict.keys())[list(count_dict.values()).index(max(count_dict.values()))]
            if min_element == semi_tone_absolute[i] and count_dict[min_element] < threshold:
                semi_tone_absolute[i] = max_element

    semi_normalized = []
    for i in range(len(semi_tone_absolute)):
        normalized_value = (semi_tone_absolute[i] - np.min(np.array(semi_tone_absolute))) / (np.max(np.array(semi_tone_absolute) - np.min(np.array(semi_tone_absolute))))
        semi_normalized.append(normalized_value)

    semi_normalized = [i*120 for i in semi_normalized]
    semi_normalized = [round(i) for i in semi_normalized]

    return semi_tone_absolute, semi_normalized
















