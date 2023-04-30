import GetMusicFeatures
import numpy as np
from matplotlib import pyplot as plt
import math

def normalized_intensity(frIsequence):
    pitch=frIsequence[0, :]
    # correlation=frIsequence[1, :]
    intensity=frIsequence[2, :]

    frame_num = frIsequence.shape[1]
    pitch_log = np.zeros(frame_num)
    intensity_log = np.zeros(frame_num)
    intensity_nor = np.zeros(frame_num)
    # for each frame
    # change pitch and intensity to log-scale
    for i in range(0, frame_num):
        pitch_log[i] = np.log(pitch[i])
        intensity_log[i] = np.log(intensity[i])
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

    # filter out unvoiced frame based on intensity and correlation
    threshold_cor = 0.8
    threshold_intensity_nor = np.mean(intensity_nor)
    for i in range(0, frame_num):
        if pitch_log[i] < 3.46 or pitch_log[i] > 6.9 or intensity_nor[i] < threshold_intensity_nor or correlation[i] < threshold_cor:
            pitch_log[i] = 0 # define as unvoiced
            pitch[i] = 0
    return pitch, pitch_log


    # test code in main(test pass)

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
    octave_C_log = np.log(octave_C)
    semi_tones = []  # (5,12)
    # print(octave_C_log)

    for i in range(len(octave_C) - 1):
        semi_tones.append(np.linspace(octave_C_log[i], octave_C_log[i + 1], 12).tolist())
    # print(semi_tones)

    octave = 0
    for i, c in enumerate(octave_C):
        if c <= test_tone < octave_C[i + 1]:
            octave = i + 1
            break

    # print("The test tone belongs to octave C{}".format(octave))
    semitones_specific = semi_tones[octave - 1]

    # print(semi_tones[octave - 1])

    test_tone_log = np.log(test_tone)
    semi_tone_relative = 0

    for i, c in enumerate(semitones_specific):
        if c <= test_tone_log < semitones_specific[i + 1]:      # find the specific interval
            if test_tone_log - c > semitones_specific[i + 1] - test_tone_log:
                semi_tone_relative = i + 1
            else: semi_tone_relative = i                        # approximate to the nearest semitone
            break

    note = note_name(semi_tone_relative)
    # use relative semi_tone can address the problem of jump one octave(8)
    return octave, semi_tone_relative, note


# test_tone = 329.628
# octave, semi_tone_relative, note = analysis_note(test_tone)









