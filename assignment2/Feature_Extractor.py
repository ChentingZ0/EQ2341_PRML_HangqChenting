import GetMusicFeatures
import numpy as np
from matplotlib import pyplot as plt


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
    pitch_log, intensity_nor = normalized_intensity(frIsequence)

    # filter out unvoiced frame based on intensity and correlation
    threshold_cor = 0.75
    threshold_intensity_nor = np.mean(intensity_nor)
    for i in range(0, frame_num):
        if pitch_log[i] < 3.46 or pitch_log[i] > 6.9 or intensity_nor[i] < threshold_intensity_nor or correlation[i] < threshold_cor:
            pitch_log[i] = 0 # define as unvoiced
    return pitch_log


    # test code in main(test pass)

def 





