import numpy as np

def PitchSpectralHps(X, f_s):
    iOrder = 4
    f_min = 300
    f = np.zeros(X.shape[1])

    iLen = int((X.shape[0] -1) / iOrder)
    afHps = X[np.arange(0, iLen), :]
    k_min = int(round(f_min / f_s * 2 * (X.shape[0] -1)))

    for j in range(1, iOrder):
        X_d = X[::(j + 1), :]
        afHps *= X_d[np.arange(0, iLen), :]

    f = np.argmax(afHps[np.arange(k_min, afHps.shape[0])], axis=0)

    f = (f + k_min) / (X.shape[0] - 1) * f_s / 2

    return f

import numpy as np
import math
from scipy.signal import filtfilt
from scipy.signal import find_peaks

from .ToolGammatoneFb import ToolGammatoneFb


def PitchTimeAuditory(x, iBlockLength, iHopLength, f_s):

    # initialize
    iNumOfBlocks = math.ceil(x.size / iHopLength)
    f = np.zeros(iNumOfBlocks)
    f_max = 2000
    iNumBands = 20
    fLengthLpInS = 0.001

    iLengthLp = math.ceil(fLengthLpInS * f_s)

    # compute time stamps
    t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

    # apply filterbank
    X = ToolGammatoneFb(x, f_s, iNumBands)

    # half wave rectification
    X[X < 0] = 0

    # smooth the results with a moving average filter
    b = np.ones(iLengthLp) / iLengthLp
    X = filtfilt(b, 1, X)

    for n in range(0, iNumOfBlocks):

        eta_min = int(round(f_s / f_max))
        afSumCorr = np.zeros(iBlockLength - 1)
        x_tmp = np.zeros(iBlockLength)

        i_start = n * iHopLength
        i_stop = np.min([x.size - 1, i_start + iBlockLength - 1])

        # compute ACF per band and summarize
        for k in range(0, iNumBands):
            # get current block
            if X[k, np.arange(i_start, i_stop + 1)].sum() < 1e-20:
                continue
            else:
                x_tmp[np.arange(0, i_stop - i_start + 1)] = X[k, np.arange(i_start, i_stop + 1)]

            afCorr = np.correlate(x_tmp, x_tmp, "full") / np.dot(x_tmp, x_tmp)

            # aggregate bands with simple sum before peak picking
            afSumCorr += afCorr[np.arange(iBlockLength, afCorr.size)]

        if afSumCorr.sum() < 1e-20:
            continue

        # find the highest local maximum
        iPeaks = find_peaks(afSumCorr, height=0)
        if iPeaks[0].size:
            eta_min = np.max([eta_min, iPeaks[0][0] - 1])
        f[n] = np.argmax(afSumCorr[np.arange(eta_min + 1, afSumCorr.size)]) + 1

        # convert to Hz
        f[n] = f_s / (f[n] + eta_min + 1)

    return (f, t)

