# Copyright 2020 Deepwave Digital Inc.

from matplotlib import pyplot as plt
import cupy


def psd(y1, y2, fs1, fs2, f1, f2, nfft=16384, title=''):
    # Plot signals
    plt.figure(figsize=(7, 5))
    plt.subplot(211)
    plt.psd(cupy.asnumpy(y1), Fs=fs1, Fc=f2, NFFT=nfft)
    plt.ylim((-160, -75))
    plt.title('{} Before Filter'.format(title))
    plt.subplot(212)
    plt.psd(cupy.asnumpy(y2), Fs=fs2, Fc=f1, NFFT=nfft)
    plt.ylim((-160, -75))
    plt.title('{} After Filter'.format(title))
    plt.show()
