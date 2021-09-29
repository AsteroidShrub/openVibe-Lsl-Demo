#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from pylsl import StreamInlet, resolve_stream, resolve_byprop
import time

def main():

    fs = 500
    fftSize = 128
    displayWindow = 100

    # LSL
    print("looking for a stream...")
    streams = resolve_byprop("type", "signal")

    for strIdx in range(0,len(streams)):
        print("---STREAM ", strIdx)
        print("info.type() ", streams[strIdx].type())
        print("info.name() ", streams[strIdx].name())
        print("info.nominal_srate() ", streams[strIdx].nominal_srate())
        print("info.channel_format() ", streams[strIdx].channel_format())
        if streams[strIdx].name() == "openvibeSignal":
            inlet_signal = StreamInlet(streams[strIdx])
        if streams[strIdx].name() == "openvibeFft":
            inlet_fft = StreamInlet(streams[strIdx])

    f, (ax1, ax2) = plt.subplots(1, 2)
    buffer = []
    buffer_fft = []

    while True:

        sample, timestamp = inlet_signal.pull_sample()
        buffer = np.append(buffer, sample)

        if len(buffer) >= displayWindow:
            start = time.time()
            ax1.clear()
            ax1.plot(buffer)
            plt.pause(0.001)
            buffer = []
            fft, timeFft = inlet_fft.pull_sample()
            buffer_fft = np.append(buffer_fft, fft)
            if len(buffer_fft) >= fftSize:
                f = range(0, len(buffer_fft))
                ax2.clear()
                ax2.semilogy(f, buffer_fft)
                plt.pause(0.001)
                buffer_fft = []
                end = time.time()
                print("ELAPSED : ", end - start)


    plt.show()

if __name__ == '__main__':
    main()
