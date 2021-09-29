#!/usr/bin/env python

import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from pylsl import StreamInlet, resolve_stream, resolve_byprop
import time
import mne


def main():

    plotMatrices = False
    plotConnectomes = True
    threshold = 0.33
    thresholdPercent = 5
    thresholdType = 'percent' # 'absolute' or 'percent'
    thresholdedMatrix = 'MSC' # 'ImCoh' or 'MSC'

    nbChannels = 63
    electrodes = ['FP2','AF4','AF8','F2','F4','F6','F8','FC2','FC4','FC6','FT8','FT10','C2','C4','C6','T8','CP2','CP4','CP6','TP8','P2','P4','P6','P8','PO4','PO8','O2','PO10','Oz','POz','Pz','CPz','Cz','FCz','Fz','O1','PO9','PO3','PO7','P1','P3','P5','P7','CP1','CP3','CP5','TP7','TP9','C1','C3','C5','T7','FC1','FC3','FC5','FT7','F1','F3','F5','F7','AF3','AF7','FP1']

    # --- LSL stream
    print("looking for a stream...")
    streams = resolve_byprop("type", "signal")

    for strIdx in range(0, len(streams)):
        print("---STREAM ", strIdx)
        print("info.type() ", streams[strIdx].type())
        print("info.name() ", streams[strIdx].name())
        print("info.nominal_srate() ", streams[strIdx].nominal_srate())
        print("info.channel_format() ", streams[strIdx].channel_format())
        if streams[strIdx].name() == "openvibeConnectMSC":
            inlet_msc = StreamInlet(streams[strIdx])
        if streams[strIdx].name() == "openvibeConnectImCoh":
            inlet_imcoh = StreamInlet(streams[strIdx])

    if plotMatrices:
        # --- Plot inits
        f, axes = plt.subplots(1, 2)
        # --- initializing "empty" (masked) matrices of nbChan x nbChan
        im1 = axes[0].imshow(ma.array(np.zeros((nbChannels,nbChannels)),mask=np.ones((nbChannels,nbChannels))),
                             cmap='hot', vmin=0, vmax=1)
        im2 = axes[1].imshow(ma.array(np.zeros((nbChannels, nbChannels)), mask=np.ones((nbChannels, nbChannels))),
                             cmap='hot', vmin=0, vmax=1)

        f.tight_layout()

        # --- Titles, colourbar, etc.
        cbar = f.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_ticks(np.arange(0, 0.25, 1.0))

        axes[0].set_title('MSC')
        axes[1].set_title('ImCoh')
        axes[0].set_xticks(np.arange(len(electrodes)))
        axes[0].set_yticks(np.arange(len(electrodes)))
        axes[1].set_xticks(np.arange(len(electrodes)))
        axes[1].set_yticks(np.arange(len(electrodes)))
        axes[0].set_xticklabels(electrodes)
        axes[0].set_yticklabels(electrodes)
        axes[1].set_xticklabels(electrodes)
        axes[1].set_yticklabels(electrodes)
        axes[0].grid(which="minor", color="b", linestyle='-', linewidth=1)
        axes[1].grid(which="minor", color="b", linestyle='-', linewidth=1)

        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # --- Mask for displaying only the upper diagonal of the matrices
        mask = np.zeros((nbChannels,nbChannels))
        mask[np.tril_indices_from(mask)] = True

    elif plotConnectomes:
        f, axes = plt.subplots(num=None, facecolor='black')
        f, axes = mne.viz.plot_connectivity_circle(ma.array(np.zeros((nbChannels,nbChannels)),mask=np.ones((nbChannels,nbChannels))),
                                                   electrodes, fig=f, subplot=(1, 1, 1),
                                                   facecolor='black', textcolor='white', node_edgecolor='black',
                                                   linewidth=1.5, colormap='hot', vmin=0, vmax=1, show=False)
        # f, axes = mne.viz.plot_connectivity_circle(ma.array(np.zeros((nbChannels,nbChannels)),mask=np.ones((nbChannels,nbChannels))),
        #                                           electrodes, fig = f, subplot=(1, 2, 2),
        #                                           facecolor='black', textcolor='white', node_edgecolor='black',
        #                                           linewidth=1.5, colormap='hot', vmin=0, vmax=1, show=False)


    # --- Matrices & vectors init
    matrix_msc = []
    matrix_imcoh = []
    vector_msc = []
    vector_imcoh = []

    # --- Nb of elements in a matrix - because we'll accumulate serialized data in a vector
    lastIdx = nbChannels*nbChannels

    # --- Main Loop
    while True:
        startLoop = time.time()

        # --- Get Connectivity Matrices values
        matrix_msc = []
        matrix_imcoh = []
        if 'inlet_msc' in locals():
            while len(vector_msc) < lastIdx:
                sample_msc, timestamp_msc = inlet_msc.pull_sample()
                vector_msc = np.append(vector_msc, sample_msc)
        if 'inlet_imcoh' in locals():
            while len(vector_imcoh) < lastIdx:
                sample_imcoh, timestamp_imcoh = inlet_imcoh.pull_sample()
                vector_imcoh = np.append(vector_imcoh, sample_imcoh)

        # --- Shape vectors as matrices
        matrix_msc   = vector_msc[0:lastIdx].reshape(nbChannels,nbChannels)
        if 'inlet_imcoh' in locals(): matrix_imcoh = vector_imcoh[0:lastIdx].reshape(nbChannels,nbChannels)

        # --- Keep "tail" of vectors, which wasn't used for the current matrix
        indicesToRemove = np.arange(0,1,lastIdx)
        vector_msc = vector_msc[lastIdx+1:]
        if 'inlet_imcoh' in locals(): vector_imcoh = vector_imcoh[lastIdx+1:]

        midLoop = time.time()

        if plotMatrices:
            # --- Update heatmap datas
            matrix_msc = ma.array(matrix_msc, mask=mask)
            matrix_imcoh = ma.array(matrix_imcoh, mask=mask)
            im1.set_array(matrix_msc)
            im2.set_array(matrix_imcoh)

        elif plotConnectomes:

            if thresholdedMatrix == 'MSC':
                if thresholdType == 'absolute':
                    matrix_thresholded = (matrix_msc > threshold) * matrix_msc
                    title = "MSC Connectivity, threshold " + repr(threshold)
                elif thresholdType == 'percent':
                    sortedVec = np.sort(matrix_msc, axis=None)
                    idxPercent = int( np.ceil(len(sortedVec)*(100-thresholdPercent)/100) )
                    matrix_thresholded = (matrix_msc > sortedVec[idxPercent]) * matrix_msc
                    title = "MSC Connectivity, " + repr(thresholdPercent) + "% strongest links"

            elif (thresholdedMatrix == 'ImCoh') and ('inlet_imcoh' in locals()):
                if thresholdType == 'absolute':
                    matrix_thresholded = (matrix_imcoh > threshold) * matrix_imcoh
                    title = "ImCoh Connectivity, threshold " + repr(threshold)
                elif thresholdType == 'percent':
                    sortedVec = np.sort(matrix_imcoh, axis=None)
                    idxPercent = int(np.ceil(len(sortedVec) * (100 - thresholdPercent)/100))
                    matrix_thresholded = (matrix_imcoh > sortedVec[idxPercent]) * matrix_imcoh
                    title = "ImCoh Connectivity, " + repr(thresholdPercent) + "% strongest links"

            plt.cla()
            f, axes = mne.viz.plot_connectivity_circle(matrix_thresholded, electrodes, fig=f, subplot=(1, 1, 1),
                                                        facecolor='black', textcolor='white', node_edgecolor='black',
                                                        linewidth=1.5, colormap='hot', vmin=0, vmax=1, show=False,
                                                        title=title)
            # f, axes = mne.viz.plot_connectivity_circle(matrix_imcoh, electrodes, fig=f, subplot=(1, 2, 2),
            #                                 facecolor='black', textcolor='white', node_edgecolor='black',
            #                                 linewidth=1.5, colormap='hot', vmin=0, vmax=1, show=False)

        plt.pause(0.001)

        endLoop = time.time()
        print("ELAPSED ALL : ", endLoop - startLoop, " sec / DATA ACQ : ", midLoop-startLoop, " sec / DISPLAY : ", endLoop-midLoop, " sec")

    plt.show()

if __name__ == '__main__':
    main()
