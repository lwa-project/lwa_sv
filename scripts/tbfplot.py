#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Selectively plots data from the *_tbfspecs.dat files output by PASI.
"""

import matplotlib; matplotlib.use('Agg')

import optparse
import struct
import sys
import numpy as np

from lsl.common import stations


class TbfData:
    colors = {'X': '#008000', 'Y': '#0000C0'}
    
    def __init__(self, filename):
        with open(filename, 'r') as f:
            nStands, nChans = struct.unpack('ll', f.read(16))
            self.specs = np.fromfile(
                f, count = nStands * 2 * nChans, dtype = np.float32
                ).reshape(nStands, 2, nChans)
        self.freqs = np.arange(nChans) * 25e3
        self.specs = np.ma.array(self.specs, mask=np.where(self.specs == 0, 1, 0))
        self.stdIndices = {}
	lwa1 = stations.parseSSMIF('/home/adp/tbfspecs/SSMIF_CURRENT.txt')
        for i, a in enumerate(lwa1.getAntennas()):
            if i % 2 == 0:
                self.stdIndices[a.stand.id] = i / 2
    
    def plotMany(self, stds, filename = None):
        n = len(stds)
        nCols = int(np.ceil(np.sqrt(n)))
        nRows = int(np.ceil(n / float(nCols)))
        
        if filename is None:
            fig = plt.figure()
        else:
            fig = matplotlib.figure.Figure(figsize=(9,9))
        fig.subplots_adjust(wspace = 0, hspace = 0, left=0.12, bottom=0.12)
	fig.suptitle('%s'% inFileName)
        axes = []
        
        x = self.freqs / 1e6
        
        xMin, xMax = 0, 98
        yMin, yMax = +9e99, -9e99
        
        for iPlot, std in enumerate(stds):
            leftCol = (iPlot % nCols == 0)
            bottomRow = (iPlot / nCols + 1 == nRows)
            ax = fig.add_subplot(nRows, nCols, iPlot + 1)
            
            i = self.stdIndices[std]
            for j, pol in enumerate(['Y', 'X']):
                y = 10 * np.log10(self.specs[i, j, :])
                ax.plot(x, y, color = self.colors[pol])
                if yMax < y.max(): yMax = y.max()
                if yMin > y.min(): yMin = y.min()
            
            if leftCol:
                if iPlot / nCols == (nRows - 1) / 2:
                    ax.set_ylabel('Power (dB)')

            else:
                ax.set_yticklabels([])
            if bottomRow:
                if iPlot % nCols == (nCols - 1) / 2:
                    ax.set_xlabel('Frequency (MHz)')
            else:
                ax.set_xticklabels([])
            
            axes.append(ax)
        
        yMin, yMax = yMin, yMin + 60
        for ax, std in zip(axes, stds):
            ax.text(xMin * 0.05 + xMax * 0.95, yMin * 0.05 + yMax * 0.95,
                    '%d' % std, ha = 'right', va = 'top')
            ax.set_xlim(xMin, xMax)
            ax.set_ylim(yMin, yMax)
        
        if filename is None:
            plt.show()
        else:
            canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
            canvas.print_figure(filename, dpi = 80)

if __name__ == '__main__':
    parser = optparse.OptionParser(
        usage = 'usage: %prog [options] DATFILE STD [STD...]', description =
        'Plots the spectra of the specified STD(s) for the given TBF spectra '
        'file from ADP.  Green shows X pol and blue shows Y pol.')
    parser.add_option('-o', dest = 'outfile', metavar = 'FILE', help =
                      'write output to FILE; default is to plot to screen')
    (options, args) = parser.parse_args()
    if len(args) < 2:
        sys.stderr.write('You must specify at least two arguments: the input '
                         'file name and a stand.\n')
        sys.exit(2)
    
    inFileName = args[0]
    stds = [int(arg) for arg in args[1:]]
    
    if options.outfile:
        import matplotlib.figure
        import matplotlib.backends.backend_agg
    else:
        import matplotlib.pyplot as plt
    
    tbfdata = TbfData(inFileName)
    tbfdata.plotMany(stds, filename = options.outfile)
