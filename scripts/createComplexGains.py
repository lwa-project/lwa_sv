#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Given 3 pointing centers, and custom beam sizes, generate the full set of complex gains for all frequencies in the band for each beam."""

import os
import sys
import argparse
import numpy as np

from lsl.common import stations
from lsl.misc import parser as aph
from lsl.misc import beamformer

from astropy.constants import c as vLight
vLight = vLight.to('m/s').value

from adp.AdpCommon import *

def main(args):

    #Build the station.
    station = stations.lwasv
    antennas = station.antennas
 
    #Set up the antenna locations and find the center of the array.
    xyz = np.array([(a.stand.x, a.stand.y, a.stand.z) for a in antennas]).T
    center = np.mean(xyz[:,:510], axis=1)

    #Set up the frequencies.
    #Center freqs.
    cFreq1 = 60.0e6
    cFreq2 = 75.0e6

    nchan_server = 132
    nservers = 6

    freqs = np.zeros((2, nservers, nchan_server))
    for i, cfreq in enumerate((cFreq1, cFreq2)):
        ch0 = int(round(cfreq / CHAN_BW)) - nservers*nchan_server//2 

        freqs[i,:,:] = ((ch0 + np.arange(nservers*nchan_server)) * CHAN_BW).reshape((nservers, nchan_server))
  
    #Now lets build the complex gains for all frequencies. 
    #shape = (server x beam #/tuning/beam pol x channel x ant pol) (6 x 12 x 132 x 512)
    cgains = np.zeros((freqs.shape[1], 12, nchan_server, 512), dtype=np.complex64)

    for i in range(6): #Loop over servers
        print('Working on server %i...' % (i+1))
        serverFreqs = freqs[:,i,:]

        for j in range(6): #Loop over beams/tunings

            tuning = serverFreqs[j % 2,:]

            x0, y0, theta =  args.azimuths[j // 2], args.elevations[j // 2], args.thetas[j // 2]

            m = 0
            for freq in tuning:
                if theta != 0:
                    #Find the gains first. We'll use a smooth Gaussian taper for now.

                    #Correct for cable attenuation.
                    att = np.sqrt(np.array([a.cable.attenuation(freq) for a in antennas]))

                    #Compute desired perpendicular baseline (r_u) and required parallel baseline (r_v)
                    #for a given pointing such that the projected r_v equals r_u. The line of sight is along the v-axis (y' in the paper).
                    r_u = (vLight/freq / (theta*np.pi/180.0))
                    r_v = r_u / np.sin(y0*np.pi/180.0)

                    #Apply a Gaussian weighting scheme across the dipoles.
                    sigma_u = r_u / (2.0*np.sqrt(2.0*np.log(5.0)))
                    sigma_v = r_v / (2.0*np.sqrt(2.0*np.log(5.0)))

                    #Roation matrix describing the u (x') and v (y') coordinates.   
                    rot = np.array([[np.cos(x0*np.pi/180), -np.sin(x0*np.pi/180.0), 0], [np.sin(x0*np.pi/180), np.cos(x0*np.pi/180), 0], [0, 0, 1]])

                    xyz2 = xyz - np.array([[center[0]], [center[1]], [0]])

                    uvw = np.matmul(rot,xyz2)

                    wgt = np.zeros(len(antennas))

                    for k in range(wgt.size):
                        wgt[k] = att[k]*np.exp(-(uvw[0,k]**2/(2*sigma_u**2) + uvw[1,k]**2/(2*sigma_v**2)))

                    wgt[[k for k,a in enumerate(antennas) if a.combined_status != 33]] = 0.0

                    #Set the weights between 0 and 1.
                    wgt[::2] /= wgt[::2].max() #antenna X pol weights
                    wgt[1::2] /= wgt[1::2].max() #antenna Y pol weights

                    #Compute the delays (nanoseconds) for a custom beam which uses the proper delays for the desired frequency.
                    delays = beamformer.calc_delay(antennas, freq=freq, azimuth=x0, elevation=y0)
                    delays *= 1e9

                    #Convert to the corrective term in order to get all the antennas phased up.
                    delays = delays.max() - delays

                    #Put it all together.
                    if args.fringe != 0:
                        cgains[i,2*j,m,::2] = wgt[::2]*np.exp(-2j*np.pi*(freq/1e9)*delays[::2]) #Beam X
                        cgains[i,2*j,m, 2*(args.fringe-1)] = 5*np.exp(-2j*np.pi*(freq/1e9)*delays[2*(args.fringe-1)]) #Fringing dipole alone on Y pol
                    else:
                        cgains[i,2*j,m,::2] = wgt[::2]*np.exp(-2j*np.pi*(freq/1e9)*delays[::2]) #Beam X
                        cgains[i,2*j+1,m,1::2] = wgt[1::2]*np.exp(-2j*np.pi*(freq/1e9)*delays[1::2]) #Beam Y

                else:
                    cgains[i,2*j:2*(j+1),m,:] = np.zeros((2,512))

                m += 1
    #Add a "pointing" dimension (size 1 for static runs).
    #This will match the shape required by adp_drx.py for dynamic runs.
    cgains = cgains.reshape((1, cgains.shape[0], cgains.shape[1], cgains.shape[2], cgains.shape[3]))

    #Save the files.
    for i in range(6):
        np.savez('/home/adp/complexGains_adp'+str(i+1)+'.npz', cgains=cgains[:,i,:,:,:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Return the full set of complex gains given 3 pointing centers and 3  beam sizes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-a','--azimuths', nargs='+', type=aph.positive_or_zero_float, default=90.0,
        help='Azimuth east of north in degrees for the pointing center (Takes up to 3 numbers)')
    parser.add_argument('-e','--elevations', nargs='+', type=aph.positive_float, default=90.0,
        help='Elevation above the horizon in degrees for the pointing center (Takes up to 3 numbers)')
    parser.add_argument('-t','--thetas', nargs='+', type=aph.positive_or_zero_float, default=5.0,
        help='Shaped beam width in degrees (Takes up to 3 numbers). An entry of 0 will mean that beam will be a normal beam.')
    parser.add_argument('-f', '--fringe', type=int, default=0,
        help='Reference stand for a frining run. (Beam on X pol, fringing dipole on Y pol, 0 = no fringing)')
    args = parser.parse_args()
    main(args)
