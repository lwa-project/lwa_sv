#!/bin/bash

cdir=`pwd`

# Move into the correct directory
mkdir -p /home/adp/tbfspecs/
cd /home/adp/tbfspecs/

# Build the data set
/home/adp/lwa_sv/scripts/convertTBF2PASI.py /data1/health_*

# Find the latest check that was processed
latestCheck=`ls -t *_antennas.txt | head -n1 `

# Make sure we are up to date
for dat in *.dat; do
    txt=${dat%%_tbfspecs.dat}_antennas.txt
    if [ -e $txt ]; then continue; fi
    echo -e "\nPerforming antenna health test for $dat ..."
    python /home/adp/lwa_sv/scripts/AntennaHealthTest.py $dat $txt
done

latestCheck2=`ls -t *_antennas.txt | head -n1 `
if [[ "${latestCheck2}" != "${latestCheck}" ]]; then
        echo -e "\nAnalyzing ${latestCheck2}..."
       /home/adp/lwa_sv/scripts/checkStatusChanges.sh ${latestCheck2}
fi

cd ${cdir}
