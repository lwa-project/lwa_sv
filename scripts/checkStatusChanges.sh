#!/bin/bash

SSMIF=~/tbfspecs/SSMIF_CURRENT.txt

echo "checking "$1" against current SSMIF"
grep " B " $1 > thelist
n=`cat thelist | wc -l`
echo "There appear to be "$n" bad dipoles."
count=0
badx=0
bady=0

while [ $n -gt 0 ] 
do
    stand=`tail -$n thelist | gawk '{print ($2+0); exit(0)}'`
    pol=`tail -$n thelist | gawk '{print ($3+0); exit(0)}'`
    if [[ $pol == 0 ]]; then
        let "badx = $badx + 1" 
    fi
    if [[ $pol == 1 ]]; then
        let "bady = $bady + 1" 
    fi
#    echo "Bad stand and pol", $stand, $pol
    dipole=$(( 2*$stand + $pol - 1))
    dstring=$(printf "ANT_STAT["%d"]" $dipole)
#    echo "Bad stand and pol", $stand, $pol "= "$dipole
    grep -F $dstring $SSMIF > statline
    linesize=`wc -l statline | gawk '{print ($1+0); exit(0)}'`
#    echo linesize is $linesize, dipole $dipole
#    cat statline
    statcode=`cat statline | gawk '{print ($2+0); exit(0)}'`
    if [[ $linesize == 0 ]]; then 
       statcode=3
#       echo not found in ssmif
    fi
    if [[ $statcode == 0 ]]; then 
       let "count = $count + 1"
#       echo "found in SSMIF" 
#       echo "Bad stand and pol", $stand, $pol "= "$dipole
#       cat statline
    fi
    if [[ $statcode == 1 ]]; then 
       let "count = $count + 1"
#       echo "Bad stand and pol", $stand, $pol "= "$dipole
#       cat statline
#       echo "found in SSMIF" 
    fi
    if [[ $statcode == 2 ]]; then 
       echo "Bad stand and pol", $stand, $pol "same as dipole "$dipole
       cat statline
       echo "shows suspect in SSMIF"
    fi
    if [[ $statcode == 3 ]]; then 
#       echo "code is" $statcode
       echo "Bad stand and pol", $stand, $pol "same as dipole "$dipole
       cat statline
       echo "shows good in SSMIF"
    fi
    let "n = $n - 1" 
done

echo "Found "$count" dipoles in SSMIF"
echo "Found "$badx" bad x pol dipoles in SSMIF"
echo "Found "$bady" bad y pol dipoles in SSMIF"

grep " G " $1 > thelist
grep " S " $1 >> thelist
sort -n -k2 thelist > sortlist
mv sortlist thelist

n=`cat thelist | wc -l`
nn=`cat thelist | wc -l`
echo "There appear to be "$n" good dipoles."
echo "Now plotting up first 9 good dipoles"
count=0
goodx=0
goody=0
goodboth=0
laststand=0

while [ $n -gt 0 ] 
do
    stand=`tail -$n thelist | gawk '{print ($2+0); exit(0)}'`
    pol=`tail -$n thelist | gawk '{print ($3+0); exit(0)}'`
    if [[ $pol == 0 ]]; then
        let "goodx = $goodx + 1" 
    fi
    if [[ $pol == 1 ]]; then
        let "goody = $goody + 1" 
    fi

    if [[ $stand == $laststand ]]; then
        let "goodboth = $goodboth + 1" 
    fi
    let "laststand = $stand"
    let "count = $count + 1" 
    let "n = $n - 1" 
done

# 1 19 91 258
good1=225
good2=202
good3=204
good4=253
good5=255
good6=215
good7=35
good8=228
good9=208

echo "Found "$goodx" good x pol dipoles in SSMIF"
echo "Found "$goody" good y pol dipoles in SSMIF"
echo "Found "$goodboth" good both dipoles in SSMIF"

plotfile=${1:0:13}"_tbfspecs.dat"
outfile=${1:0:13}"_tbfspecs.png"
`/home/adp/lwa_sv/scripts/tbfplot.py -o $outfile $plotfile $good1 $good2 $good3 $good4 $good5 $good6 $good7 $good8 $good9`
echo generating plot $outfile
`/home/adp/lwa_sv/scripts/postSpectra.py $outfile`
echo posting image $outfile to lwalab
rm statline thelist
