#! /bin/bash
for d in 1
    do 
    if [ ! -d "$d" ]
        then
        mkdir $d
        fi
    # number of sensor vs coverage
    t=10
    for sensor in 500 520 660 700 720 920
        do 
        echo "200 $t 50 $sensor" | python3 sim_getdata.py | tee "$d/SvsC$sensor.dat"
        done
    for csize in 130 140 160 200 210 220 240 
        do
        hsize=$(echo "$csize / 10 * 3" | bc)
        echo "$csize $t $hsize 800" | python3 sim_getdata.py | tee "$d/CvsC$csize.dat"
        done
    done