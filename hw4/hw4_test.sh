#!/bin/bash 
wget -O model/m1.h5 ''
wget -O model/m2.h5 ''
wget -O model/m3.h5 ''
wget -O model/m4.h5 ''
wget -O model/m5.h5 ''
wget -O model/m6.h5 ''
wget -O model/m7.h5 ''
wget -O model/m8.h5 ''
python test.py $1 $2
