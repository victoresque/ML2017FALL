#!/bin/bash 
wget -O cnn.h5 'https://www.dropbox.com/s/45esimbgcpvild3/0.67623.h5?dl=1'
python test.py $1 $2
