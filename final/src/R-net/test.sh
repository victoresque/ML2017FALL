#!/bin/bash 
mkdir -p data
mkdir -p checkpoint
wget -O checkpoint/checkpoint.pth.tar 'https://www.dropbox.com/s/n1x20x6403rx3zo/checkpoint.pth.tar?dl=1'
wget -O checkpoint/model_best.pth.tar 'https://www.dropbox.com/s/nshlz20dz32w501/model_best.pth.tar?dl=1'
wget -O data/dict.txt.big 'https://www.dropbox.com/s/rnqgwd6tm89rv5i/dict.txt.big?dl=1'
wget -O data/zh.bin 'https://www.dropbox.com/s/y6pw3x2i4evnk85/zh.bin?dl=1'
wget -O data/zh.bin.syn0.npy 'https://www.dropbox.com/s/7npyko1nu8ufqt8/zh.bin.syn0.npy?dl=1'
wget -O data/zh.bin.syn1neg.npy 'https://www.dropbox.com/s/r017t4wr0ejdi70/zh.bin.syn1neg.npy?dl=1'
python test.py
