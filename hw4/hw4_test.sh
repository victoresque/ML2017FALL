#!/bin/bash 
mkdir model
wget -O model/m1.h5 'https://www.dropbox.com/s/e1sguq88602to8k/m1.h5?dl=1'
wget -O model/m2.h5 'https://www.dropbox.com/s/y4i677bgp7r6028/m2.h5?dl=1'
wget -O model/m3.h5 'https://www.dropbox.com/s/e2ztfigeeosak7t/m3.h5?dl=1'
wget -O model/m4.h5 'https://www.dropbox.com/s/b6rjqtgpbyu02ot/m4.h5?dl=1'
wget -O model/m5.h5 'https://www.dropbox.com/s/3i5j2u9op04lf9n/m5.h5?dl=1'
wget -O model/m6.h5 'https://www.dropbox.com/s/cgoluogold2aerc/m6.h5?dl=1'
wget -O model/m7.h5 'https://www.dropbox.com/s/lsbtd6klkioih33/m7.h5?dl=1'
wget -O model/m8.h5 'https://www.dropbox.com/s/o3mtrq39rnx932f/m8.h5?dl=1'
wget -O model/m9.h5 'https://www.dropbox.com/s/gwmin3uh0oqhty1/m9.h5?dl=1'
wget -O model/m10.h5 'https://www.dropbox.com/s/vzffp528suba2x1/m10.h5?dl=1'
wget -O model/cmap.pkl 'https://www.dropbox.com/s/km1aq82mu2zv2k7/cmap.pkl?dl=1'
wget -O model/word2vec.pkl 'https://www.dropbox.com/s/3wd49l3nndvvfsv/word2vec.pkl?dl=1'
python test.py $1 $2
