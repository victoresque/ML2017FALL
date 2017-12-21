import sys
import numpy as np
from util import *
from model import *

movieToGenres = parseMovies(sys.argv[3])
userDict = parseUsers(sys.argv[4])
testData = parseTesting(sys.argv[1])
max_userid = len(userDict)
max_movieid = len(movieToGenres)

Users, Movies = [], []
for user, movie in testData:
    Users.append(user)
    Movies.append(movie)

Users = np.array(Users)
Movies = np.array(Movies)

mean = 3.58171208604
std = 1.11689766115

model = MFModel(max_userid, max_movieid, d=4)
model.load_weights('models/m4.h5')
y = model.predict([Users, Movies], batch_size=512, verbose=1).flatten()
y4 = (y * std) + mean

model = MFModel(max_userid, max_movieid, d=8)
model.load_weights('models/m8.h5')
y = model.predict([Users, Movies], batch_size=512, verbose=1).flatten()
y8 = (y * std) + mean

model = MFModel(max_userid, max_movieid, d=16)
model.load_weights('models/m16.h5')
y = model.predict([Users, Movies], batch_size=512, verbose=1).flatten()
y16 = (y * std) + mean

model = MFModel(max_userid, max_movieid, d=32)
model.load_weights('models/m32.h5')
y = model.predict([Users, Movies], batch_size=512, verbose=1).flatten()
y32 = (y * std) + mean

model = MFModel(max_userid, max_movieid, d=64)
model.load_weights('models/m64.h5')
y = model.predict([Users, Movies], batch_size=512, verbose=1).flatten()
y64 = (y * std) + mean

model = MFModel(max_userid, max_movieid, d=128)
model.load_weights('models/m128.h5')
y = model.predict([Users, Movies], batch_size=512, verbose=1).flatten()
y128 = (y * std) + mean

y = (y4+y8+y16+y32+y64+y128) / 6
savePrediction(y, sys.argv[2])
