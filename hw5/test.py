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

model = MFModel(max_userid, max_movieid)
model.load_weights('models/m64.h5')

y = model.predict([Users, Movies], batch_size=512, verbose=1).flatten()
y = (y * std) + mean
savePrediction(y, sys.argv[2])
