import numpy as np
from util import *
from preproc import *
from model import *

movieToGenres = parseMovies('data/movies.csv')
userDict = parseUsers('data/users.csv')
userReviews = parseReview('data/train.csv')
testData = parseTesting('data/test.csv')
max_userid = len(userDict)
max_movieid = len(movieToGenres)

userReviewsList = []
for i in range(1, max_userid+1):
    for movie, rating in userReviews[i]:
        userReviewsList.append((i, movie, rating))
np.random.shuffle(userReviewsList)

Users, Movies, Ratings = [], [], []
for i, movie, rating in userReviewsList:
    Users.append(i)
    Movies.append(movie)
    Ratings.append(rating)

Ratings = np.array(Ratings)
mean = np.mean(Ratings)
std = np.std(Ratings)

print(mean)
print(std)

Users, Movies = [], []
for user, movie in testData:
    Users.append(user)
    Movies.append(movie)

Users = np.array(Users)
Movies = np.array(Movies)

model = MFModel(max_userid, max_movieid, 100)
model.load_weights('model_0.6064.h5')

y = model.predict([Users, Movies], batch_size=512, verbose=1).flatten()
y = (y * std) + mean
savePrediction(y, 'result/prediction.csv')
