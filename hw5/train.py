import numpy as np
from preproc import *
from util import *
from model import *

movieToGenres = parseMovies('data/movies.csv')
userDict = parseUsers('data/users.csv')
userReviews = parseReview('data/train.csv')
max_userid = len(userDict)
max_movieid = len(movieToGenres)

model = MFModel(max_userid, max_movieid, 100)
opt = optimizers.adamax()
model.compile(loss='mse', optimizer=opt)
model.summary()

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

Users = np.array(Users)
Movies = np.array(Movies)
Ratings = np.array(Ratings)
Ratings = (Ratings - np.mean(Ratings)) / np.std(Ratings) # expected mse = 0.6

callbacks = [EarlyStopping('val_loss', patience=2),
             ModelCheckpoint('model_{val_loss:.4f}.h5', save_best_only=True, save_weights_only=False)]
model.fit([Users, Movies], Ratings, epochs=30,
          validation_split=.1, verbose=1, callbacks=callbacks)
model.save('mf.h5')