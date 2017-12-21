import numpy as np
from util import *
from model import *

movieToGenres = parseMovies('data/movies.csv')
userDict = parseUsers('data/users.csv')
userReviews = parseReview('data/train.csv')
max_userid = len(userDict)
max_movieid = len(movieToGenres)

userReviewsList = []
for i in range(1, max_userid+1):
    for movie, rating in userReviews[i]:
        userReviewsList.append((i, movie, rating))
np.random.shuffle(userReviewsList)

Users, Movies, Ratings, Genres, Genders, Ages = [], [], [], [], [], []
for i, movie, rating in userReviewsList:
    Users.append(i)
    genreEncode = np.zeros((18,))
    for i in range(18):
        if i in movieToGenres[movie]:
            genreEncode[i] = 1
    Genders.append(0 if userDict[i]['gender']=='M' else 1)

    ageEncode = np.zeros((7,))
    ageid = userDict[i]['age']
    if ageid == 1: ageEncode[0] = 1
    elif ageid == 18: ageEncode[1] = 1
    elif ageid == 25: ageEncode[2] = 1
    elif ageid == 35: ageEncode[3] = 1
    elif ageid == 45: ageEncode[4] = 1
    elif ageid == 50: ageEncode[5] = 1
    else: ageEncode[6] = 1
    Ages.append(ageEncode)

    Genres.append(genreEncode)
    Movies.append(movie)
    Ratings.append(rating)

Users = np.array(Users)
Movies = np.array(Movies)
Ratings = np.array(Ratings)
Genres = np.array(Genres)
Genders = np.array(Genders)
Ages = np.array(Ages)

Ratings = (Ratings - np.mean(Ratings)) / np.std(Ratings)

opt = optimizers.adam()
callbacks = [EarlyStopping('val_loss', patience=4),
             ModelCheckpoint('model{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5', save_best_only=True, save_weights_only=False)]

model = MFModel(max_userid, max_movieid, d=256)
model.compile(loss=['mse'], optimizer=opt)
model.summary()

#from keras.utils.vis_utils import plot_model
#plot_model(model, 'model.png')

model.fit([Users, Movies], [Ratings], epochs=200, batch_size=512,
          validation_split=.02, verbose=1, callbacks=callbacks)