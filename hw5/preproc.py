import pandas as pd

def parseMovies(path):
    genreDict = {}
    movieToGenres = []
    with open(path, 'r', encoding='ISO-8859-1') as f:
        next(f)
        for i, line in enumerate(f):
            genres = line.split('::')[2].split('|')
            for j, genre in enumerate(genres):
                genreDict[genre] = len(genreDict)
                genres[j] = genreDict[genre]
            movieToGenres.append(genres)
    return movieToGenres

def parseUsers(path):
    userDict = {}
    with open(path, 'r', encoding='ISO-8859-1') as f:
        next(f)
        for i, line in enumerate(f):
            line = line.split('::')
            userid, gender, age = int(line[0]), line[1], int(line[2])
            userDict[userid] = {'gender':gender, 'age':age}
    return userDict

def parseReview(path):
    userReviews = {}
    rawReview = pd.read_csv(path).drop(['TrainDataID'], axis=1).values
    for review in rawReview:
        userid, movie, rating = review[0:3]
        if userid not in userReviews:
            userReviews[userid] = []
        userReviews[userid].append((movie, rating))
    return userReviews

def parseTesting(path):
    testData = []
    rawTest = pd.read_csv(path).drop(['TestDataID'], axis=1).values
    for test in rawTest:
        userid, movie = test[0:2]
        testData.append((userid, movie))
    return testData
