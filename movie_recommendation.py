'''script that suggests movies based on three movie inputs'''

import numpy as np
import pandas as pd
import pickle
import Levenshtein
from fuzzywuzzy import process


def main():
    print_header()
    file_path = "C:\\Users\\Lillian\\Documents\\bootcamp\\ml-latest-small\\"
    movies, ratings = get_files(file_path)
    movie_1 = input("Please provide a movie title: ")
    movie_2 = input("Please provide another movie title: ")
    movie_3 = input("Please provide another movie title: ")
    movie_list = fuzzy_lookup(movie_1, movie_2, movie_3, movies)
    ratings_matrix = create_matrix(ratings)
    id_list = get_ids(movie_list, movies)
    user_list = add_user(ratings_matrix, id_list)
    pickle_in = open(file_path + 'NMF.pickle', 'rb')
    ranking = predict_recs(pickle_in, ratings_matrix, user_list)
    print()
    print("Here are your recommendations: ")
    print()
    result = print_output(ranking, movies)
    print(result.head(20))


def print_header():
    '''header for recommender'''
    print('----------------------------')
    print('    Movies Recommender')
    print('----------------------------')
    print()


def get_files(file_path):
    '''returns files containing movies and ratings'''
    movies = pd.read_csv(file_path + "movies.csv")
    ratings = pd.read_csv(file_path + "ratings.csv")
    ratings = ratings.drop(['timestamp'], axis=1)
    return movies, ratings


def fuzzy_lookup(movie_1, movie_2, movie_3, movies):
    '''matches movie titles and movie inputs despite misspellings, etc'''
    movie_titles = movies['title'].tolist()
    movie_1c = process.extractOne(movie_1, movie_titles)[0]
    movie_2c = process.extractOne(movie_2, movie_titles)[0]
    movie_3c = process.extractOne(movie_3, movie_titles)[0]
    movie_list = [movie_1c, movie_2c, movie_3c]
    return movie_list


def create_matrix(ratings):
    ''' create a sparse user_rating x movies matrix '''
    ratings_matrix = ratings.pivot('movieId', 'userId')
    ratings_matrix.fillna(0.0, inplace=True)
    return ratings_matrix


def get_ids(movie_list, movies):
    ''' return the movie IDs for the movies listed by user '''
    id_list = []
    for i in movie_list:
        a = movies['movieId'][movies['title'] == i]
        id_list.append(a)
    return id_list


def add_user(ratings_matrix, id_list):
    '''input user preferences into ratings matrix'''
    user_list = pd.Series(
        np.zeros(ratings_matrix.shape[0]), index=ratings_matrix.index)
    for i in id_list:
        user_list[i] = 5
    return user_list


def predict_recs(pickle_in, ratings_matrix, user_list):
    '''uses pre-trained model for prediction'''
    model = pickle.load(pickle_in)
    H = model.components_
    W = model.transform(ratings_matrix)
    profile = np.dot(user_list, W)
    ranking = np.dot(profile, W.T)
    ranking = pd.Series(ranking, index=ratings_matrix.index)
    return ranking


def print_output(ranking, movies):
    '''display movie recommendations'''
    result = pd.DataFrame({'title': movies['title'], 'rank': ranking})
    result.sort_values('rank', ascending=False, inplace=True)
    return result


if __name__ == '__main__':
    main()
