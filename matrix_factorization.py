'''trains model for movie recommender'''

import pandas as pd
import pickle
from sklearn.decomposition import NMF


def main(n_comps):
    file_path = "C:\\Users\\Lillian\\Documents\\bootcamp\\ml-latest-small\\"
    ratings = get_data(file_path)
    run_model(n_comps, ratings, file_path)


def get_data(file_path):
    '''creates a sparse matrix from the ratings file'''
    df = pd.read_csv(file_path + "ratings.csv")
    del df['timestamp']
    ratings = df.pivot('movieId', 'userId')
    ratings.fillna(0.0, inplace=True)
    return ratings


def run_model(n_comps, ratings, file_path):
    ''' trains model on provided data'''
    model = NMF(n_components=n_comps, init='nndsvd', random_state=42, alpha=0)
    model.fit(ratings)
    with open(file_path + "NMF.pickle", 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
