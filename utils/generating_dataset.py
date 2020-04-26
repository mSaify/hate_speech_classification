import pandas as pd
from .preprocessing import truncated_svd, tfidf_vector
from sklearn.model_selection import train_test_split



def reading_data():
    df = pd.read_csv("hate_speech.csv")

    df.loc[1]['tweet']
    #df.loc[1]
    df.count
    df['class'].hist()
    df.describe()


    xy = df[df['hate_speech'] >= 6]
    xy = list(xy['tweet'])
    for y in xy:
      print(y)

    return df




def get_train_test_split_data(truncated_svd_size=1000):
    df = pd.read_csv("hate_speech.csv")
    tweets = df.tweet

    # get truncated svd with 100 features
    tfidf_trunc = truncated_svd(tfidf_vector(tweets=tweets), truncated_svd_size)

    # split train and test dataset with 10% test data
    X = pd.DataFrame(tfidf_trunc)
    y = df['class'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    return X_train, X_test, y_train, y_test