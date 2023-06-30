Building a movie recommender with
Naïve Bayes


First, we import all the necessary modules and variables:
>>> import numpy as np
>>> from collections import defaultdict
>>> data_path = 'ml-1m/ratings.dat'
>>> n_users = 6040
>>> n_movies = 3706




We then develop the following function to load the rating data from ratings.dat:
>>> def load_rating_data(data_path, n_users, n_movies):
... """
... Load rating data from file and also return the number of
... ratings for each movie and movie_id index mapping
... @param data_path: path of the rating data file
... @param n_users: number of users
... @param n_movies: number of movies that have ratings
... @return: rating data in the numpy array of [user, movie];
... movie_n_rating, {movie_id: number of ratings};
... movie_id_mapping, {movie_id: column index in
... rating data}
... """
... data = np.zeros([n_users, n_movies], dtype=np.float32)
... movie_id_mapping = {}
... movie_n_rating = defaultdict(int)
... with open(data_path, 'r') as file:
... for line in file.readlines()[1:]:
... user_id, movie_id, rating, _ = line.split("::")
... user_id = int(user_id) - 1
... if movie_id not in movie_id_mapping:
... movie_id_mapping[movie_id] =
... len(movie_id_mapping)
... rating = int(rating)
... data[user_id, movie_id_mapping[movie_id]] = rating
.. if rating > 0:
... movie_n_rating[movie_id] += 1
... return data, movie_n_rating, movie_id_mapping






And then we load the data using this function:
>>> data, movie_n_rating, movie_id_mapping =
... load_rating_data(data_path, n_users, n_movies)




It is always recommended to analyze the data distribution. We do the following:
>>> def display_distribution(data):
... values, counts = np.unique(data, return_counts=True)
... for value, count in zip(values, counts):
... print(f'Number of rating {int(value)}: {count}')
>>> display_distribution(data)
Number of rating 0: 21384032
Number of rating 1: 56174
Number of rating 2: 107557
Number of rating 3: 261197
Number of rating 4: 348971
Number of rating 5: 226309





As you can see, most ratings are unknown; for the known ones, 35% are of rating 4,
followed by 26% of rating 3, and 23% of rating 5, and then 11% and 6% of ratings 2
and 1, respectively.
Since most ratings are unknown, we take the movie with the most known ratings as
our target movie:
>>> movie_id_most, n_rating_most = sorted(movie_n_rating.items(),
... key=lambda d: d[1], reverse=True)[0]
>>> print(f'Movie ID {movie_id_most} has {n_rating_most} ratings.')
Movie ID 2858 has 3428 ratings.






The movie with ID 2858 is the target movie, and ratings of the rest of the movies are
signals. We construct the dataset accordingly:
>>> X_raw = np.delete(data, movie_id_mapping[movie_id_most],
... axis=1)
>>> Y_raw = data[:, movie_id_mapping[movie_id_most]]





We discard samples without a rating in movie ID 2858:
>>> X = X_raw[Y_raw > 0]
>>> Y = Y_raw[Y_raw > 0]
>>> print('Shape of X:', X.shape)
Shape of X: (3428, 3705)
>>> print('Shape of Y:', Y.shape)
Shape of Y: (3428,)




Again, we take a look at the distribution of the target movie ratings:
>>> display_distribution(Y)
Number of rating 1: 83
Number of rating 2: 134
Number of rating 3: 358
Number of rating 4: 890
Number of rating 5: 1963



We can consider movies with ratings greater than 3 as being liked (being
recommended):
>>> recommend = 3
>>> Y[Y <= recommend] = 0
>>> Y[Y > recommend] = 1
>>> n_pos = (Y == 1).sum()
>>> n_neg = (Y == 0).sum()
>>> print(f'{n_pos} positive samples and {n_neg} negative
... samples.')
2853 positive samples and 575 negative samples.





As a rule of thumb in solving classification problems, we need to always analyze the
label distribution and see how balanced (or imbalanced) the dataset is.
Next, to comprehensively evaluate our classifier's performance, we can randomly
split the dataset into two sets, the training and testing sets, which simulate learning
data and prediction data, respectively. Generally, the proportion of the original
dataset to include in the testing split can be 20%, 25%, 33.3%, or 40%. We use
the train_test_split function from scikit-learn to do the random splitting and to
preserve the percentage of samples for each class:
>>> from sklearn.model_selection import train_test_split
>>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
... test_size=0.2, random_state=42







We check the training and testing sizes as follows:
>>> print(len(Y_train), len(Y_test))
2742 686



Next, we train a Naïve Bayes model on the training set. You may notice that the
values of the input features are from 0 to 5, as opposed to 0 or 1 in our toy example.
Hence, we use the MultinomialNB module (https://scikit-learn.org/stable/
modules/generated/sklearn.naive_bayes.MultinomialNB.html) from scikit-learn
instead of the BernoulliNB module, as MultinomialNB can work with integer
features. We import the module, initialize a model with a smoothing factor of 1.0
and prior learned from the training set, and train this model against the training set
as follows:
>>> from sklearn.naive_bayes import MultinomialNB
>>> clf = MultinomialNB(alpha=1.0, fit_prior=True)
>>> clf.fit(X_train, Y_train




Then, we use the trained model to make predictions on the testing set. We get the
predicted probabilities as follows:
>>> prediction_prob = clf.predict_proba(X_test)
>>> print(prediction_prob[0:10])
[[7.50487439e-23 1.00000000e+00]
[1.01806208e-01 8.98193792e-01]
[3.57740570e-10 1.00000000e+00]
[1.00000000e+00 2.94095407e-16]
[1.00000000e+00 2.49760836e-25]
[7.62630220e-01 2.37369780e-01]
[3.47479627e-05 9.99965252e-01]
[2.66075292e-11 1.00000000e+00]
[5.88493563e-10 9.99999999e-01]
[9.71326867e-09 9.99999990e-01]]





We get the predicted class as follows:
>>> prediction = clf.predict(X_test)
>>> print(prediction[:10])
[1. 1. 1. 0. 0. 0. 1. 1. 1. 1.]





Finally, we evaluate the model's performance with classification accuracy, which is
the proportion of correct predictions:
>>> accuracy = clf.score(X_test, Y_test)
>>> print(f'The accuracy is: {accuracy*100:.1f}%')
The accuracy is: 71.6%












