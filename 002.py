mplementing Naïve Bayes with scikit-learn

Coding from scratch and implementing your own solutions is the best way to learn
about machine learning models. Of course, you can take a shortcut by directly using
the BernoulliNB module (https://scikit-learn.org/stable/modules/generated/
sklearn.naive_bayes.BernoulliNB.html) from the scikit-learn API:
>>> from sklearn.naive_bayes import BernoulliNB





Let's initialize a model with a smoothing factor (specified as alpha in scikit-learn)
of 1.0, and prior learned from the training set (specified as fit_prior=True in
scikit-learn):
>>> clf = BernoulliNB(alpha=1.0, fit_prior=True)



To train the Naïve Bayes classifier with the fit method, we use the following line of code:
>>> clf.fit(X_train, Y_train)



And to obtain the predicted probability results with the predict_proba method, we
use the following lines of code:
>>> pred_prob = clf.predict_proba(X_test)
>>> print('[scikit-learn] Predicted probabilities:\n', pred_prob)
[scikit-learn] Predicted probabilities:
[[0.07896399 0.92103601]]





Finally, we do the following to directly acquire the predicted class with the predict
method (0.5 is the default threshold, and if the predicted probability of class Y is
greater than 0.5, class Y is assigned; otherwise, N is used):
>>> pred = clf.predict(X_test)
>>> print('[scikit-learn] Prediction:', pred)
[scikit-learn] Prediction: ['Y']






