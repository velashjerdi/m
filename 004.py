Evaluating classification performance


To illustrate this, we can compute the confusion matrix of our Naïve Bayes classifier.
We use the confusion_matrix function from scikit-learn to compute it, but it is
very easy to code it ourselves:
>>> from sklearn.metrics import confusion_matrix
>>> print(confusion_matrix(Y_test, prediction, labels=[0, 1]))
[[ 60 47]
[148 431]]





Let's compute these three measurements using corresponding functions from
scikit-learn, as follows:
>>> from sklearn.metrics import precision_score, recall_score, f1_score
>>> precision_score(Y_test, prediction, pos_label=1)
0.9016736401673641
>>> recall_score(Y_test, prediction, pos_label=1)
0.7443868739205527
>>> f1_score(Y_test, prediction, pos_label=1)
0.815515610217597





On the other hand, the negative (dislike) class can also be viewed as positive,
depending on the context. For example, assign the 0 class as pos_label and we have
the following:
>>> f1_score(Y_test, prediction, pos_label=0)
0.38095238095238093



To obtain the precision, recall, and f1 score for each class, instead of exhausting
all class labels in the three function calls as shown earlier, a quicker way is to call
the classification_report function:
>>> from sklearn.metrics import classification_report
>>> report = classification_report(Y_test, prediction)
>>> print(report)
precision recall f1-score support
0.0 0.29 0.56 0.38 107
1.0 0.90 0.74 0.82 579
micro avg 0.72 0.72 0.72 686
macro avg 0.60 0.65 0.60 686
weighted avg 0.81 0.72 0.75 686





The ROC curve is a plot of the true positive rate versus the false positive rate at
various probability thresholds, ranging from 0 to 1. For a testing sample, if the
probability of a positive class is greater than the threshold, then a positive class
is assigned; otherwise, we use a negative class. To recap, the true positive rate is
equivalent to recall, and the false positive rate is the fraction of negatives that are
incorrectly identified as positive. Let's code and exhibit the ROC curve (under
thresholds of 0.0, 0.1, 0.2, …, 1.0) of our model:
>>> pos_prob = prediction_prob[:, 1]
>>> thresholds = np.arange(0.0, 1.1, 0.05)
>>> true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
>>> for pred, y in zip(pos_prob, Y_test):
... for i, threshold in enumerate(thresholds):
... if pred >= threshold:
... # if truth and prediction are both 1
... if y == 1:
... true_pos[i] += 1
... # if truth is 0 while prediction is 1
... else:
... false_pos[i] += 1
... else:
... break




Then, let's calculate the true and false positive rates for all threshold settings
(remember, there are 516.0 positive testing samples and 1191 negative ones)
>>> n_pos_test = (Y_test == 1).sum()
>>> n_neg_test = (Y_test == 0).sum()
>>> true_pos_rate = [tp / n_pos_test for tp in true_pos]
>>> false_pos_rate = [fp / n_neg_test for fp in false_pos]






Now, we can plot the ROC curve with Matplotlib:
>>> import matplotlib.pyplot as plt
>>> plt.figure()
>>> lw = 2
>>> plt.plot(false_pos_rate, true_pos_rate,
... color='darkorange', lw=lw)
>>> plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
>>> plt.xlim([0.0, 1.0])
>>> plt.ylim([0.0, 1.05])
>>> plt.xlabel('False Positive Rate')
>>> plt.ylabel('True Positive Rate')
>>> plt.title('Receiver Operating Characteristic')
>>> plt.legend(loc="lower right")
>>> plt.show(






In the graph, the dashed line is the baseline representing random guessing, where
the true positive rate increases linearly with the false positive rate; its AUC is 0.5.
The solid line is the ROC plot of our model, and its AUC is somewhat less than 1.
In a perfect case, the true positive samples have a probability of 1, so that the ROC
starts at the point with 100% true positive and 0% false positive. The AUC of such a
perfect curve is 1. To compute the exact AUC of our model, we can resort to the roc_
auc_score function of scikit-learn:
>>> from sklearn.metrics import roc_auc_score
>>> roc_auc_score(Y_test, pos_prob)
0.6857375752586637