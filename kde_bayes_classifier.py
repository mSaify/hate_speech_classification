from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.model_selection import GridSearchCV
from utils.generating_dataset import get_train_test_split_data
from sklearn.model_selection import StratifiedKFold


class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE

    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """

    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel, metric='manhattan', atol=0, rtol=0, breadth_first=True,
                                      leaf_size=40).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self

    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]



X_train, X_test, y_train, y_test = get_train_test_split_data(1000)

grid = GridSearchCV(KDEClassifier(kernel='gaussian'), {'bandwidth': [0.1,1,10]}, refit='True',cv=StratifiedKFold(n_splits=5,
                                              random_state=42).split(X_train, y_train))
grid.fit(X_train, y_train)
y_preds = grid.best_estimator_.predict(X_test)


#classification report
#               precision    recall  f1-score   support
#
#            0       0.07      0.01      0.02       164
#            1       0.78      0.98      0.87      1905
#            2       0.57      0.07      0.13       410
#
#     accuracy                           0.76      2479
#    macro avg       0.47      0.35      0.34      2479
# weighted avg       0.70      0.76      0.69      2479