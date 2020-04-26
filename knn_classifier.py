from utils.generating_dataset import get_train_test_split_data
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


X_train, X_test, y_train, y_test = get_train_test_split_data(100)



k = [1, 3, 5] #different k nn values
p = ['manhattan', 'euclidean', 'chebyshev'] #different distance calculations

for ik, kv in enumerate(k):
    for ip, pv in enumerate(p):

        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('knn', KNeighborsClassifier(algorithm='brute', n_neighbors=kv, metric=pv))
        ])

        grid_search = GridSearchCV(pipe,
                                   [{}],
                                   cv=StratifiedKFold(n_splits=5,
                                                      random_state=42).split(X_train, y_train),
                                   verbose=2)
        model = grid_search.fit(X_train, y_train)

        y_preds = model.predict(X_test)
        report = classification_report(y_test, y_preds)
        print(report)


# with k = [ 1 , 3, 5 ]
# trained model with truncated svd with 100 features.
# Best scores with k=5

#classification report K=1 and distance manhattan
#              precision    recall  f1-score   support
#
#            0       0.24      0.21      0.22       164
#            1       0.90      0.89      0.89      1905
#            2       0.67      0.73      0.70       410
#
#     accuracy                           0.82      2479
#    macro avg       0.60      0.61      0.61      2479
# weighted avg       0.82      0.82      0.82      2479\


# classification report k=1 and distance eculidean
#               precision    recall  f1-score   support
#
#            0       0.24      0.20      0.21       164
#            1       0.88      0.88      0.88      1905
#            2       0.64      0.69      0.66       410
#
#     accuracy                           0.80      2479
#    macro avg       0.59      0.59      0.59      2479
# weighted avg       0.80      0.80      0.80      2479


# classification report k=1 and distance chebyshev
#               precision    recall  f1-score   support
#
#            0       0.15      0.13      0.14       164
#            1       0.86      0.85      0.86      1905
#            2       0.54      0.60      0.57       410
#
#     accuracy                           0.76      2479
#    macro avg       0.52      0.53      0.52      2479
# weighted avg       0.76      0.76      0.76      2479


# classification report k=3 and distance manhattan
#               precision    recall  f1-score   support
#
#            0       0.33      0.22      0.26       164
#            1       0.90      0.93      0.92      1905
#            2       0.73      0.72      0.72       410
#
#     accuracy                           0.85      2479
#    macro avg       0.65      0.62      0.63      2479
# weighted avg       0.83      0.85      0.84      2479


# classification report k=3 and distance eculidean
#               precision    recall  f1-score   support
#
#            0       0.32      0.20      0.25       164
#            1       0.90      0.92      0.91      1905
#            2       0.71      0.70      0.71       410
#
#     accuracy                           0.84      2479
#    macro avg       0.64      0.61      0.62      2479
# weighted avg       0.83      0.84      0.83      2479


# classification report k=3 and distance chebyshev
#               precision    recall  f1-score   support
#
#            0       0.16      0.10      0.12       164
#            1       0.85      0.89      0.87      1905
#            2       0.58      0.57      0.57       410
#
#     accuracy                           0.78      2479
#    macro avg       0.53      0.52      0.52      2479
# weighted avg       0.76      0.78      0.77      2479


# classification report k=5 and distance manhattan
#               precision    recall  f1-score   support
#
#            0       0.38      0.18      0.24       164
#            1       0.89      0.94      0.91      1905
#            2       0.74      0.68      0.71       410
#
#     accuracy                           0.85      2479
#    macro avg       0.67      0.60      0.62      2479
# weighted avg       0.83      0.85      0.83      2479


# classification report k=5 and distance eculidean
#               precision    recall  f1-score   support
#
#            0       0.44      0.20      0.28       164
#            1       0.88      0.94      0.91      1905
#            2       0.73      0.65      0.68       410
#
#     accuracy                           0.84      2479
#    macro avg       0.68      0.60      0.62      2479
# weighted avg       0.82      0.84      0.83      2479


# classification report k=5 and distance chebyshev
#               precision    recall  f1-score   support
#
#            0       0.31      0.10      0.16       164
#            1       0.85      0.92      0.88      1905
#            2       0.61      0.54      0.57       410
#
#     accuracy                           0.80      2479
#    macro avg       0.59      0.52      0.54      2479
# weighted avg       0.77      0.80      0.78      2479