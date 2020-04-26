from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from utils.generating_dataset import get_train_test_split_data

from sklearn.metrics import classification_report


X_train, X_test, y_train, y_test = get_train_test_split_data(100)


# Set the parameters
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10, 100],
                     'C': [0.01, 0.1, 1, 10, 100]}]


grid_search = GridSearchCV(SVC(), tuned_parameters, cv=StratifiedKFold(n_splits=5,
                                          random_state=1).split(X_train, y_train),
                           verbose=2)

model = grid_search.fit(X_train, y_train)


y_preds = model.predict(X_test)
report = classification_report( y_test, y_preds )
print(report)

# gamma {0.01, 0. 1, 1, 10, 100 } C {0.01, 0.1, 1, 10, 100}
# trained model with truncated svd with 100 features.
# best score with gamma = 0.01 and c = 100

#               precision    recall  f1-score   support
#
#            0       0.41      0.35      0.38       164
#            1       0.89      0.91      0.90      1905
#            2       0.75      0.75      0.75      410
#
#     accuracy                           0.84      2479
#    macro avg       0.68      0.67      0.68      2479
# weighted avg       0.83      0.84      0.83      2479