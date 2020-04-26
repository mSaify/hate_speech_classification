
from .utils.generating_dataset import get_train_test_split_data
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report


X_train, X_test, y_train, y_test = get_train_test_split_data(1000)



#model definition
pipe = Pipeline(
        [('select', SelectFromModel(LogisticRegression(class_weight='balanced',
                                                  penalty="l1", C=0.01))),
        ('model', LogisticRegression(class_weight='balanced'))])

#5 fold split
grid_search = GridSearchCV(pipe,
                           [{}],
                           cv=StratifiedKFold(n_splits=5,
                                              random_state=42).split(X_train, y_train),
                           verbose=2)

model = grid_search.fit(X_train, y_train)
y_preds = model.predict(X_test)


report = classification_report( y_test, y_preds )
print(report)


#classification report

#               precision    recall  f1-score   support
# #
# #            0       0.45      0.57      0.50       164
# #            1       0.96      0.91      0.93      1905
# #            2       0.82      0.94      0.88       410
# #
# #     accuracy                           0.89      2479
# #    macro avg       0.74      0.81      0.77      2479
# # weighted avg       0.91      0.89      0.90      2479


#logistic regression gave the highest accuracy and the algorithm ran in minutes.