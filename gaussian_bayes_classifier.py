from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from utils.generating_dataset import get_train_test_split_data
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report




X_train, X_test, y_train, y_test = get_train_test_split_data(100)

pipe = Pipeline([
        ('gaussian', GaussianNB())
    ])
grid_search = GridSearchCV(pipe,
                           [{}],
                           cv=StratifiedKFold(n_splits=5,
                                              random_state=1).split(X_train, y_train),
                           verbose=2)
model = grid_search.fit(X_train, y_train)

y_preds = model.predict(X_test)
report = classification_report( y_test, y_preds )
print(report)



#classification report
#              precision    recall  f1-score   support
#
#            0       0.11      0.35      0.16       164
#            1       0.90      0.70      0.79      1905
#            2       0.56      0.61      0.59       410
#
#     accuracy                           0.66      2479
#    macro avg       0.52      0.55      0.51      2479
# weighted avg       0.79      0.66      0.71      2479
