import pickle
import pandas as pd
from pipeline import pipe
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('data/train.csv')

X = train.drop(columns=['TARGET'], axis=1)
y = train['TARGET']

param_grid = {
    #'classifier__max_iter': [100, 200],
    #'classifier__max_depth': [3, 5],
    #'classifier__learning_rate': [0.1, 0.01],
    'classifier__class_weight': ['balanced', None]
}

optimized = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=5,
    n_jobs=8,
    verbose=1,
    scoring='roc_auc'
)

optimized.fit(X, y)

with open('models/model.pkl', 'wb') as file:
    pickle.dump(optimized.best_estimator_, file)