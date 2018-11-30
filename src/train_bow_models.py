"""Module to train classifiers using bag-of-words as features."""
import itertools

from scipy.stats import randint, uniform
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from xgboost import XGBClassifier


# TODO: pickle models to disk to avoid re-computing
# TODO: print best parameter after tuning hyperparameters

def evaluate(y_pred, y_true):
    """Prints the scores."""
    pr = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    print('precision = {}, recall = {}, f-score = {}'.format(
        round(pr, 2), round(r, 2), round(f, 2)))


def naive_bayes(train, test, model_path):
    """Trains a Naive Bayes classifier and prints the evaluation metrics. If model is provided,
    it skips training and reports the scores using the trained model.

    Args
    ----
    train: tuple of training (text, labels)
    test: tuple of testing (text, labels)
    model_path: pathlib.Path, path to a previously trained model, if it exists
    """
    if not model_path.exists():
        clf = GaussianNB()
        clf.fit(train[0].toarray(), train[1])
        # write to model_path
        joblib.dump(clf, model_path)
    else:
        clf = joblib.load(model_path)
    y_pred = clf.predict(test[0].toarray())
    evaluate(test[1], y_pred)


def svm(train, test, model_path):
    """Trains a SVM with linear kernel and prints the evaluation metrics. If model is provided,
    it skips training and reports the scores using the trained model.

    Args
    ----
    train: tuple of training (text, labels)
    test: tuple of testing (text, labels)
    model_path: pathlib.Path, path to a previously trained model, if it exists
    """
    if not model_path.exists():
        clf = LinearSVC()
        params = {'C': [1.0, 0.5, 0.1, 10.0]}
        gs = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=2, scoring='f1_macro')
        gs.fit(train[0], train[1])
        joblib.dump(gs, model_path)
    else:
        gs = joblib.load(model_path)
    y_pred = gs.predict(test[0].toarray())
    print('Best params: ', gs.best_params_)
    evaluate(test[1], y_pred)


def xgboost(train, test, model_path):
    """Trains a XGBoost model and prints evaluation metrics.If model is provided,
    it skips training and reports the scores using the trained model.

    Args
    ----
    train: tuple of training (text, labels)
    test: tuple of testing (text, labels)
    model_path: pathlib.Path, path to a previously trained model, if it exists
    """
    if not model_path.exists():
        clf = XGBClassifier(objective='binary:logistic')
        param_dist = {'max_depth': randint(3, 10), 'learning_rate': uniform(0.1, 0.5)}
        rcv = RandomizedSearchCV(clf, param_dist, n_iter=20, cv=5, n_jobs=2, scoring='f1_macro')
        rcv.fit(train[0], train[1])
        joblib.dump(rcv, model_path)
    else:
        rcv = joblib.load(model_path)
    y_pred = rcv.predict(test[0])
    print('Best params: ', rcv.best_params_)
    evaluate(test[1], y_pred)


def mlp(train, test, model_path):
    """Trains a MLP model and prints evaluation metrics.If model is provided,
    it skips training and reports the scores using the trained model.

    Args
    ----
    train: tuple of training (text, labels)
    test: tuple of testing (text, labels)
    model_path: pathlib.Path, path to a previously trained model, if it exists
    """
    if not model_path.exists():
        clf = MLPClassifier(early_stopping=True)
        params = {'hidden_layer_sizes': list(itertools.product([3, 4], [20, 50, 100])),
                  'learning_rate': ['adaptive']}
        gs = GridSearchCV(clf, params, cv=5, n_jobs=2, scoring='f1_macro')
        gs.fit(train[0], train[1])
        joblib.dump(gs, model_path)
    else:
        gs = joblib.load(model_path)
    y_pred = gs.predict(test[0])
    print('Best params: ', gs.best_params_)
    evaluate(test[1], y_pred)
