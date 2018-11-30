from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif


def word_counts(texts):
    """Vectorize sentences into tokens after removing stopwords."""
    vec = CountVectorizer(analyzer='word', stop_words='english', min_df=2)
    X_train = vec.fit_transform(texts)
    return X_train


def ngram_features(train_msg, val_msg):
    """Build n-gram feature vectors, using bag-of-words model. Applies the vectorizer
    learnt on train_msg to val_msg.

    Args
    ----
    train_msg: training message
    val_msg: validation message

    Returns
    -------
    X_train, X_val
    """
    vec = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 2),
                          min_df=2)

    X_train = vec.fit_transform(train_msg)
    X_val = vec.transform(val_msg)

    return X_train, X_val


def select_best_features(X_train, y_train, X_val, K=15000):
    """
    Selects the top K features using SelectKBest. Applies the selector learnt on
    X_train to X_val.
    """
    feature_selector = SelectKBest(f_classif, k=min(X_train.shape[1], K))
    X_train_new = feature_selector.fit_transform(X_train, y_train)
    X_val_new = feature_selector.transform(X_val)
    return X_train_new, X_val_new
