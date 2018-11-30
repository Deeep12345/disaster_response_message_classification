"""Trains a CNN model using spacy."""
import spacy
from load_data import load_dataset
from spacy.util import minibatch, compounding


def get_categories(labels):
    """Get annotated labels, which is a list of dictionaries."""
    # spacy needs a positive t/f annotation
    categories = [{'POSITIVE': bool(y)} for y in labels]
    return categories


def evaluate(tokenizer, textcat, text, categories):
    """Evaluates the classifier."""
    docs = (tokenizer(text) for text in text)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = categories[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                true_positives += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                false_positives += 1.
            elif score < 0.5 and gold[label] < 0.5:
                true_negatives += 1
            elif score < 0.5 and gold[label] >= 0.5:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_score = 2 * (precision * recall) / (precision + recall)
    return 'Precision: {}, recall: {}, f-score: {}'.format(precision, recall, f_score)


def train_cnn(X_train, cats_train,
              X_val, cats_val,
              X_test, cats_test,
              n_iter=5):
    """Train CNN on training data and updates model using validation data. Finally,
    print scores on test data. It also prints training progres."""
    nlp = spacy.load('en_core_web_sm')
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe('textcat')
    textcat.add_label('POSITIVE')

    # only train textcat, disable other pipes in pipeline
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']

    train_data = list(zip(X_train, [{'cats': cats} for cats in cats_train]))

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for i in range(n_iter):
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                scores = evaluate(nlp.tokenizer, textcat, X_val, cats_val)
                print(scores)

    # save model to disk
    nlp.to_disk('../models/cnn_classifier')

    print('Scores on test data...')
    print(evaluate(nlp.tokenizer, textcat, X_test, cats_test))


def main():
    X_train, y_train = load_dataset('../data/disaster_response_messages_training.csv',
                                    'weather_related')
    X_val, y_val = load_dataset('../data/disaster_response_messages_training.csv',
                                'weather_related')
    X_test, y_test = load_dataset('../data/disaster_response_messages_test.csv',
                                  'weather_related')

    # get annotated labels
    cats_train = get_categories(y_train)
    cats_val = get_categories(y_val)
    cats_test = get_categories(y_test)

    train_cnn(X_train, cats_train,
              X_val, cats_val,
              X_test, cats_test, 10)


if __name__ == '__main__':
    main()
