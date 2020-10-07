import csv
import datetime
import math
import faulthandler
from timeit import default_timer as timer
import eli5
import sklearn
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import cross_val_predict
from gold_standard_funcs import has_gold_word
from util_funcs import print2both
from lev_distance import get_smallest_lev_dist_in_para
from grk_xml_funcs import get_grk_sentences


def get_gold_sentences(sents):
    """Extracts from the given list of sentences those that
    contain a word from the gold standard lists and returns them."""
    g_sentences = [sentence for sentence in sents if has_gold_word([token[0] for token in sentence])]
    print(f'found {len(g_sentences)} sentences that contain a name from the gold standard lists\n\n')
    return g_sentences


def word2features(sent, i):
    """Returns the feature dictionary for a token."""
    word = sent[i][0]
    postag = sent[i][1]
    smallest_lev_dist = get_smallest_lev_dist_in_para(token_para_no=sent[i][3], token=sent[i][0])

    features = {
        'bias': 1.0,
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.istitle()': word.istitle(),
        'postag': postag,
        'postag[:1]': postag[:1],
        'smallest_lev_dist': smallest_lev_dist
    }

    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:postag': postag1,
            '-1:postag[:1]': postag1[:1]
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:postag': postag1,
            '+1:postag[:1]': postag1[:1]
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    """Returns a list containing the feature dictionary for each token in a sentence."""
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    """Returns a list containing the labels (entity type) for each token in a sentence."""
    return [label for token, pos, label, para in sent]


def sent2tokens(sent):
    """Returns a list containing the forms for each token in a sentence."""
    return [form for form, pos, label, para in sent]


def train_crf_model(gold_sentences):
    """Returns [0] a CRF model trained to recognize toponyms and ethnonyms in Herodotus' Histories
    and [1] the features and [2] labels used for training it."""
    features = [sent2features(s) for s in gold_sentences]
    labels = [sent2labels(s) for s in gold_sentences]
    crf_model = sklearn_crfsuite.CRF(c1=0.1, c2=0.1, max_iterations=100)
    crf_model = crf_model.fit(X=features, y=labels)
    return crf_model, features, labels


def make_predictions(crf_model, sents):
    """Uses the given trained CRF model to predict toponyms and ethnonyms in Herodotus' Histories."""
    # list of lists of feature dicts
    # print(f'complete sentences:\n\n{sents}')
    features = [sent2features(s) for s in sents]
    # print(f'features:\n\n{features}')
    return crf_model.predict(X=features)


def save_predictions(sents, y_hat):
    """Saves all of the model's predictions to a CSV file."""
    csv_file = open(f'results/all_predictions_{datetime.datetime.today().date()}.csv', 'w')
    number = 0
    header = ['no', 'token', 'pos', 'actual_label', 'predicted_label',
              'sent_no', 'token_no', 'paragraph', 'sent']
    writer = csv.DictWriter(csv_file, fieldnames=header)
    writer.writeheader()
    for i in range(len(y_hat)):
        for j in range(len(y_hat[i])):
            sent = [sents[i][k][0] for k in range(len(sents[i]))]
            number += 1
            writer.writerow({
                'no': str(number),
                'token': sents[i][j][0],
                'pos': sents[i][j][1],
                'actual_label': sents[i][j][2],
                'predicted_label': y_hat[i][j],
                'sent_no': str(i),
                'token_no': str(j),
                'paragraph': sents[i][j][3],
                'sent': ' '.join(sent)
            })
    csv_file.close()


def categorize_predictions(gold_sentences, y_hat, y_actual):
    """Categorizes the predictions of the cross-validation into potentially correct and incorrect classifications and
    saves them in two seperate CSV files."""
    f1 = open(f'results/potentially_correct_predictions_{datetime.datetime.today().date()}.csv', 'w')
    f2 = open(f'results/misclassifications_{datetime.datetime.today().date()}.csv', 'w')
    number1 = 0
    number2 = 0
    header = ['no', 'token', 'pos', 'actual_label', 'predicted_label', 'sent_no', 'token_no', 'sent']
    writer1 = csv.DictWriter(f1, fieldnames=header)
    writer1.writeheader()
    writer2 = csv.DictWriter(f2, fieldnames=header)
    writer2.writeheader()
    for i in range(len(y_hat)):
        for j in range(len(y_hat[i])):
            sent = [gold_sentences[i][k][0] for k in range(len(gold_sentences[i]))]
            if y_actual[i][j] != y_hat[i][j]:
                # predictions that could be right
                if y_actual[i][j] == '0':
                    number1 += 1
                    writer1.writerow({
                        'no': str(number1),
                        'token': gold_sentences[i][j][0],
                        'pos': gold_sentences[i][j][1],
                        'actual_label': gold_sentences[i][j][2],
                        'predicted_label': y_hat[i][j],
                        'sent_no': str(i),
                        'token_no': str(j),
                        'sent': ' '.join(sent)
                    })
                # misclassifications
                else:
                    number2 += 1
                    writer2.writerow({
                        'no': str(number2),
                        'token': gold_sentences[i][j][0],
                        'pos': gold_sentences[i][j][1],
                        'actual_label': gold_sentences[i][j][2],
                        'predicted_label': y_hat[i][j],
                        'sent_no': str(i),
                        'token_no': str(j),
                        'sent': ' '.join(sent)
                    })
    f1.close()
    f2.close()


def performance_measurement(crf_model, x, y, gold_sentences):
    """Utilizes different functions to measure the model's performance and saves the results to files for review."""
    # Cross-validating the model
    cross_val_predictions = cross_val_predict(estimator=crf_model, X=x, y=y, cv=5)
    report = flat_classification_report(y_pred=cross_val_predictions, y_true=y)
    file = open(f'results/performance_measurement_results_{datetime.datetime.today().date()}.txt', 'a')
    file.seek(0)
    file.truncate()
    print2both('created on:', str(datetime.datetime.today().date()), '\n', file=file)
    print2both('flat_classification_report:\n\n', report, '\n\n', file=file)
    print2both('cross_val_predict:\n\n', cross_val_predictions, '\n\n', file=file)
    # Showing the weights assigned to each feature
    print2both('eli5.explain_weights(crf, top=100):\n\n',
               eli5.format_as_text(eli5.explain_weights(crf_model, top=100)), '\n\n', file=file)
    file.close()
    # Saving the potentially correct and the incorrect classifications in separate CSV files for review
    categorize_predictions(gold_sentences=gold_sentences, y_hat=cross_val_predictions, y_actual=y)


if __name__ == '__main__':
    faulthandler.enable()
    print('\nsklearn: %s' % sklearn.__version__)  # 0.23.2
    print('\nrunning ner_w_crf.py...')
    start = timer()

    sentences = get_grk_sentences()
    gold_sentences = get_gold_sentences(sents=sentences)
    crf = train_crf_model(gold_sentences=gold_sentences)
    predictions = make_predictions(crf_model=crf[0], sents=sentences)
    save_predictions(sents=sentences, y_hat=predictions)
    performance_measurement(crf_model=crf[0], x=crf[1], y=crf[2], gold_sentences=gold_sentences)

    end = timer()
    print(f'\nelapsed time: {math.ceil((end - start) / 60)} minutes')