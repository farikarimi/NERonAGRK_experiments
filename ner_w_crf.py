import csv
import pickle
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
from util_funcs import print2both, pickle_obj, unpickle_obj
from lev_distance import get_smallest_lev_dist_in_para
from grk_xml_funcs import get_grk_sentences

current_token_no = 0
lev_distances = {}
from_scratch = False


def get_gold_sentences(sents):
    """Extracts from the given list of sentences those that
    contain a word from the gold standard lists and returns them."""
    g_sentences = [sentence for sentence in sents if has_gold_word([token[0] for token in sentence])]
    print(f'found {len(g_sentences)} sentences that contain a name from the gold standard lists\n\n')
    return g_sentences


def word2features(sent, i):
    """Returns the feature dictionary for a token."""
    word = sent[i][0]
    print(f'\nword from word2features: {word}')
    postag = sent[i][1]

    global current_token_no
    current_token_id = str(current_token_no) + '_' + word

    if from_scratch:
        smallest_lev_dist = get_smallest_lev_dist_in_para(token_para_no=sent[i][3], token=sent[i][0])
        lev_distances[current_token_id] = smallest_lev_dist
    else:
        smallest_lev_dist = lev_distances.get(current_token_id)

    print(f'\nadded to or read from dict: {current_token_id}: {smallest_lev_dist}\n')

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

    current_token_no += 1

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


def train_crf_model(gold_stan_sentences):
    """Returns [0] a CRF model trained to recognize toponyms and ethnonyms in Herodotus' Histories
    and [1] the features and [2] labels used for training it."""
    features = [sent2features(s) for s in gold_stan_sentences]
    labels = [sent2labels(s) for s in gold_stan_sentences]
    crf_model = sklearn_crfsuite.CRF(c1=0.1, c2=0.1, max_iterations=100)
    crf_model = crf_model.fit(X=features, y=labels)
    return crf_model, features, labels


def make_predictions(crf_model, sents):
    """Uses the given trained CRF model to predict toponyms and ethnonyms in Herodotus' Histories."""
    # list of lists of feature dicts
    features = [sent2features(s) for s in sents]
    return crf_model.predict(X=features)


def save_predictions(sents, y_hat):
    """Saves all of the model's predictions to a CSV file."""
    csv_file = open(f'results/all_predictions_{datetime.datetime.today().date()}.csv', 'w',
                    newline='', encoding='utf-8')
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


def categorize_predictions(gold_sents, y_hat, y_actual):
    """Categorizes the predictions of the cross-validation into potentially correct and incorrect classifications and
    saves them in two seperate CSV files."""
    f1 = open(f'results/potentially_correct_predictions_{datetime.datetime.today().date()}.csv', 'w',
              newline='', encoding='utf-8')
    f2 = open(f'results/misclassifications_{datetime.datetime.today().date()}.csv', 'w',
              newline='', encoding='utf-8')
    number1 = 0
    number2 = 0
    header = ['no', 'token', 'pos', 'actual_label', 'predicted_label', 'sent_no', 'token_no', 'sent']
    writer1 = csv.DictWriter(f1, fieldnames=header)
    writer1.writeheader()
    writer2 = csv.DictWriter(f2, fieldnames=header)
    writer2.writeheader()
    for i in range(len(y_hat)):
        for j in range(len(y_hat[i])):
            sent = [gold_sents[i][k][0] for k in range(len(gold_sents[i]))]
            if y_actual[i][j] != y_hat[i][j]:
                # predictions that could be right
                if y_actual[i][j] == '0':
                    number1 += 1
                    writer1.writerow({
                        'no': str(number1),
                        'token': gold_sents[i][j][0],
                        'pos': gold_sents[i][j][1],
                        'actual_label': gold_sents[i][j][2],
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
                        'token': gold_sents[i][j][0],
                        'pos': gold_sents[i][j][1],
                        'actual_label': gold_sents[i][j][2],
                        'predicted_label': y_hat[i][j],
                        'sent_no': str(i),
                        'token_no': str(j),
                        'sent': ' '.join(sent)
                    })
    f1.close()
    f2.close()


def performance_measurement(crf_model, x, y, g_sentences):
    """Utilizes different functions to measure the model's performance and saves the results to files for review."""
    # Cross-validating the model
    cross_val_predictions = cross_val_predict(estimator=crf_model, X=x, y=y, cv=5)
    report = flat_classification_report(y_pred=cross_val_predictions, y_true=y)
    file = open(f'results/performance_measurement_results_{datetime.datetime.today().date()}.txt', 'a', encoding='utf-8')
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
    categorize_predictions(gold_sents=g_sentences, y_hat=cross_val_predictions, y_actual=y)


# TODO: using 3 sentences to test if changes are bug-free --> change back
def run_and_pickle():
    print(f'\n{datetime.datetime.now()}: getting Greek sentences...')
    # sentences = get_grk_sentences()   # TODO: uncomment to get sentences from XML
    # change date in file name accordingly
    sentences = unpickle_obj('pickle_files/grk_sentences_2020-12-03.pickle')
    pickle_obj(obj=sentences, path=f'pickle_files/grk_sentences_{datetime.datetime.today().date()}.pickle')

    print(f'\n{datetime.datetime.now()}: getting gold sentences...')
    global from_scratch
    from_scratch = True
    # gold_sentences = get_gold_sentences(sents=sentences) # TODO: uncomment to get sentences from scratch
    # change date in file name accordingly
    gold_sentences = unpickle_obj('pickle_files/gold_sentences_2020-12-03.pickle')
    pickle_obj(obj=gold_sentences, path=f'pickle_files/gold_sentences_{datetime.datetime.today().date()}.pickle')

    print(f'\n{datetime.datetime.now()}: training CRF model...')
    crf = train_crf_model(gold_stan_sentences=gold_sentences[:2])
    global lev_distances
    pickle_obj(obj=lev_distances, path=f'pickle_files/lev_distances_training_{datetime.datetime.today().date()}.pickle')
    pickle_obj(obj=crf[0], path=f'pickle_files/crf_model_{datetime.datetime.today().date()}.pickle')
    pickle_obj(obj=crf[1], path=f'pickle_files/features_{datetime.datetime.today().date()}.pickle')
    pickle_obj(obj=crf[2], path=f'pickle_files/labels_{datetime.datetime.today().date()}.pickle')

    print(f'\n{datetime.datetime.now()}: making predictions with trained model...')
    lev_distances = {}
    global current_token_no
    current_token_no = 0
    predictions = make_predictions(crf_model=crf[0], sents=sentences[:2])
    pickle_obj(obj=lev_distances, path=f'pickle_files/lev_distances_pred_{datetime.datetime.today().date()}.pickle')

    print(f'\n{datetime.datetime.now()}: saving predictions...')
    save_predictions(sents=sentences[:2], y_hat=predictions)

    # print(f'\n{datetime.datetime.now()}: measuring performance...')
    # performance_measurement(crf_model=crf[0], x=crf[1], y=crf[2], g_sentences=gold_sentences)


# TODO: using 3 sentences to test if changes are bug-free --> change back
def run_from_pickle():
    print(f'\n{datetime.datetime.now()}: unpickling Greek sentences...')
    # change date in file name accordingly
    sentences_unpickled = unpickle_obj('pickle_files/grk_sentences_2020-12-03.pickle')

    print(f'\n{datetime.datetime.now()}: unpickling gold sentences...')
    # change date in file name accordingly
    gold_sentences_unpickled = unpickle_obj('pickle_files/gold_sentences_2020-12-03.pickle')

    print(f'\n{datetime.datetime.now()}: training crf model...')
    global lev_distances
    # change date in file name accordingly
    lev_distances = unpickle_obj('pickle_files/lev_distances_training_2020-12-04.pickle')
    crf = train_crf_model(gold_stan_sentences=gold_sentences_unpickled[:2])

    print(f'\n{datetime.datetime.now()}: making predictions with trained model...')
    global current_token_no
    current_token_no = 0
    # change date in file name accordingly
    lev_distances = unpickle_obj('pickle_files/lev_distances_pred_2020-12-04.pickle')
    predictions = make_predictions(crf_model=crf[0], sents=sentences_unpickled[:2])

    print(f'\n{datetime.datetime.now()}: saving predictions...')
    save_predictions(sents=sentences_unpickled[:2], y_hat=predictions)

    # print(f'\n{datetime.datetime.now()}: measuring performance...')
    # performance_measurement(crf_model=crf[0], x=crf[1], y=crf[2], g_sentences=gold_sentences_unpickled)


if __name__ == '__main__':
    faulthandler.enable()
    print('\nsklearn: %s' % sklearn.__version__)  # 0.23.2
    print(f'\n{datetime.datetime.now()}: running ner_w_crf.py...')
    start = timer()

    # comment out / uncomment next two lines to create objects from scratch or get data from pickle files
    run_and_pickle()
    # run_from_pickle()

    end = timer()
    print(f'\nelapsed time: {math.ceil((end - start) / 60)} minutes')
