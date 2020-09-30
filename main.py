import math
import faulthandler
from timeit import default_timer as timer
import sklearn
from xml_funcs import get_agrk_sentences
from ner_w_crf import get_gold_sentences, train_crf_model, make_predictions, save_predictions, performance_measurement

if __name__ == '__main__':
    faulthandler.enable()
    print('\nsklearn: %s' % sklearn.__version__)  # 0.23.2
    print('\nrunning main script...')
    start = timer()

    sentences = get_agrk_sentences('data/hdt_complete-greek_postags.xml')
    gold_sentences = get_gold_sentences(sents=sentences)
    crf = train_crf_model(gold_sentences=gold_sentences)
    predictions = make_predictions(crf_model=crf[0], sents=sentences)
    save_predictions(sents=sentences, y_hat=predictions)
    performance_measurement(crf_model=crf[0], x=crf[1], y=crf[2], gold_sentences=gold_sentences)

    end = timer()
    print(f'\nelapsed time: {math.ceil((end - start) / 60)} minutes')
