import re
import datetime
import csv
from fuzzywuzzy import fuzz
from util_funcs import strip_diacritics, lst2file
from eng_xml_funcs import get_eng_nes_per_para, get_eng_para_nos
from grk_xml_funcs import get_grk_para_nos

pred_phase = False
token_no = 0
csv_file = None
csv_writer = None


def combine_lists(diction):
    names_list = []
    for lst in list(diction.values()):
        for name in lst:
            name = re.sub(r'\n\s+', ' ', name)
            names_list.append(name)
    return names_list


def remove_duplicates(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item


def sort_and_deduplicate(lst):
    return list(remove_duplicates(sorted(lst, key=str.casefold)))


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def create_eng_grk_dict():
    eng_names_list = combine_lists(get_eng_nes_per_para(ne_type='all'))
    eng_names_set_list = sort_and_deduplicate(eng_names_list)
    lst2file(eng_names_set_list, 'results/english_names.txt')
    with open('data/eng_to_grk_name_translations.txt', 'r', encoding='utf-8') as translation_file:
        eng_to_grk_name_translations = [name for name in translation_file.read().splitlines()]
    eng_to_grk_dict_clean = {eng_name: grk_name for eng_name, grk_name in
                             zip(eng_names_set_list, eng_to_grk_name_translations) if not is_english(grk_name)}
    return eng_to_grk_dict_clean


def open_csv_file():
    global csv_file
    global csv_writer
    csv_file = open(f'results/match_scores_{datetime.datetime.today().date()}.csv',
                    'w', newline='', encoding='utf-8')
    header = ['token_no', 'para_no', 'token', 'max_match_score', 'match_scores', 'para_dict']
    csv_writer = csv.DictWriter(csv_file, fieldnames=header)


def get_highest_match_score_in_para(token_para_no, token):
    global token_no
    token_no += 1
    mapped_para_nos = {grk_para_no: eng_para_no for grk_para_no, eng_para_no
                       in zip(get_grk_para_nos(), get_eng_para_nos())}
    eng_para_no = mapped_para_nos[token_para_no]
    eng_nes_per_para = get_eng_nes_per_para(ne_type='all')
    eng_grk_dict = create_eng_grk_dict()
    eng_grk_dict_for_para = {eng_name: eng_grk_dict[eng_name] for eng_name in eng_nes_per_para[eng_para_no]
                             if eng_name in eng_grk_dict.keys()}
    # list containing the match score of the greek token with each name in the paragraph
    match_scores = [fuzz.ratio(strip_diacritics(token), strip_diacritics(grk_name))
                    for grk_name in eng_grk_dict_for_para.values()]
    max_match_score = max(match_scores) if match_scores else 0

    if pred_phase:
        csv_writer.writerow({
            'token_no': token_no,
            'para_no': token_para_no,
            'token': token,
            'max_match_score': max_match_score,
            'match_scores': {name: match_score for name, match_score in zip(eng_grk_dict_for_para.values(), match_scores)},
            'para_dict': eng_grk_dict_for_para
        })

    print(f'\nparagraph: {token_para_no}'
          f'\ntoken number {token_no}: {token}'
          f'\nnames detected in the paragraph from the ENG-GRK dictionary: {eng_grk_dict_for_para}'
          f'\nmatch score of token with each name: ',
          {name: match_score for name, match_score in zip(eng_grk_dict_for_para.values(), match_scores)},
          f'\nhighest match score: {max_match_score}'
          f'\n\n******************************************************************************************************',
          sep='')

    return max_match_score

