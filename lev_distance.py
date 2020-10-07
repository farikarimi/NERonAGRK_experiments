import re
from fuzzywuzzy import fuzz
from util_funcs import strip_diacritics, lst2file
from eng_xml_funcs import get_eng_nes_per_para, get_eng_para_nos
from grk_xml_funcs import get_grk_para_nos


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def remove_duplicates(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item


def sort_and_deduplicate(lst):
    return list(remove_duplicates(sorted(lst, key=str.casefold)))


def combine_lists(diction):
    names_list = []
    for lst in list(diction.values()):
        for name in lst:
            name = re.sub(r'\n\s+', ' ', name)
            names_list.append(name)
    return names_list


def create_eng_grk_dict():
    eng_names_list = combine_lists(get_eng_nes_per_para(ne_type='all'))
    eng_names_set_list = sort_and_deduplicate(eng_names_list)
    lst2file(eng_names_set_list, 'results/english_names.txt')
    with open('data/eng_to_grk_name_translations.txt', 'r') as translation_file:
        eng_to_grk_name_translations = [name for name in translation_file.read().splitlines()]
    eng_to_grk_dict_clean = {eng_name: grk_name for eng_name, grk_name in
                             zip(eng_names_set_list, eng_to_grk_name_translations) if not is_english(grk_name)}
    return eng_to_grk_dict_clean


def get_smallest_lev_dist_in_para(token_para_no, token):
    mapped_para_nos = {grk_para_no: eng_para_no for grk_para_no, eng_para_no
                       in zip(get_grk_para_nos(), get_eng_para_nos())}
    eng_para_no = mapped_para_nos[token_para_no]
    eng_nes_per_para = get_eng_nes_per_para(ne_type='all')
    eng_grk_dict = create_eng_grk_dict()
    eng_grk_dict_for_para = {eng_name: eng_grk_dict[eng_name] for eng_name in eng_nes_per_para[eng_para_no]
                             if eng_name in eng_grk_dict.keys()}
    print(eng_grk_dict_for_para)
    lev_distances = [fuzz.ratio(strip_diacritics(token), strip_diacritics(grk_name))
                     for grk_name in eng_grk_dict_for_para.values()]
    print('\n\ntoken:', token, '\nlevenshtein distance of token to names in paragraph:',
          {name: lev_distance for name, lev_distance in zip(eng_grk_dict_for_para.values(), lev_distances)})
    if lev_distances:
        return max(lev_distances)
    return 0

