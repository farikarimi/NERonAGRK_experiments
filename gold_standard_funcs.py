from preprocess_util import strip_diacritics


def get_gold_toponyms():
    """Returns the gold standard list for toponyms."""
    with open('data/places_gold-standard_extended.txt', 'r') as places_file:
        places_txt = places_file.read()
    return [strip_diacritics(place) for place in places_txt.splitlines()]


def get_gold_ethnonyms():
    """Returns the gold standard list for ethnonyms."""
    with open('data/ethnics_gold-standard_extended.txt', 'r') as ethnics_file:
        ethnics_txt = ethnics_file.read()
    return [strip_diacritics(ethnic) for ethnic in ethnics_txt.splitlines()]


def has_gold_word(sent):
    """Returns True if the sentence contains a token from the gold standard lists."""
    toponyms_and_ethnonyms = get_gold_toponyms() + get_gold_ethnonyms()
    for token in sent:
        if strip_diacritics(token) in toponyms_and_ethnonyms:
            return True
    return False


def get_type(token):
    """Returns the entity type for a given token based on the gold standard lists."""
    token = strip_diacritics(token)
    if token in get_gold_toponyms():
        # Label for toponyms
        return 'place'
    if token in get_gold_ethnonyms():
        # Label for ethnonyms
        return 'ethnic'
    else:
        # Label for any other word
        return '0'

