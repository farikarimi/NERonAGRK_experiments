from greek_accentuation.characters import base

PUNCTUATION = ['.', ',', '“', '”', ';', '·']


def strip_diacritics(s):
    """Removes all diacritics from the given string and returns it."""
    return ''.join(base(c) for c in s)


def print_both(*args, file):
    text = ' '.join([str(arg) for arg in args])
    print(text)
    file.write(text + '\n')
