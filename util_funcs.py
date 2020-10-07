from greek_accentuation.characters import base

PUNCTUATION = ['.', ',', '“', '”', ';', '·']


def strip_diacritics(s):
    """Removes all diacritics from the given string and returns it."""
    return ''.join(base(c) for c in s)


def print2both(*args, file):
    """Writes the given arguments to the console and the given file."""
    text = ' '.join([str(arg) for arg in args])
    print(text)
    file.write(text + '\n')


def lst2file(lst, path):
    """Writes the items in the given list to the file at the given path separated by a newline."""
    with open(path, 'w') as file:
        for name in lst:
            file.write(name + '\n')
