import xml.etree.ElementTree as eTree
from preprocess_util import PUNCTUATION
from gold_standard_funcs import get_type


def get_xml_root(path):
    """Returns the root element of the XML file from the given path."""
    return eTree.parse(path).getroot()


def get_agrk_sentences(path):
    """Returns a list containing all of the sentences from Histories' Ancient Greek XML file from the given path."""
    # Reading the XML file with the annotated text of Histories.
    root = get_xml_root(path)
    sentences = []
    print(f'\nfound {len(root.findall("s"))} sentences in {path}\n')
    for s in root.findall('s'):
        ts = [t for t in s.findall('t') if t.text is not None and t.text not in PUNCTUATION]
        tokens = []
        for t in ts:
            # Each token is a 4-tuple consisting of
            token_tuple = (t.text,              # a form,
                           t.get('o'),          # a POS-tag,
                           get_type(t.text),    # an entity type and
                           t.get('p'))          # a paragraph number.
            tokens.append(token_tuple)
        # Each sentence consists of a list of tokens.
        sentences.append(tokens)
    return sentences
