import xml.etree.ElementTree as eTree
from util_funcs import PUNCTUATION
from gold_standard_funcs import get_type

GRK_XML_ROOT = eTree.parse('data/hdt_complete-greek_postags.xml').getroot()


def get_grk_para_nos():
    grk_para_nos = []
    for t in GRK_XML_ROOT.iter('t'):
        if t.get('p') not in grk_para_nos:
            grk_para_nos.append(t.get('p'))
    return grk_para_nos


def get_grk_sentences():
    """Returns a list containing all of the sentences from Histories' Ancient Greek XML file from the given path."""
    # Reading the XML file with the annotated text of Histories.
    sentences = []
    print(f'\nfound {len(GRK_XML_ROOT.findall("s"))} sentences in Greek XML file\n')
    for s in GRK_XML_ROOT.findall('s'):
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
