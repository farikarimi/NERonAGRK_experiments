import re
import xml.etree.ElementTree as eTree


ENG_XML_ROOT = eTree.parse('data/hdt_eng-annotated-perseus.xml').getroot()
TEI_NAMESPACE = {'tei': 'http://www.tei-c.org/ns/1.0'}
VALID_TYPES = {'place', 'ethnic', 'person', 'all'}


def get_eng_para_nos():
    eng_para_nos = []
    for book in ENG_XML_ROOT[1][0][0].findall('tei:div', TEI_NAMESPACE):
        for chapter in book.findall('tei:div', TEI_NAMESPACE):
            for section in chapter.findall('tei:div', TEI_NAMESPACE):
                book_chap_sect = book.get('n') + '.' + chapter.get('n') + '.' + section.get('n')
                eng_para_nos.append(book_chap_sect)
    return eng_para_nos


def get_eng_nes_per_para(ne_type):
    if ne_type not in VALID_TYPES:
        raise ValueError("results: type must be one of %r." % VALID_TYPES)
    places_per_para_en = {}
    ethnics_per_para_en = {}
    persons_per_para_en = {}
    all_names_per_para_en = {}
    for book in ENG_XML_ROOT[1][0][0].findall('tei:div', TEI_NAMESPACE):
        for chapter in book.findall('tei:div', TEI_NAMESPACE):
            for section in chapter.findall('tei:div', TEI_NAMESPACE):
                paragraph = section.find('tei:p', TEI_NAMESPACE)
                toponyms = []
                ethnonyms = []
                persons = []
                for name in paragraph.findall('tei:name', TEI_NAMESPACE):
                    if name.get('type') == 'place':
                        place_name = name.find('tei:placeName', TEI_NAMESPACE)
                        reg = name.find('tei:reg', TEI_NAMESPACE)
                        if place_name is not None:
                            toponyms.append(re.sub(r'\n\s*', ' ', place_name.text))
                        elif name.text is not None:
                            toponyms.append(re.sub(r'\n\s*', ' ', name.text))
                        elif reg.tail is not None:
                            toponyms.append(re.sub(r'\n\s*', ' ', reg.tail))
                        else:
                            print('PROBLEM')
                    if name.get('type') == 'ethnic':
                        ethnonyms.append(re.sub(r'\n\s*', ' ', name.text))
                    if name.get('type') == 'pers':
                        persons.append(re.sub(r'\n\s*', ' ', name.text))
                book_chap_sect = book.get('n') + '.' + chapter.get('n') + '.' + section.get('n')
                places_per_para_en[book_chap_sect] = toponyms
                ethnics_per_para_en[book_chap_sect] = ethnonyms
                persons_per_para_en[book_chap_sect] = persons
                all_names_per_para_en[book_chap_sect] = toponyms + ethnonyms

    if ne_type == 'place':
        return places_per_para_en
    elif ne_type == 'ethnic':
        return ethnics_per_para_en
    elif ne_type == 'person':
        return persons_per_para_en
    elif ne_type == 'all':
        return all_names_per_para_en


def save_eng_pers_names():
    persons_file = open('results/eng_pers_names.txt', 'a', encoding='utf-8')
    persons_file.seek(0)
    persons_file.truncate()
    for pers_list in get_eng_nes_per_para('person').values():
        for name in pers_list:
            persons_file.write(name.strip() + '\n')


# if __name__ == '__main__':
#     save_eng_pers_names()
