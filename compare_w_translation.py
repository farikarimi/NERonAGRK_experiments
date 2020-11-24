import datetime
from openpyxl import Workbook
from eng_xml_funcs import get_eng_nes_per_para
from grk_xml_funcs import get_grk_nes_per_para


def make_table(eng_places, grk_places, eng_ethnics, grk_ethnics):
    wb = Workbook()
    ws = wb.active
    ws.append(['ENG toponyms', 'AGRK toponyms', 'ENG ethnonyms', 'AGRK ethnonyms', 'book.chapter.section (ENG / AGRK)'])
    for (eng_place_k, eng_place_v), (grk_place_k, grk_place_v), (eng_ethn_v, eng_ethn_v), (grk_ethn_k, grk_ethn_v) \
            in zip(eng_places.items(), grk_places.items(), eng_ethnics.items(), grk_ethnics.items()):
        if eng_place_v or grk_place_v or eng_ethn_v or grk_ethn_v:
            eng_places_str = ', '.join(eng_place_v)
            grk_places_str = ', '.join(grk_place_v)
            eng_ethn_str = ', '.join(eng_ethn_v)
            grk_ethn_str = ', '.join(grk_ethn_v)
            bcs = eng_place_k if eng_place_k == grk_place_k else eng_place_k + ' / ' + grk_place_k
            ws.append([eng_places_str, grk_places_str, eng_ethn_str, grk_ethn_str, bcs])
    wb.save(f'results/comparison_w_translation_{datetime.datetime.today().date()}.xlsx')


if __name__ == '__main__':
    make_table(eng_places=get_eng_nes_per_para(ne_type='place'), grk_places=get_grk_nes_per_para(ne_type='place'),
               eng_ethnics=get_eng_nes_per_para(ne_type='ethnic'), grk_ethnics=get_grk_nes_per_para(ne_type='ethnic'))
