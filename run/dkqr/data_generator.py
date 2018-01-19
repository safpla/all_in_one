import os
import pickle as pkl
import numpy as np
import sys

root_path = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, root_path)

from all_in_one.data.data_utils import utils
from all_in_one.data.data_utils.data_processor import DataProcessor


def data_generator(excel_name, data_name):
    kwargs = {
    'label1' : 2,
    'label2' : 3}
    test = DataProcessor(excel_name, data_name, tokenizer_tool='thulac', **kwargs)
    test.excel2raw()
    test.excel2id(modify_vocab=utils.prepro_vocab1, call_back=utils.sub_num_name_add_padding, reflush=False)
    test.build_vocab_inword(modify_vocab=utils.prepro_vocab1, call_back=utils.sub_num_name_add_padding, reflush=False)
    test.build_label_class_mapping()
    test.build_word_embedding(word_embedding_name='thulac',
                              modify_vocab=utils.prepro_vocab1,
                              call_back=utils.sub_num_name_add_padding, reflush=False)
    test.build_cross_validation(modify_vocab=utils.prepro_vocab1, call_back=utils.sub_num_name_add_padding, reflush=True)


if __name__ == '__main__':
    data_generator('dkqr.xlsx', 'dkqr')
