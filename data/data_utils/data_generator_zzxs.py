import os
import pickle as pkl
import numpy as np
import utils
from data_processor import DataProcessor


class YouGenerator(DataProcessor):
    '''使用示例1'''

    def __init__(self, file_path, sheet_num=0, tokenizer_tool='thulac', model_path='thulac_models', seg_only=False, **kwargs):
        super(self, DataProcessor).__init__(file_path=file_path, sheet_num=sheet_num,
                                            tokenizer_tool=tokenizer_tool, model_path=model_path, seg_only=seg_only, kwargs=kwargs)

    def somefun(self):
        pass


def data_generator_zzxs(excel_name, data_name):
    test = DataProcessor('../data_excel/' + excel_name, tokenizer_tool='thulac', model_path='thulac_models')
    test.excel2raw(data_name)
    test.excel2id(data_name, modify_vocab=utils.prepro_vocab1, call_back=utils.sub_num_name_add_padding, reflush=False)
    test.build_vocab_inword(data_name, modify_vocab=utils.prepro_vocab1, call_back=utils.sub_num_name_add_padding, reflush=False)
    test.build_label_class_mapping(data_name)
    test.build_word_embedding(file_path='./thulac_models/chinesegigawordv5.deps.vec', model_name=data_name,
                              modify_vocab=utils.prepro_vocab1,
                              call_back=utils.sub_num_name_add_padding, reflush=False)
    test.build_cross_validation(data_name, modify_vocab=utils.prepro_vocab1, call_back=utils.sub_num_name_add_padding, reflush=False)


if __name__ == '__main__':
    data_generator_zzxs('zzxs.xlsx', 'zzxs')
