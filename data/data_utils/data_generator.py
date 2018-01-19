import os
import pickle as pkl
import numpy as np
import utils
# import random
from data_processor import DataProcessor


class YouGenerator(DataProcessor):
    '''使用示例1'''

    def __init__(self, file_path, sheet_num=0, tokenizer_tool='thulac', model_path='thulac_models', seg_only=False, **kwargs):
        super(DataProcessor, self).__init__(file_path=file_path, sheet_num=sheet_num,
                                            tokenizer_tool=tokenizer_tool, model_path=model_path, seg_only=seg_only, kwargs=kwargs)

    def somefun():
        pass


def data_generator():
    '''使用示例2'''
    # test = DataProcessor(file_path='qrsd4268.xlsx', model_name='qrsd', tokenizer_tool='thulac', label2=3, label3=4)
    test = DataProcessor(file_path='zzxs.xlsx', model_name='zzxs', tokenizer_tool='thulac',)
    print(test)
    test.excel2raw()
    # test.get_cuted_data()
    # test.get_cuted_data(call_back=utils.sub_num_name_add_padding)
    test.build_vocab_inword(modify_vocab=utils.prepro_vocab1,
                            call_back=utils.sub_num_name_add_padding, reflush=False)
    test.excel2id(modify_vocab=utils.prepro_vocab1,
                  call_back=utils.sub_num_name_add_padding, reflush=False)
    # test.build_vocab_inchar(call_back=utils.remove_signal)
    test.build_label_class_mapping()
    test.build_word_embedding(modify_vocab=utils.prepro_vocab1,
                              call_back=utils.sub_num_name_add_padding, reflush=False)
    # 做cross validation，调用get_excel2id,get_cuted_data,build_vocab_inword 一并输出
    test.build_cross_validation(modify_vocab=utils.prepro_vocab1,
                                call_back=utils.sub_num_name_add_padding, reflush=False)
