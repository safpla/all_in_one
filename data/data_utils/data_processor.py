import os
import re
import sys
import math
import time

from multiprocessing import Process, Manager
from tqdm import tqdm
import xlrd
import numpy as np
import pickle as pkl

from all_in_one.data.data_utils import utils
from all_in_one.data.data_utils import tokenizer
from all_in_one.config import config_utils
root_path = sys.path[0]
config_path = root_path + '/all_in_one/config/base_config.ini'


class DataProcessor:

    def __init__(self, file_path, model_name, sheet_num=0, tokenizer_tool='thulac', seg_only=False, max_pro=5, **kwargs):
        self.config_args = config_utils.Config()(config_path)['data_utils']
        self.file_path = os.path.join(root_path, 'all_in_one/data/data_excel/' + file_path)
        self.save_path = os.path.join(root_path, self.config_args['save_path'])
        self.workbook = xlrd.open_workbook(self.file_path)
        self.sheet = self.workbook.sheet_by_index(sheet_num)
        self.model_name = model_name
        self.max_pro = max_pro
        self.kwargs = {
            'category': 0,
            'content': 1,
            'label1': 2,
        }
        if kwargs:
            self.kwargs.update(kwargs)
        if tokenizer_tool:
            self.tokenizer_dict = {'char': tokenizer.CharacterTokenizer,
                                   'thulac': tokenizer.ThulacTokenizer,
                                   'jieba': tokenizer.HanlpTokenizer,
                                   'hanlp': tokenizer.JiebaTokenizer}
            if tokenizer_tool not in self.tokenizer_dict.keys():
                raise NotImplementedError
            if tokenizer_tool == 'char':
                self.tokenizer = tokenizer.CharacterTokenizer()
            if tokenizer_tool == 'thulac':
                segment_model_path = os.path.join(root_path, self.config_args['thulac_path'])
                self.tokenizer = self.tokenizer_dict[tokenizer_tool](
                    segment_model_path=segment_model_path, seg_only=seg_only)
            if tokenizer_tool == 'jieba':
                segment_model_path = os.path.join(root_path, self.config_args['jieba_path'])
        else:
            self.tokenizer = tokenizer.CharacterTokenizer()

    def get_formed_data(self, call_back=None):
        nrows = self.sheet_nrows
        lines = []
        for line_num in range(1, nrows):
            # line = self.sheet.row_values(line_num)
            line = [line if line else 0 for line in self.sheet.row_values(line_num)]
            if not any(line):
                continue
            label_col_num = sorted([v for k, v in self.kwargs.items() if re.match('^label.+', k)])
            line = [line[self.kwargs['content']].strip().replace('\n', ''), [
                int(float(line[label])) for label in label_col_num if line[label]]]
            lines.append(line)
        if call_back:
            lines = call_back(lines)
        # [[content, [label1,label2]], , [content,[lable1,label2,label3]]] content='我 在 人民广场 吃 炸鸡'
        return lines

    def get_cuted_data(self, write=True, call_back=None):
        file_name = os.path.join(self.save_path, self.model_name,
                                 self.model_name + '_cut_result.txt')
        dir_path, _ = os.path.split(file_name)
        np.random.seed(8899)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        lines = self.get_formed_data()
        m = Manager()
        d = m.list()
        tmp_n = int(math.ceil(len(lines) / self.max_pro))
        tmp_lines = [lines[i:i + tmp_n] for i in range(0, len(lines), tmp_n)]

        tbar = tqdm(total=len(lines))

        def cut_func(*lines):
            cut_tmp_lines = []
            n_update = np.random.randint(20, 50)
            for line in lines:
                cut_tmp_lines.append([self.tokenizer.cut(line[0]), line[1]])
                if len(cut_tmp_lines) % n_update == 0:
                    d.extend(cut_tmp_lines)
                    cut_tmp_lines = []
            d.extend(cut_tmp_lines)

        def bar_func():
            l0 = 0
            while l0 < len(lines):
                time.sleep(1.0)
                tbar.update(len(d) - l0)
                l0 = len(d)

        process_list = []
        for ls in tmp_lines:
            p = Process(target=cut_func, args=(ls))
            p.start()
            process_list.append(p)
        p = Process(target=bar_func)
        p.start()
        process_list.append(p)
        for p in process_list:
            p.join()

        lines = sorted(list(d))
        if call_back:
            lines = call_back(lines)
        if write:
            with open(file_name, 'w') as f:
                for line in lines:
                    f.write(line[0] + '\t' + '\t'.join([str(i) for i in line[1]]) + '\n')
                print('Write {} Success!'.format(file_name))
            file_name = os.path.join(self.save_path, self.model_name,
                                     self.model_name + '_cut_result.pkl')
            with open(file_name, 'wb') as f:
                pkl.dump(lines, f)
        return lines

    def excel2raw(self, write=True, call_back=None):
        file_name = os.path.join(self.save_path, self.model_name,
                                 self.model_name + '_excel2raw.txt')
        dir_path, _ = os.path.split(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        lines = self.get_formed_data()
        if call_back:
            lines = call_back(lines)
        if write:
            with open(file_name, 'w') as f:
                for line in lines:
                    f.write(line[0] + '\t' + '\t'.join([str(i) for i in line[1]]) + '\n')
                print('Write {} Success!'.format(file_name))
        return lines

    def excel2id(self, modify_vocab=utils.default4vocab, write=True, call_back=None, reflush=False):
        file_name = os.path.join(self.save_path, self.model_name, self.model_name + '_excel2id.pkl')
        dir_path, _ = os.path.split(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # load vocab_inword
        if not reflush:
            load_file_name = os.path.join(self.save_path, self.model_name,
                                          self.model_name + '_vocab_inword.pkl')
            if os.path.exists(load_file_name):
                with open(load_file_name, 'rb') as f:
                    vocab_dict = pkl.load(f)
            else:
                print('Not Found {}, is reflushing!'.format(load_file_name))
                vocab_dict = self.build_vocab_inword(
                    modify_vocab=modify_vocab, write=True, call_back=call_back)
        else:
            vocab_dict = self.build_vocab_inword(
                modify_vocab=modify_vocab, write=True, call_back=call_back)
        # load cuted_data
        if not reflush:
            load_file_name = os.path.join(self.save_path, self.model_name,
                                          self.model_name + '_cut_result.pkl')
            if os.path.exists(load_file_name):
                with open(load_file_name, 'rb') as f:
                    lines = pkl.load(f)
            else:
                print('Not Found {}, is reflushing!'.format(load_file_name))
                lines = self.get_cuted_data(write=True, call_back=call_back)
        else:
            lines = self.get_cuted_data(write=True, call_back=call_back)

        labels_dict = self.build_label_class_mapping(write=False, call_back=None)
        def mapping_label(labels_list):
            labels_list = [labels_dict[label] for label in labels_list]
            label_id = [1 if i in labels_list else 0 for i in range(len(labels_dict))]
            return label_id
        lines = [[[vocab_dict.get(word, vocab_dict['OOV']) for word in line[0].split(' ')], mapping_label(line[1])]
                 for line in lines]
        if write:
            with open(file_name, 'wb') as f:
                pkl.dump(lines, f)
            print('Write {} Success!'.format(file_name))
        return lines  # [[1,2,3,4...word_id],[1,0...label_id]...]

    def build_vocab_inword(self, modify_vocab=None,  write=True, call_back=None, reflush=False):
        file_name = os.path.join(self.save_path, self.model_name,
                                 self.model_name + '_vocab_inword.pkl')
        dir_path, _ = os.path.split(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not reflush:
            load_file_name = os.path.join(self.save_path, self.model_name,
                                          self.model_name + '_cut_result.pkl')
            if os.path.exists(load_file_name):
                with open(load_file_name, 'rb') as f:
                    lines = pkl.load(f)
            else:
                print('Not Found {}, is reflushing!'.format(load_file_name))
                lines = self.get_cuted_data(write=True, call_back=call_back)
        else:
            lines = self.get_cuted_data(write=True, call_back=call_back)

        vocab_dict = {}
        if modify_vocab:
            vocab_dict.update(modify_vocab())
        vocab_list = []
        for line in lines:
            vocab_list.extend(line[0].strip().split(' '))
        vocab_list = sorted(list(set(vocab_list)))
        for word in vocab_list:
            vocab_dict.setdefault(word, len(vocab_dict))
        if write:
            with open(file_name, 'wb') as f:
                pkl.dump(vocab_dict, f)
            print('Write {} Success!'.format(file_name))
        return vocab_dict

    def build_word_embedding(self, word_embedding_name='thulac', word_embedding_dims=100, modify_vocab=None, write=True, call_back=None, reflush=False):
        file_name = os.path.join(self.save_path, self.model_name,
                                 self.model_name + '_word_embedding.pkl')
        dir_path, _ = os.path.split(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        word_vec = {}
        if word_embedding_name == 'thulac':
            word_embedding_path = os.path.join(root_path, self.config_args[
                                               'thulac_word_embedding_path'])
        # if word_embedding_name == 'thulac':
        #     word_embedding_path = self.config_args['thulac_word_embedding_path']
        with open(word_embedding_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                v = [float(line[i]) for i in range(1, len(line))]
                word_vec[line[0]] = np.asarray(v)
        word_embedding = list()
        word_embedding_rand = list()
        word_embedding_pretrained = 0
        # get vocab_dict
        if not reflush:
            load_file_name = os.path.join(self.save_path, self.model_name,
                                          self.model_name + '_vocab_inword.pkl')
            if os.path.exists(load_file_name):
                with open(load_file_name, 'rb') as f:
                    vocab_dict = pkl.load(f)
            else:
                print('Not Found {}, is reflushing!'.format(load_file_name))
                vocab_dict = self.build_vocab_inword(
                    write=True, modify_vocab=modify_vocab, call_back=call_back)
        else:
            vocab_dict = self.build_vocab_inword(
                write=True, modify_vocab=modify_vocab, call_back=call_back)
        vocab_list = sorted(vocab_dict.items(), key=lambda x: x[1])  # sorted?
        # get maxlen
        if not reflush:
            load_file_name = os.path.join(self.save_path, self.model_name,
                                          self.model_name + '_cut_result.pkl')
            if os.path.exists(load_file_name):
                with open(load_file_name, 'rb') as f:
                    lines = pkl.load(f)
            else:
                print('Not Found {}, is reflushing!'.format(load_file_name))
                lines = self.get_cuted_data(write=True, call_back=call_back)
        else:
            lines = self.get_cuted_data(write=True, call_back=call_back)
        maxlen = max([len(line[0].split()) for line in lines])
        for i in range(len(vocab_list)):
            if i == 0:
                word_embedding.append(np.zeros(word_embedding_dims, dtype='float32'))
                word_embedding_rand.append(np.zeros(word_embedding_dims, dtype='float32'))
            else:
                if vocab_list[i][0] in word_vec:
                    word_embedding.append(word_vec[vocab_list[i][0]])
                    word_embedding_pretrained += 1
                else:
                    word_embedding.append(np.random.uniform(-0.25, 0.25, word_embedding_dims))
                word_embedding_rand.append(np.random.uniform(-0.25, 0.25, word_embedding_dims))
        print('{}/{} in Pre-trained Word Embeddings.'.format(word_embedding_pretrained, len(word_embedding)))
        word_embedding = {"pretrain": {"word_embedding": word_embedding},
                          "random": {"word_embedding": word_embedding_rand},
                          "maxlen": maxlen}
        if write:
            with open(file_name, 'wb') as f:
                pkl.dump(word_embedding, f)
            print('Write {} Success!'.format(file_name))
        return word_embedding

    def build_label_class_mapping(self, write=True, call_back=None, reflush=False):
        file_name = os.path.join(self.save_path, self.model_name,
                                 self.model_name + '_label_class_mapping.pkl')
        dir_path, _ = os.path.split(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not reflush and os.path.exists(file_name):
            labels_dict = pkl.load(open(file_name, 'rb'))
            print('Load {} directly!'.format(file_name))
        else:
            label_col_num = sorted([v for k, v in self.kwargs.items() if re.match('^label.+', k)])
            labels_list = [int(float(label))
                           for col in label_col_num for label in self.sheet.col_values(col)[1:] if label]  # [1:]去标题
            labels_list = list(set(labels_list))
            labels_list.sort()
            if call_back:
                call_back(labels_list)
            labels_dict = dict(zip(labels_list, range(len(labels_list))))
            if write:
                with open(file_name, 'wb') as f:
                    pkl.dump(labels_dict, f)
                print('Write {} Success!'.format(file_name))
        return labels_dict

    def build_cross_validation(self, file_num=10, modify_vocab=None, call_back=None, reflush=False):
        np.random.seed(8899)
        if not reflush:
            load_file_name = os.path.join(self.save_path, self.model_name,
                                          self.model_name + '_excel2id.pkl')
            if os.path.exists(load_file_name):
                with open(load_file_name, 'rb') as f:
                    lines = pkl.load(f)
            else:
                print('Not Found {}, is reflushing!'.format(load_file_name))
                lines = self.excel2id(write=True, modify_vocab=modify_vocab, call_back=call_back)
        else:
            lines = self.excel2id(write=True, modify_vocab=modify_vocab, call_back=call_back)
        utils.cv(model_name=self.model_name, save_path=self.save_path,
                 lines=lines, write=True, segment_num=file_num)

    @property
    def sheet_nrows(self):
        return self.sheet.nrows

    @property
    def sheet_ncols(self):
        return self.sheet.ncols

    @property
    def sheet_head(self):
        return self.sheet.row_values(0)

    def __repr__(self):
        row0 = '\t\t\t'.join([str(item)[:30] for item in self.sheet.row_values(0)])
        row1 = '\t'.join([str(item)[:30] for item in self.sheet.row_values(1)])
        row2 = '\t'.join([str(item)[:30] for item in self.sheet.row_values(2)])
        row3 = '\t'.join([str(item)[:30] for item in self.sheet.row_values(3)])
        rown = '\t'.join([str(item)[:30] for item in self.sheet.row_values(self.sheet_nrows - 1)])
        sheet_info = 'total:\trow:' + str(self.sheet_nrows) + '\tcol:' + str(self.sheet_ncols)
        CRLF = '\n'
        preview_data = row0 + CRLF + row1 + CRLF + row2 + CRLF + row3 + CRLF * 2 + rown + CRLF + sheet_info
        return preview_data

    __str__ = __repr__


if __name__ == '__main__':
    # test = DataProcessor('qrsd4268.xlsx', tokenizer_tool='thulac', label2=3, label3=4)
    test = DataProcessor(file_path='zzxs.xlsx', model_name='zzxs', tokenizer_tool='thulac',)
    test.excel2raw()
    # test.get_cuted_data()
    # test.get_cuted_data(call_back=utils.sub_num_name_add_padding)
    test.build_vocab_inword(modify_vocab=utils.prepro_vocab1,
                            call_back=utils.sub_num_name_add_padding, reflush=False)
    test.excel2id(modify_vocab=utils.prepro_vocab1,
                  call_back=utils.sub_num_name_add_padding, reflush=False)
    test.build_label_class_mapping()
    test.build_word_embedding(modify_vocab=utils.prepro_vocab1,
                              call_back=utils.sub_num_name_add_padding, reflush=False)
    # 做cross validation，调用get_excel2id,get_cuted_data,build_vocab_inword 一并输出
    test.build_cross_validation(modify_vocab=utils.prepro_vocab1,
                                call_back=utils.sub_num_name_add_padding, reflush=False)
