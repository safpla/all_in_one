import os
import re
import pickle as pkl
import json
import numpy as np
import sys
from tqdm import tqdm
import copy

root_path = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, root_path)

from all_in_one.data.data_utils import utils
from all_in_one.data.data_utils.tokenizer import ThulacTokenizer
from all_in_one.data.data_utils.data_processor import DataProcessor

def remove_special_char(text):
    text = re.sub(r'\?', '\?', text)
    text = re.sub(r'\*', '\*', text)
    text = re.sub(r'\.', '\.', text)
    text = re.sub(r'\+', '\+', text)
    text = re.sub(r'\[', '\[', text)
    text = re.sub(r'\]', '\[', text)
    text = re.sub(r'\(', '\(', text)
    text = re.sub(r'\)', '\)', text)
    text = re.sub(r'\$', '\$', text)
    text = re.sub(r'\^', '\^', text)
    return text

def preprocessor(sents, tokenizer, word2index):
    cut_sents = []
    for sent in sents:
        cut_sents.append(tokenizer.cut(sent))

    pad_sents = [[sent, []] for sent in cut_sents]
    pad_sents = utils.sub_num_name_add_padding(pad_sents)
    pad_sents = [sent[0].split() for sent in pad_sents]

    id_sents = [[word2index.get(word, word2index['OOV']) for word in sent] for sent in pad_sents]
    return id_sents

def data_generator(input_json_file, output_json_path, num_classes,
                   mode='Train_Valid_Test'):
    with open(input_json_file, 'r') as f:
        input_data = json.load(f)

    def tag_sent2doc(tag_sent):
        tag_sent = np.asarray(tag_sent)
        tag_sent = np.sum(tag_sent, axis=0)
        tag_doc = [0 if tag == 0 else 1 for tag in tag_sent]
        return tag_doc

    # load tokenizer
    segment_model_path = os.path.join(root_path, 'all_in_one/data/data_utils/thulac_models')
    thu_tokenizer = ThulacTokenizer(segment_model_path=segment_model_path,
                                              seg_only=False)
    # load word2index dictionary
    path_prefix = root_path + '/all_in_one/data/data_train/focus_hierarchical_supervision/focus_hierarchical_supervision'
    word_dict_path = path_prefix + '_vocab_inword.pkl'
    with open(word_dict_path, 'rb') as f:
        word2index = pkl.load(f)

    features = []
    num = 0
    for case in tqdm(input_data):
        # if num > 100:
        #     continue
        # num += 1
        dfdt_input = case['candidate_sent_dfdt']
        court_input = case['candidate_sent_court']
        content = case['content']
        infos = case['info']

        # defendant_paragraph
        sent_tags = {}
        for sent in dfdt_input:
            sent_tags[sent] = [0] * num_classes

        found = [0] * len(infos)
        for ind, info in enumerate(infos):
            tag = info['tag']
            span = info['span']
            text = content[span[0] : span[1]]
            for text_label in re.split('[;；。\n]', content[span[0] : span[1]]):
                if text_label == '':
                    continue

                text_label = remove_special_char(text_label)
                for text_origin in dfdt_input:
                    if re.search(text_label, text_origin):
                        found[ind] = 1
                        sent_tags[text_origin][tag] = 1

        dfdt_label = []
        for sent in dfdt_input:
            dfdt_label.append(sent_tags[sent])

        dfdt_input = preprocessor(dfdt_input, thu_tokenizer, word2index)
        dfdt_sl = [len(sent) for sent in dfdt_input]

        # court_paragraph
        sent_tags = {}
        for sent in court_input:
            sent_tags[sent] = [0] * num_classes

        for ind, info in enumerate(infos):
            tag = info['tag']
            span = info['span']
            text = content[span[0] : span[1]]
            for text_label in re.split('[;；。\n]', content[span[0] : span[1]]):
                if text_label == '':
                    continue

                text_label = remove_special_char(text_label)
                for text_origin in court_input:
                    if re.search(text_label, text_origin):
                        found[ind] = 1
                        sent_tags[text_origin][tag] = 1
                if found[ind] == 0:
                    print('sentence not found: {}'.format(text_label))
                    #for sent in dfdt_input:
                    #    print(sent)
                    #for sent in court_input:
                    #    print(sent)
                    #print('content:', content)
                    #exit()

        court_label = []
        for sent in court_input:
            court_label.append(sent_tags[sent])

        court_input = preprocessor(court_input, thu_tokenizer, word2index)
        court_sl = [len(sent) for sent in court_input]

        # document label
        tag_sent = copy.deepcopy(court_label)
        tag_sent.extend(dfdt_label)
        try:
            docu_label = tag_sent2doc(tag_sent)
            feature = {'dfdt_input' : dfdt_input,
                       'dfdt_label' : dfdt_label,
                       'dfdt_sl' : dfdt_sl,
                       'court_input' : court_input,
                       'court_label' : court_label,
                       'court_sl' : court_sl,
                       'docu_label' : docu_label}
            features.append(feature)
        except:
            print('errors here:', tag_sent)

    # shuffle
    features_id = [i for i in range(len(features))]
    np.random.seed(8899)
    np.random.shuffle(features_id)
    features = [features[features_id[i]] for i in range(len(features))]

    if mode == 'Train_Valid':
        # split train, valid
        features_train = features[: round(len(features) * 0.9)]
        features_valid = features[round(len(features_train) * 0.9) :]

        train_data_file = os.path.join(output_json_path, 'data_train.json')
        valid_data_file = os.path.join(output_json_path, 'data_valid.json')

        with open(train_data_file, 'w') as f:
            json.dump(features_train, f, ensure_ascii=False)
        with open(valid_data_file, 'w') as f:
            json.dump(features_valid, f, ensure_ascii=False)
    elif mode == 'Train_Valid_Test':
        # split train, valid, test
        features_train = features[: round(len(features) * 0.9)]
        features_test = features[round(len(features) * 0.9) :]
        features_valid = features_train[round(len(features_train) * 0.9) :]
        features_train = features_train[: round(len(features_train) * 0.9)]

        train_data_file = os.path.join(output_json_path, 'data_train.json')
        valid_data_file = os.path.join(output_json_path, 'data_valid.json')
        test_data_file = os.path.join(output_json_path, 'data_test.json')

        with open(train_data_file, 'w') as f:
            json.dump(features_train, f, ensure_ascii=False)
        with open(valid_data_file, 'w') as f:
            json.dump(features_valid, f, ensure_ascii=False)
        with open(test_data_file, 'w') as f:
            json.dump(features_test, f, ensure_ascii=False)
    elif mode == 'Test':
        test_data_file = os.path.join(output_json_path, 'data_test.json')
        with open(test_data_file, 'w') as f:
            json.dump(features, f, ensure_ascii=False)
    else:
        raise Exception('unrecognized mode')


if __name__ == '__main__':
    input_json_file = os.path.join(root_path, 'all_in_one/data/data_json/labeled-Focus4Project-189-2018.01.18-train.json')
    output_json_path = os.path.join(root_path, 'all_in_one/data/data_train/focus_hierarchical_supervision')
    if not os.path.exists(output_json_path):
        os.makedirs(output_json_path)
    num_classes = 20
    data_generator(input_json_file, output_json_path, num_classes,
                   mode='Train_Valid')
    input_json_file = os.path.join(root_path, 'all_in_one/data/data_json/labeled-Focus4Project-189-2018.01.22-test.json')
    data_generator(input_json_file, output_json_path, num_classes,
                   mode='Test')
