import re
import os
import xlrd
import pickle as pkl
import numpy as np

# process sents


def remove_pos(lines):  # [['我 在 人民广场 吃 炸鸡']...]
    def some_func(sents):
        sents = ' '.join([word.split('_')[0] for word in sents.split(' ')])
        return sents
    lines = [[some_func(line[0]), line[1]] for line in lines]
    return lines


def remove_signal(lines):
    def remove_func(sents):
        sents = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", sents)
        return sents
    lines = [[remove_func(line[0]), line[1]] for line in lines]
    return lines


def sub_num(lines):
    def sub_func(sents):
        pattern = re.compile('[0-9]{3,}')
        sents = pattern.sub('0000', sents)
        sents = sents.lower()
        return sents
    lines = [[sub_func(line[0]), line[1]] for line in lines]
    return lines


# def sub_name(lines):
#     def sub_func(sents):
#         names = {}
#         new_sents_list = []
#         word_property_list = sents.split(' ')
#         for word_property_one in word_property_list:
#             word_property = word_property_one.split('_')
#             if word_property[-1] == 'np':
#                 name = ''.join(word_property[:-1])
#                 if name not in names:
#                     names[name] = len(names)
#                 new_sents_list.append('name' + str(names[name]))
#             else:
#                 new_sents_list.append(word_property[0])
#         return ' '.join(new_sents_list)
#     lines = [[sub_func(line[0]), line[1]] for line in lines]
#     return lines

def sub_name_zzxs(lines):
    pre_dict0 = ['街', '镇', '乡', '村', '厂', '开发区']
    # ni, ns
    p_ns = re.compile('ns( [^ ]+_ns)+')
    p_nsnz = re.compile('ns( [^ ]+_nz)+')

    def sub_func(sents):
        sents = p_ns.sub('ns', sents)
        sents = p_nsnz.sub('ns', sents)
        sents = sents.strip().split(' ')
        l = []
        w = 0
        while w < len(sents):
            if w < len(sents) - 2:
                w0 = sents[w].split('_')
                w1 = sents[w + 1].split('_')
                w2 = sents[w + 2].split('_')
                if '事' == w0[0][-3:] and '主' == w1[0][:3]:
                    if w2[1] == 'np' or w2[1] == 'a':
                        w0[0] += '主'
                        w2[0] = w1[0][3:] + w2[0]
                        l.append('_'.join(w0))
                        l.append(w2[0] + '_np')
                        w += 2
                    else:
                        w0[0] += '主'
                        w1[0] = w1[0][3:]
                        l.append('_'.join(w0))
                        if len(w1[0]) > 0:
                            l.append('_'.join(w1))
                        w += 1
                else:
                    l.append(sents[w])
            else:
                l.append(sents[w])
            w += 1
        tmp_l = []
        # find name
        names = {}
        w = 0
        while w < len(l):
            is_error = False

            w_tmp = l[w].split('_')
            if w_tmp[-1] == 'np':
                name = ''.join(w_tmp[:-1])
                # replace ns and ni
                if w != len(l) - 1:
                    for p in pre_dict0:
                        if p == l[w + 1][:len(p)]:
                            is_error = True
                            break
                    if is_error:
                        tmp_l.append('ns')
                        w += 1
                    else:
                        tmp_l.append(''.join(w_tmp[:-1]))
                if name not in names and not is_error:
                    names[name] = len(names)
            else:
                if w_tmp[-1] != 'ni' and w_tmp[-1] != 'ns':
                    tmp_l.append(''.join(w_tmp[:-1]))
                else:
                    tmp_l.append(w_tmp[-1])
            w += 1
        # replace name
        from jellyfish import jaro_distance
        keys = sorted(names.keys(), key=lambda x: len(x), reverse=True)
        for k0 in keys:
            for k1 in keys:
                if jaro_distance(k0, k1) > 0.7:
                    names[k1] = names[k0]

        l_out = []
        i = 0
        while i < len(tmp_l) - 1:
            ws_old = tmp_l[i] + tmp_l[i + 1]
            is_find = False
            i += 1
            for name in sorted(names.keys(), key=lambda x: len(x), reverse=True):
                try:
                    idx = ws_old.index(name)
                    if len(ws_old[:idx]) > 0:
                        l_out.append(ws_old[:idx])
                    l_out.append('name' + str(names[name]))
                    if len(ws_old[idx + len(name):]) > 0:
                        l_out.append(ws_old[idx + len(name):])
                    i += 1
                    is_find = True
                    break
                except:
                    continue
            if not is_find:
                l_out.append(tmp_l[i - 1])
                if i == len(tmp_l) - 1:
                    l_out.append(tmp_l[-1])
        t_out = ' '.join(l_out)
        return t_out
    lines = [[sub_func(line[0]), line[1]] for line in lines]
    return lines


def add_padding(lines):
    def add_func(sents):
        sents = 'padding ' * 4 + sents + ' padding' * 4
        return sents
    lines = [[add_func(line[0]), line[1]] for line in lines]
    return lines


def sub_num_name_add_padding(lines):
    lines = sub_name_zzxs(lines)
    lines = sub_num(lines)
    lines = add_padding(lines)
    return lines


def sub_num_name_add_padding_tag(lines):
    lines = sub_name_zzxs(lines)
    lines = sub_num(lines)
    lines = add_tag_padding(lines)
    return lines


def add_tag_padding(lines):
    '''
    add padding and add tag index sequence after the origin text

    '''
    def add_func(sent):
        sent = 'padding ' * 4 + sent + ' padding' * 4
        return sent

    def put_sub_str_1(substr, sen, sen_index, index_flag):
        '''
        to put the word index which suits the substr 1
        :param substr:
        :param sen:
        :param sen_index:
        :return: the changed sen_index
        '''
        sen_index = sen_index
        len_sub = len(substr.split(' '))
        start_str_ind = 0  
        while sen.find(substr, start_str_ind) > 0:
            find_start_str_ind = sen.find(substr, start_str_ind)
            before = sen[:find_start_str_ind]
            find_start_word_ind = len(before.strip().split(' '))
            find_end_word_ind = len_sub + find_start_word_ind

            for i in range(find_start_word_ind, find_end_word_ind):
                sen_index[i] = index_flag
            start_str_ind = find_start_str_ind + 1
        return sen_index

    def put_tag(sen):
        '''
        :param sen:
        :return:tag_list, sen_index_tag/string split by ' '
            the sen_index_tag has the same length with the sen.
            in sen_index_tag: every word_index belongs 000-111, every bit word_index[i] indicates the word is in key_words[i] or not
        '''
        def put_tag_122(sen):
            '''
            if donot suit the regrex, cannot be 122
            :param sen: the sen
            :return: tag/int, find_all/list
            tag:0/1
            find_all: all key_words for tag122
            '''
            import re
            find_all = []
            find_flag = False
            tag = 0
            # pattern1 = '(?<!一个|一名|一位)[23456789两二三四五六七八九十几0][^，一号日甲（）：.:；：。,、]*?(?:小伙子|外地工|男人|男的|男子|男青年|男性|年青人|年轻人|女子|外地人|同伙|可疑人员|嫌疑人员|同案人员|青年|案犯|男员工)'
            pattern1 = '[^ ]*?(?<!一个|一名|一位)[23456789两二三四五六七八九十几][^，一号日甲时（）：.:；：。,、]*?' \
                       '(?:小[ ]?伙[ ]?子|外[ ]?地[ ]?工|男[ ]?人|男[ ]?的|男[ ]?子|男[ ]?青[ ]?年|男[ ]?性|年[ ]?青[ ]?人|' \
                       '年[ ]?轻[ ]?人|女[ ]?子|外[ ]?地[ ]?人|同[ ]?伙|可[ ]?疑[ ]?人[ ]?员|嫌[ ]?疑[ ]?人[ ]?员|同[ ]?案[ ]?人[ ]?员|青[ ]?年|案[ ]?犯|男[ ]?员[ ]?工)[^ ]*?'
            # pattern2 = '[23456789二三四五六七八九十几0数多两]+[来多余]?[个名]?(?:作案者|人|嫌疑人|小偷|犯罪嫌疑人|违法嫌疑人|盗窃嫌疑人)'
            pattern2 = '[^ ]*?[23456789二三四五六七八九十几0数多两]+[来多余]?[ ]?[个名]?(?:作[ ]?案[ ]?者|人|嫌[ ]?疑[ ]?人|小[ ]?偷|犯[ ]?罪[ ]?嫌[ ]?疑[ ]?人|违[ ]?法[ ]?嫌[ ]?疑[ ]?人|盗[ ]?窃[ ]?嫌[ ]?疑[ ]?人)[^ ]*'
            # pattern3 = '一伙|伙同|一帮|多名案犯|一群|一大帮|部分村民|合伙|结伙|带人|叫人'
            pattern3 = '[^ ]*?一[ ]?伙|伙[ ]?同|一[ ]?帮|多[ ]?名[ ]?案[ ]?犯|一[ ]?群|一[ ]?大[ ]?帮|部[ ]?分[ ]?村[ ]?民|合[ ]?伙|结[ ]?伙|带[ ]?人|叫[ ]?人[^ ]*?'
            # pattern4 = '[一二三两][男女][一二三两][男女]'
            pattern4 = '[^ ]*?[一二三两][ ]?[男女][ ]?[一二三两][ ]?[男女][^ ]*?'
            # pattern5 = '被[23456789二三四五六七八九十几0数多两拾]'
            pattern5 = '[^ ]*?被[ ]?[23456789二三四五六七八九十几0数多两拾][^ ]*?'
            # pattern6 = '被[^（）：.:；：。,，群众]*等人'
            pattern6 = '[^ ]?被[ ]?[^（）：.:；：。,，群众]*等[ ]?人[^ ]?'
            # pattern7 = '(?<!将)[23456789二三四五六七八九十几0数多两]+[个名][^，（）：.:；：。,、]*(?:人|嫌疑人|小偷|犯罪嫌疑人|违法嫌疑人|盗窃嫌疑人|女子|老乡)'
            pattern7 = '[^ ]*(?<!将)[23456789二三四五六七八九十几0数多两]+[ ]?[个名][^，（）：.:；：。,、]*(?:人|嫌疑人|小偷|犯罪嫌疑人|违法嫌疑人|盗窃嫌疑人|女子|老乡)[^ ]*'
            # pattern8 = '叫来[^。：（(:]*殴打'
            pattern8 = '[^ ]*?叫[ ]?来[^。：（(:]*殴[ ]?打[^ ]*?'
            # pattern9 = '(?:查获|抓获|遭|查处)[^,，.。]*[23456789两二三四五六七八九十几0][名个]'
            pattern9 = '[^ ]*?(?:查[ ]?获|抓[ ]?获|遭|查[ ]?处)[^,，.。]*[23456789两二三四五六七八九十几0][ ]?[名个][^ ]*?'
            # pattern10 = '双方[^，、]*(?:殴打|斗殴|受伤|打斗|打在一起)'
            pattern10 = '[^ ]*?双[ ]?方[^，、]*(?:殴[ ]?打|斗[ ]?殴|受[ ]?伤|打[ ]?斗|打[ ]?在[ ]?一[ ]?起)[^ ]*?'
            # pattern11 = '被[^，]*他们'
            pattern11 = '[^ ]*?被[^，]*他[ ]?们[^ ]*?'
            # pattern_other1 = '(?<=疑人|遭到)[^，。：（(:在将用]+?(?=，|等|组织|在|将|经)'
            pattern_other1 = '[^ ]*?(?<=疑人|遭到)[^，。：（(:在将用]+?(?=，|等|组[ ]?织|在|将|经)[^ ]*?'
            pattern_other2 = '赌博'
            # pattern_other3 = '(?:抓获|犯罪嫌疑人)[^（]*(?:交代|交待|审讯|审查)'
            pattern_other3 = '[^ ]*?(?:抓[ ]?获|犯[ ]?罪[ ]?嫌[ ]?疑[ ]?人)[^（]*(?:交[ ]?代|交[ ]?待|审[ ]?讯|审[ ]?查)[^ ]*?'

            pattern = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8, pattern9, pattern10, pattern11]
            for pat in pattern:
                if re.findall(pat, sen):
                    tag = 1
                    find_all.extend(re.findall(pat,sen))
            matches = re.findall(pattern_other1, (sen))
            if matches:
                for match in matches:
                    if '、' in match:
                        tag = 1
                        find_all.append(match)
            if not find_flag:
                if ('赌博' in (sen)) and ('、' in (sen)):
                    # find_flag = True
                    tag = 1
                    find_all.append('赌博')
            if not find_flag:
                matches = re.findall(pattern_other3, (sen))
                if matches:
                    for match in matches:
                        if '、' in match:
                            tag = 1
                            find_all.append(match)
            return tag, find_all

        def put_tag_110(sen):
            find_all = []
            pattern1 = re.compile('[^ ]*?被[ ]?[1一][ ]?[名个男女][^ ]*?')
            pattern1_not = re.compile('[2-9二三两四五六七八九十几][个名男女]|十来|一伙|公司|另一')
            pattern2 = re.compile('[^ ]*?嫌[ ]?疑[ ]?人[ ]?name0[^、和伙等][^ ]*?')
            pattern2_not = re.compile('name1')
            pattern_not = re.compile('伙同')
            if re.findall(pattern1, sen) and (not re.findall(pattern1_not, sen.replace(' ', ''))) and (not re.findall(pattern_not, sen.replace(' ', ''))):
                find_all.extend(re.findall(pattern1, sen))
                tag = 1
            else:
                if re.findall(pattern2, sen) and (not re.findall(pattern2_not, sen.replace(' ', ''))) and (not re.findall(pattern_not, sen.replace(' ', ''))):
                    tag = 1
                    find_all.extend(re.findall(pattern2, sen))
                else:
                    tag = 0
            return tag, find_all

        def put_tag_124(sen):
            pattern1 = re.compile('黑社会')
            find_all = re.findall(pattern1, sen)
            if find_all:
                tag = 1
            else:
                tag = 0
            return tag, find_all

        tag_list = []
        # sen = sen.replace(' ', '')
        sen = sen.strip('\n')

        flag1, find_all1 = put_tag_122(sen)
        flag2, find_all2 = put_tag_110(sen)
        flag3, find_all3 = put_tag_124(sen)

        tag_list.append('tag11' if flag1 else 'tag10')
        tag_list.append('tag21' if flag2 else 'tag20')
        tag_list.append('tag31' if flag3 else 'tag30')
        # sen_index = ['0' for _ in range(len(sen.split(' ')))]
        sen_index = []
        for find_all in [find_all1, find_all2, find_all3]:
            tmp_sen_index = ['0' for _ in range(len(sen.split(' ')))]
            for substr in find_all:
                tmp_sen_index = put_sub_str_1(substr, sen, tmp_sen_index, '1')
            if sen_index:
                for i, ind in enumerate(tmp_sen_index):
                    sen_index[i] += ind
            else:
                sen_index = tmp_sen_index
        # tag_list = ['tag1' if i ==1 else 'tag0' for i in tag_list]
        return ' '.join(tag_list), ' '.join(sen_index)


    def add_babei(sent):
        tag_list_str, tag_sen_index = put_tag(sent)
        line_all = add_func(sent) + ' tagsplit ' + '000 ' *4 + tag_sen_index + ' 000' *4
        return line_all

    lines = [[add_babei(line[0]), line[1]] for line in lines]
    return lines

# modify vocab dict


def default4vocab():
    return {'OOV': 0}


def prepro_vocab1():
    return {'padding': 0, 'OOV': 1, 'SOS': 2, 'EOS': 3}

def prepro_vocab1_tag():
    return {'padding': 0, 'OOV': 1, 'SOS': 2, 'EOS': 3, 'tagsplit':4, '000':5, '001':6, '010':7, '011':8, '100':9, '101':10,'110':11, '111':12}
    # 'tagsplit' is the separator of the origin text and the sen index.
    # 010 indicates that the word suits pattern 2, donot suit the pattern 1 and 3

# cross validation
def cv(model_name, lines, write=True, save_path='', segment_num=10):
    file_dir = os.path.join(os.path.realpath(save_path), model_name, model_name + '_cv')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    data = {i: [] for i in range(segment_num)}
    train_data_cv = {i: [] for i in range(segment_num)}
    test_data_cv = {i: [] for i in range(segment_num)}
    # if shuffle:
    #     np.random.shuffle(lines)
    np.random.shuffle(lines)
    # for line in lines:
    #     data[np.random.randint(segment_num * 10000) % segment_num].append(line)
    for l in range(len(lines)):
        data[l % segment_num].append(lines[l])
    for i in data:
        for j in data:
            if i != j:
                train_data_cv[i].extend(data[j])
            else:
                test_data_cv[i].extend(data[j])
    response_cv = {'train_data_cv': train_data_cv, 'test_data_cv': test_data_cv}
    if write:
        def write_pkl():
            for i in range(segment_num):
                with open('{}/train_data_cv_{}.pkl'.format(file_dir, i), 'wb') as f:
                    pkl.dump(response_cv['train_data_cv'][i], f)
                with open('{}/test_data_cv_{}.pkl'.format(file_dir, i), 'wb') as f:
                    pkl.dump(response_cv['train_data_cv'][i], f)

        def write_txt():
            for i in range(segment_num):
                with open('{}/train_data_cv_{}.txt'.format(file_dir, i), 'w') as f:
                    # pkl.dump(response_cv['train_data_cv'][i], f)
                    for line in response_cv['train_data_cv'][i]:
                        line = [[str(word_id) for word_id in line[0]], [str(label_id)
                                                                        for label_id in line[1]]]
                        data_str = ' '.join(line[0]) + '\t' + ','.join(line[1]) + '\n'
                        f.write(data_str)
                print('Write {} Success'.format('{}/train_data_cv_{}.txt'.format(file_dir, i)))
                with open('{}/test_data_cv_{}.txt'.format(file_dir, i), 'w') as f:
                    # pkl.dump(response_cv['train_data_cv'][i], f)
                    for line in response_cv['test_data_cv'][i]:
                        line = [[str(word_id) for word_id in line[0]], [str(label_id)
                                                                        for label_id in line[1]]]
                        data_str = ' '.join(line[0]) + '\t' + ','.join(line[1]) + '\n'
                        f.write(data_str)
                print('Write {} Success'.format('{}/test_data_cv_{}.txt'.format(file_dir, i)))
        write_txt()
    return response_cv
