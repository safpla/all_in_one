class BaseTokenizer(object):

    def __init__(self):  # FIXME
        pass

    def cut(self, sents):  # FIXME
        raise NotImplementedError("Abstract Method")


class CharacterTokenizer(BaseTokenizer):

    def __init__(self, keep_space=False):  # FIXME
        self.num = 1
        self.keep_space = keep_space

    def cut(self, sents):
        self.num += 1
        if self.keep_space:
            sents = [i.replace(' ', '_space_') for i in sents]
        else:
            sents = [i for i in sents]
        response = ' '.join(sents)
        return response

    def cut_f(self, input_file, output_file):
        fin = open(input_file, 'r')
        fout = open(output_file, 'w')
        for l in fin:
            fout.write(self.cut(l.strip()) + '\n')
        fin.close()
        fout.close()


class JiebaTokenizer(BaseTokenizer):

    def __init__(self):  # FIXME
        import jieba
        pass

    def cut(self, sents):  # FIXME
        response = jieba.posseg.cut(sents)
        response = ' '.join(str(i) for i in response)
        return response


class HanlpTokenizer(BaseTokenizer):

    def __init__(self, whichtools):  # FIXME
        from hanlp import hanlp
        pass

    def cut(self):  # FIXME
        pass


class ThulacTokenizer(BaseTokenizer):

    def __init__(self, seg_only=False, segment_model_path='thulac_models'):  # FIXME
        import thulac
        self.thu = thulac.thulac(seg_only=seg_only, model_path=segment_model_path)
        self.num = 1

    def cut(self, sents):
        self.num += 1
        response = self.thu.cut(sents, text=True)
        return response

    def cut_f(self, input_file, output_file):
        thu.cut_f(input_file, output_file)
