# coding=utf-8

class bcolors:
    A = '\033[0;31m'
    B = '\033[0;32m'
    C = '\033[0;33m'
    D = '\033[0;34m'
    E = '\033[0;35m'
    F = '\033[0;36m'
    G = '\033[0;37m'
    H = '\033[0;m'


class LabelDict():
    def __init__(self, index2label):
        self.index2label = index2label
        self.label2index = {v: k for k, v in index2label.items()}
        self.class_num = len(index2label)
        print(self.index2label)
        print(self.label2index)

    def encode(self, labels):
        """
        labels: a list of string ['label1', 'label2']
        """
        indexs = [0] * self.class_num
        for label in labels:
            if label in self.label2index.keys():
                indexs[self.label2index[label]] = 1
            else:
                raise('unrecognized label')
        return indexs

    def decode(self, indexs):
        """
        indexs: one-hot label index
        """
        class_num = self.class_num
        if len(indexs) != class_num:
            raise('unmatched class numbers')
        labels = [self.index2label[i] for i in range(class_num) if indexs[i]]
        return labels


def show_error_case(model_result, features, output_file=None, multilabel=False,
                    if_print=True, index2label=None):
    """
    args:
        model_result = [
                        [[prediction], [ground_truth]],
                        [[prediction], [ground_truth]]
                       ]
        features: input features
                        [[features],
                         [features]]
    return:
        None
    """
    outputs = []
    class_num = len(model_result[0][0])
    if not index2label:
        index2label = {i: i for i in range(class_num)}

    label_dict = LabelDict(index2label)
    for result, feature in zip(model_result, features):
        prediction = result[0]
        groundtruth = result[1]
        pred_label = ' | '.join(label_dict.decode(prediction))
        gdth_label = ' | '.join(label_dict.decode(groundtruth))
        if prediction == groundtruth:
            print('gdth: {0}, pred: {1}, text: {2}'.format(gdth_label, pred_label, feature))
            outputs.append('gdth: {0}, pred: {1}, text: {2}\n'.format(gdth_label, pred_label, feature))
        else:
            print(bcolors.B + 'gdth: {0} '.format(gdth_label), end='')
            print(bcolors.A + 'pred: {0} '.format(pred_label), end='')
            print(bcolors.C + 'text: {}'.format(feature))
            outputs.append('gdth: {0}, pred: {1}, text: {2}\n'.format(gdth_label, pred_label, feature))


    if not output_file:
        return
    with open(output_file, 'w') as f:
        for line in outputs:
            f.write(line)


if __name__ == '__main__':
    index2label = {0:'a', 1:'b', 2:'c'}
    model_result = [[[1,0,0],[1,1,0]]]
    features = [['hello world']]
    show_error_case(model_result, features, multilabel=True, if_print=False,
                    index2label=index2label)
