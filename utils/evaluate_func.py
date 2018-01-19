import numpy as np


def evaluate_func(results, multilabel=False, print_results=False):
    def printfun():
        if print_results:
            print('{:<10}\t{}'.format('Accuracy:', Accuracy))
            print('{:<10}\t{}'.format('Precision:', Precision))
            print('{:<10}\t{}'.format('Recall:', Recall))
    Accuracy_numerator = 0
    Accuracy_denominator = 0
    Precision_numerator = 0
    Precision_denominator = 0
    Recall_numerator = 0
    Recall_denominator = 0
    if multilabel:
        Accuracy_tmp = []
        Precision_tmp = []
        Recall_tmp = []
        for index in range(len(results[0][0])):
            Accuracy_numerator_tmp = 0
            Accuracy_denominator_tmp = 0
            Precision_numerator_tmp = 0
            Precision_denominator_tmp = 0
            Recall_numerator_tmp = 0
            Recall_denominator_tmp = 0
            for result in results:
                # Accuracy
                Accuracy_denominator_tmp += 1
                if result[0][index] == result[1][index]:
                    Accuracy_numerator_tmp += 1
                # Precision
                if result[0][index]:  # do prediction
                    Precision_denominator_tmp += 1
                    if result[0][index] == result[1][index]:
                        Precision_numerator_tmp += 1
                # print(Precision_denominator_tmp, Precision_numerator_tmp)
                # Recall
                if result[1][index]:
                    Recall_denominator_tmp += 1
                    if result[0][index] == result[1][index]:
                        Recall_numerator_tmp += 1
            Accuracy_denominator_tmp = Accuracy_denominator_tmp if Accuracy_denominator_tmp else 1
            Precision_denominator_tmp = Precision_denominator_tmp if Precision_denominator_tmp else 1
            Recall_denominator_tmp = Recall_denominator_tmp if Recall_denominator_tmp else 1

            Accuracy_tmp.append(Accuracy_numerator_tmp / Accuracy_denominator_tmp)
            Precision_tmp.append(Precision_numerator_tmp / Precision_denominator_tmp)
            Recall_tmp.append(Recall_numerator_tmp / Recall_denominator_tmp)
        Accuracy = [np.mean(Accuracy_tmp), Accuracy_tmp]
        Precision = [np.mean(Precision_tmp), Precision_tmp]
        Recall = [np.mean(Recall_tmp), Recall_tmp]
        printfun()
    else:
        for result in results:
            # Accuracy
            Accuracy_denominator += 1
            if result[0] == result[1]:
                Accuracy_numerator += 1
            # Precision
            if any(result[0][1:]):  # do prediction
                Precision_denominator += 1
                if result[0][1:] == result[1][1:]:
                    Precision_numerator += 1
            # Recall
            if any(result[1][1:]):  # right lables
                Recall_denominator += 1
                if result[0][1:] == result[1][1:]:
                    Recall_numerator += 1
        Accuracy_denominator = Accuracy_denominator if Accuracy_denominator else 1
        Precision_denominator = Precision_denominator if Precision_denominator else 1
        Recall_denominator = Recall_denominator if Recall_denominator else 1

        Accuracy = [Accuracy_numerator / Accuracy_denominator, []]
        Precision = [Precision_numerator / Precision_denominator, []]
        Recall = [Recall_numerator / Recall_denominator, []]
        printfun()

    return Accuracy, Precision, Recall

if __name__ == '__main__':
    # [[[prediction], [true label]], ...,]
    # evaluate_func([[[1, 0, 0], [1, 0, 0]],
    #                [[0, 1, 0], [0, 1, 0]],
    #                [[1, 0, 0], [0, 0, 1]],
    #                [[1, 0, 0], [1, 0, 0]]])
    evaluate_func([[[0, 0, 1], [0, 0, 1]],
                   [[1, 0, 1], [1, 0, 0]],
                   [[0, 0, 0], [1, 0, 1]],
                   [[0, 1, 0], [0, 0, 0]]], multilabel=True)
    # acc = (3/4 + 3/4 + 2/4) / 3
    # pre = (1/2 + 0/1 + 1/2) / 3
    # rec = (1/2 + 0/0 + 1/2) / 3
