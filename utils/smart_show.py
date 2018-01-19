import numpy as np


def smart_show(model_result, limit_probability=0.5, relative_probability=0.0, multilabel=False):
    # model_result = [
    #         [[0.1, 0.2, 0.3, 0.5], [0, 0, 0, 1]],
    #         [[0.1, 0.25, 0.5, 0.15], [0, 0, 1, 0]]
    # ]
    return_result = []
    if multilabel:
        for r in model_result:
            return_result.append([[1 if i > limit_probability else 0 for i in r[0]], r[1]])
    else:
        for r in model_result:
            tmp = list(r[0])
            m0 = max(tmp)
            p0 = np.argmax(tmp)
            tmp[p0] = 0
            m1 = max(tmp)
            tmp = [0] * len(tmp)
            if r[0][p0] >= limit_probability or m0 - m1 > relative_probability:
                tmp[p0] = 1
            else:
                tmp[0] = 1
            return_result.append([tmp, r[1]])
    return return_result
