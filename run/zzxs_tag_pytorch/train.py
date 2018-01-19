import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import config
from torch.autograd import Variable
import utilizer
import numpy as np
import time
from fol import rule_but_bei

def train(train_iter, dev_iter, test_iter, model, word_embedding, args):
    if args.cuda:
        model.cuda()

    checkpoint_num = config.checkpoint_num
    timedelay_num = config.timedelay_num

    batch_size = config.batch_size
    best_label = 0
    best_loss = 100
    test_label = 0
    test_loss = 0
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-08)
    steps = 0
    batches = 0
    timedelay = 0
    start_time = time.time()
    i = 0
    while timedelay < timedelay_num:

        if i > train_iter[0].shape[0]:
            i = 0
        w_batch = utilizer.pad_list(train_iter[0][i: i + batch_size].tolist())
        w_batch = torch.from_numpy(w_batch).type(torch.LongTensor)
        # w_batch = Variable(w_batch)   in model.py write this

        x_1 = utilizer.pad_list(train_iter[2][0][i: i + batch_size].tolist())
        x_2 = utilizer.pad_list(train_iter[2][1][i: i + batch_size].tolist())
        x_3 = utilizer.pad_list(train_iter[2][2][i: i + batch_size].tolist())
        w_batch_sen_index_tag = np.concatenate([x_1[np.newaxis], x_2[np.newaxis], x_3[np.newaxis]])
        w_batch_sen_index_tag = torch.from_numpy(w_batch_sen_index_tag).type(torch.LongTensor)

        y_batch = train_iter[1][i: i + batch_size]
        y_batch = torch.from_numpy(y_batch)
        y_batch = Variable(y_batch, requires_grad=False).type(torch.FloatTensor)

        feature = w_batch
        target = y_batch
        feature_sen_index_tag = w_batch_sen_index_tag
        Y_distribution_config = Variable(torch.from_numpy(np.array(args.Y_distribution_config)).type(torch.FloatTensor), requires_grad=False)
        if args.cuda:
            feature, target, Y_distribution_config, feature_sen_index_tag = w_batch.cuda(), y_batch.cuda(), Y_distribution_config.cuda(), feature_sen_index_tag.cuda()

        optimizer.zero_grad()
        logit_p = model(feature, feature_sen_index_tag, is_training=True)  # after softmax ->logit_p
        if args.lossdistribution:
            loss_p = target *torch.log(logit_p) * Y_distribution_config + (1 - target) * torch.log(1 - logit_p)
        else:
            loss_p = target *torch.log(logit_p) + (1 - target) * torch.log(1 - logit_p)
        loss_p = torch.sum((-1)*loss_p)/loss_p.size()[0]

        loss_a = loss_p
        # loss = F.cross_entropy(logit, target)
        # cross_entropy : This criterion combines LogSoftMax and NLLLoss in one single class, in cross_entropy, x is the input without softmax
        
        i += batch_size
        batches += 1

        loss_a.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), config.grad_lim)
        optimizer.step()

        if batches % checkpoint_num == 0:
            corrects = (torch.max(logit_p, 1)[1].view([target.size()[0], 1]).data == torch.max(target, 1)[1].view([target.size()[0], 1]).data).sum()
            accuracy = 100.0 * corrects/batch_size
            
            sys.stdout.write(
                '\rBatch[{}] - loss_a: {:.6f} acc: {:.4f}%({}/{})'.format(batches,
                                                                         loss_a.data[0],
                                                                         accuracy,
                                                                         corrects,
                                                                         batch_size))

            dev_label, dev_loss, _ = eval(dev_iter, model, args)
            if best_loss > dev_loss or best_label < dev_label:
                timedelay = 0
            else:
                timedelay += 1

            if best_label < dev_label:
                best_label = dev_label
            if best_loss > dev_loss:
                best_loss = dev_loss
                test_label, test_loss, test_cases = eval(test_iter, model, args)

            sys.stdout.write("Batches: %d" % batches)
            sys.stdout.write("\tBatch Time: %.4fs" % (1.0 * (time.time() - start_time) / checkpoint_num))

            # sys.stdout.write("\nTrain Label: %.6f" % train_label)
            # sys.stdout.write("\tTrain Loss: %.6f" % train_loss)

            sys.stdout.write("\nValid Label: %.6f" % dev_label)
            sys.stdout.write("\tValid Loss: %.6f" % dev_loss)

            sys.stdout.write("\nTest Label: %.6f" % test_label)
            sys.stdout.write("\tTest Loss: %.6f" % test_loss)

            sys.stdout.write("\nBest Label: %.6f" % best_label)
            sys.stdout.write("\tBest Loss: %.6f" % best_loss)

            sys.stdout.write("\n\n")

            start_time = time.time()
    return test_label, test_cases

def eval(train_iter, model, args):
    # model.eval()
    corrects, avg_loss = 0, 0
    batches = 0
    timedelay = 0
    batch_size = config.batch_size
    i = 0
    cases = []
    # x = []
    preds = []
    targets = []
    while i<train_iter[0].shape[0]:
        w_batch = utilizer.pad_list(train_iter[0][i: i + batch_size].tolist())
        w_batch = torch.from_numpy(w_batch).type(torch.LongTensor)

        x_1 = utilizer.pad_list(train_iter[2][0][i: i + batch_size].tolist())
        x_2 = utilizer.pad_list(train_iter[2][1][i: i + batch_size].tolist())
        x_3 = utilizer.pad_list(train_iter[2][2][i: i + batch_size].tolist())
        w_batch_sen_index_tag = np.concatenate([x_1[np.newaxis], x_2[np.newaxis], x_3[np.newaxis]])
        w_batch_sen_index_tag = torch.from_numpy(w_batch_sen_index_tag).type(torch.LongTensor)

        y_batch = Variable(torch.from_numpy(train_iter[1][i: i + batch_size]), requires_grad=False).type(torch.FloatTensor)
        feature = w_batch
        target = y_batch
        feature_sen_index_tag = w_batch_sen_index_tag
        if args.cuda:
            feature, target, feature_sen_index_tag = feature.cuda(), target.cuda(), feature_sen_index_tag.cuda()
        i += batch_size
        logit_p = model(feature, feature_sen_index_tag, is_training=False)
        loss_p = target * torch.log(logit_p) + (1 - target) * torch.log(1 - logit_p)
        loss_p = torch.sum((-1)*loss_p)
        loss = loss_p

        avg_loss += loss.data[0]
        # print('loss.data', loss.data)   loss.data是一个list
        corrects += (torch.max(logit_p, 1)[1].view([target.size()[0], 1]).data == torch.max(target, 1)[1].view([target.size()[0], 1]).data).sum()
        #corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        pred = torch.max(logit_p, 1)[1].data
        target = torch.max(target, 1)[1].data
        # print(pred, 'pred')
        # print(target, 'target')

        preds.extend(pred)
        targets.extend(target)


    size = train_iter[0].shape[0]
    x = train_iter[0]
    # print('train_iter[0]', train_iter[0])
    # print('x', x)
    avg_loss = avg_loss/size
    accuracy = 100.0 * corrects/size
    # print('len(train_iter[0])', len(train_iter[0]))
    # print('len(preds), 'len(preds))
    # print('len(targets), 'len(targets))


    return accuracy, avg_loss, [x, preds, targets]


