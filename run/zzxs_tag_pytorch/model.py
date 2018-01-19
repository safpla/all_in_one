# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from torch.autograd import Variable
import numpy as np

class  CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text,self).__init__()
        self.args = args
        self.norm_lim = config.norm_lim
        self.grad_lim = config.grad_lim
        self.tag_babei_embed_size = 10
        self.tag_label_embed_size = 10
        self.tag_label_sen_index_embed_size = 10

        self.pi_params=[0.95, 0]

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1  # channel_input
        Co = config.filter_num # channel_output
        Ks = config.filter_lengths
        self.embed = nn.Embedding(V, D)

        self.embed_tag_sen_index_sep_1 = nn.Embedding(2, self.tag_label_sen_index_embed_size)
        self.embed_tag_sen_index_sep_2 = nn.Embedding(2, self.tag_label_sen_index_embed_size)
        self.embed_tag_sen_index_sep_3 = nn.Embedding(2, self.tag_label_sen_index_embed_size)

        self.embed_size = D

        if self.args.tagsenindexsep:
            self.embed_size = self.embed_size + 3 * self.tag_label_sen_index_embed_size

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, self.embed_size)) for K in Ks])

        self.dropout = nn.Dropout(config.dropout_mlp)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.softmax = nn.Softmax()
        self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def init_weights(self):
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0)
        for conv in self.convs1:
            self.conv_init_weights(conv)

    def conv_init_weights(self, conv):
        conv.weight.data.uniform_(0, 0.01)
        conv.bias.data.fill_(0) 

    def forward(self, x, tag_label_sen_index, is_training):
        if self.args.no_static:
            x = Variable(x)
            x = self.embed(x) # (N,W,D)
        else:
            x = Variable(x, requires_grad=False)
            x = self.embed(x)

        if self.args.tagsenindexsep:
            x_1 = self.embed_tag_sen_index_sep_1(Variable(tag_label_sen_index[0]))
            x_2 = self.embed_tag_sen_index_sep_2(Variable(tag_label_sen_index[1]))
            x_3 = self.embed_tag_sen_index_sep_3(Variable(tag_label_sen_index[2]))
            x = torch.cat([x, x_1], dim=2)
            x = torch.cat([x, x_2], dim=2)
            x = torch.cat([x, x_3], dim=2)
            
        x = x.unsqueeze(1) # (N,Ci,W,D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        if is_training:
            x = self.dropout(x) # (N,len(Ks)*Co)
        else:
            x = x * (1 - config.dropout_mlp)

        logit = self.fc1(x) # (N,C)
        logit = self.softmax(logit)

        logit = torch.clamp(logit, min=1e-6, max=1.0 - 1e-6)
        # logit = torch.log(logit)
        # print ('logit.data.shape', logit.data.shape)

        return logit

