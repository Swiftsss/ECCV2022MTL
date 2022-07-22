import torch
from torch import nn
from model_code.EmotionNet import *
import utils
import numpy as np
from model_code.backbone_code import MultiHeadAttention

class Single_model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.feature_extractor = get_backbone_from_name('inception_v3', pretrained=True, remove_classifier=True)
        self.EmotionNet = EmotionNet(args, 768)

        self.AU_metric_dim = args.AU_metric_dim

        self.temporal_u = {}
        self.temporal_u['VA'] = nn.Parameter(torch.tensor(5.))
        self.temporal_u['EXPR'] = nn.Parameter(torch.tensor(5.))
        self.temporal_u['AU'] = nn.Parameter(torch.tensor(5.))

        self.VA_MHA = MultiHeadAttention(self.AU_metric_dim, args.n_heads, self.AU_metric_dim)
        self.VA_q = utils.get_parameters((1, 1, self.AU_metric_dim))
        
        self.EXPR_MHA = MultiHeadAttention(self.AU_metric_dim, args.n_heads, self.AU_metric_dim)
        self.EXPR_q = utils.get_parameters((1, 1, self.AU_metric_dim))

        self.AU_MHA = MultiHeadAttention(self.AU_metric_dim, args.n_heads, self.AU_metric_dim)
        self.AU_q = utils.get_parameters((1, 12, self.AU_metric_dim))

        self.init_state = {}
        self.emotion_classifiers = {}
        for task in ['EXPR', 'AU', 'VA']:
            if task =='EXPR':
                self.emotion_classifiers[task] = nn.Linear(args.AU_metric_dim, args.emotion_class_nums)
            elif task == 'AU':
                self.emotion_classifiers[task] = AUClassifier(args.AU_metric_dim, 12) # AU_nums
            elif task == 'VA':
                self.emotion_classifiers[task] = nn.Linear(args.AU_metric_dim, 2)
            self.init_state[task] = nn.Parameter(torch.ones(self.AU_metric_dim))
            # self.init_state[task] = utils.get_parameters((2*args.AU_nums, self.AU_metric_dim))
    def forward(self, x, init_state=None):
        # self.EmotionNet = self.EmotionNet.cuda()
        feature_maps = self.feature_extractor(x) #(batch, 768, 5, 5)
        # outputs, metrics = self.EmotionNet(feature_maps)
        # return outputs, metrics



        outputs = self.EmotionNet(feature_maps)
        # outputs include AU_output_seq other_output_seq both [batch(seq), AU_nums, self.AU_metric_dim]
        # KV = torch.cat((outputs['AU_output_seq'], outputs['other_output_seq']), dim=1)
        KV = outputs['AU_output_seq']
        outputs = self.interaction(KV)
        outputs, new_init_state = self.temporal_smooth(outputs, init_state)

        for task in ['EXPR', 'AU', 'VA']:
            outputs[task] = self.emotion_classifiers[task](outputs[task])
        return outputs, new_init_state

    
    def interaction(self, KV):
        outputs = {}
        batch = KV.shape[0]
        outputs['VA'] = self.VA_MHA(self.VA_q.expand([batch, 1, -1]), KV, KV).reshape((batch, -1))
        outputs['EXPR'] = self.EXPR_MHA(self.EXPR_q.expand([batch, 1, -1]), KV, KV).reshape(batch, -1)
        outputs['AU'] = self.AU_MHA(self.AU_q.expand([batch, 12, -1]), KV, KV)
        return outputs
    
    def temporal_smooth(self, outputs, init_state):
        new_outputs = {}
        tasks = ['EXPR', 'AU', 'VA']
        for task in tasks:
            new_outputs[task] = torch.zeros_like(outputs[task])
        index = 0
        if init_state is None:
            init_state = self.init_state
        for key in init_state.keys():
            init_state[key] = init_state[key].cuda()
        for task in tasks:
            new_outputs[task][index] = (outputs[task][index]+self.temporal_u[task]*init_state[task])/(1+self.temporal_u[task])
        index = 1
        while index < len(outputs['VA']):
            for task in tasks:
                new_outputs[task][index] = (outputs[task][index]+self.temporal_u[task]*outputs[task][index-1])/(1+self.temporal_u[task])
            index += 1
        new_init_sate = {}
        for task in tasks:
            new_init_sate[task] = new_outputs[task][-1]
        return new_outputs, new_init_sate

    def set_gpu(self):
        for key in self.emotion_classifiers.keys():
            self.emotion_classifiers[key] = self.emotion_classifiers[key].cuda()