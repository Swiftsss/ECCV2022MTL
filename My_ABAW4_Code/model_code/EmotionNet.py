import torch
import torchvision
from torch import nn
from torch.nn import TransformerEncoderLayer
import math
from typing import Optional
from torch import Tensor
import utils
from model_code.resnet_rafdb import ResNet_RAFDB


class AUClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, seq_input):
        # self.fc = self.fc.cuda()
        bs, seq_len = seq_input.size(0), seq_input.size(1)
        weight = self.fc.weight
        bias = self.fc.bias
        seq_input = seq_input.reshape((bs*seq_len, 1, -1)) # bs*seq_len, 1, metric_dim
        weight = weight.unsqueeze(0).repeat((bs, 1, 1))  # bs,seq_len, metric_dim
        weight = weight.view((bs*seq_len, -1)).unsqueeze(-1) #bs*seq_len, metric_dim, 1
        inner_product = torch.bmm(seq_input, weight).squeeze(-1).squeeze(-1) # bs*seq_len
        inner_product = inner_product.view((bs, seq_len))
        return inner_product + bias

def Inception_V3(pretrained=True, remove_classifier=False):
	CNN = torchvision.models.inception_v3(pretrained=pretrained)
	if remove_classifier:
		layers_to_keep = [
			'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
			'maxpool1', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'maxpool2',
			'Mixed_5b','Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
			'Mixed_6d', 'Mixed_6e'
		]
		layers_to_keep = [getattr(CNN, name) for name in layers_to_keep]
		CNN = nn.Sequential(*layers_to_keep)
	return CNN

def get_backbone_from_name(name, pretrained=True, remove_classifier=True):
	if name == 'inception_v3':
		backbone =  Inception_V3(pretrained=pretrained, remove_classifier=remove_classifier)
		if remove_classifier:
			setattr(backbone, 'features_dim', 768)
			setattr(backbone, 'features_width', 17)
		else:
			setattr(backbone, 'features_dim', 2048)
			# remove the AuxLogits
			if backbone.AuxLogits is not None:
				backbone.AuxLogits = None
		return backbone
	if name == 'resnet18':
		resnet = ResNet_RAFDB()
		msceleb_model = torch.load('/data4/sunhaiyang/transformers/resnet18_msceleb.pth')
		resnet.load_state_dict(msceleb_model['state_dict'], strict=False)
		features = nn.Sequential(*list(resnet.children())[:-2])
		return features

class PositionalEncoding(nn.Module):
    def __init__(self,
        emb_size: int,
        dropout = float,
        maxlen: int = 5000,
        batch_first = False):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000)/emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) #(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2) #(maxlen, 1, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        self.batch_first = batch_first

    def forward(self, token_embedding):
        if self.batch_first:
            return self.dropout(token_embedding +
                self.pos_embedding.transpose(0,1)[:,:token_embedding.size(1)])
        else:
            return self.dropout(token_embedding +
                self.pos_embedding[:token_embedding.size(0), :])


class TransformerEncoderLayerCustom(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attention_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention_weights

class Attention_Metric_Module(nn.Module):
    def __init__(self, in_channels: int,
        metric_dim:int):
        super(Attention_Metric_Module, self).__init__()
        
        # self.name = name
        self.in_channels = in_channels
        self.metric_dim =  metric_dim
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=self.metric_dim, 
            kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.metric_dim),
            nn.ReLU())
        self.Conv2 = nn.Sequential(
            nn.Conv2d(self.metric_dim, out_channels =self.metric_dim, 
                kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm2d(self.metric_dim),
            nn.ReLU())
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.metric_dim, self.metric_dim, bias=True)

    def forward(self, x, attention):
        x = x*attention + x
        x = self.Conv1(x) # bsxn_outxWxW
        x = self.Conv2(x) # bs x n_out x 1 x 1
        x = self.GAP(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x

class EmotionNet(nn.Module):
	def __init__(self, args, backbone_feature_dim) -> None:
		super().__init__()
		self.AU_nums = args.AU_nums
		self.AU_metric_dim = args.AU_metric_dim
		self.backbone_feature_dim = backbone_feature_dim
		self.AU_hidden_channels = int(args.AU_nums*args.hidden_AU_ratio)
		self.init_channel = 768
		if args.feature_extractor == 'resnet':
			self.init_channel = 512
		self.AU_attention_convs = nn.Sequential(
            nn.Conv2d(self.init_channel, out_channels=self.AU_hidden_channels,
            kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.AU_hidden_channels),
            nn.ReLU(),
            nn.Conv2d(self.AU_hidden_channels, out_channels = args.AU_nums,
                kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(args.AU_nums),
            nn.ReLU())
		self.AU_attention_map_module = nn.Sequential(
            nn.Conv2d(args.AU_nums, out_channels=args.AU_nums,
                kernel_size = 1, stride=1, padding=0, bias=True),
            nn.Sigmoid())
		self.AU_attention_classification_module=nn.Sequential(
            nn.Conv2d(args.AU_nums, out_channels=args.AU_nums,
                kernel_size = 1, stride=1, padding=0, bias=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten())
		self.AU_Metric_Modules = []
		for au_name in range(args.AU_nums):
			self.AU_Metric_Modules.append(
				Attention_Metric_Module(self.backbone_feature_dim, args.AU_metric_dim)
			)
		self.AU_Metric_Modules = nn.ModuleList(self.AU_Metric_Modules)
		self.positional_encoding = PositionalEncoding(
			args.AU_metric_dim, dropout=args.all_drop, batch_first = True
		)
		self.AU_MHA = TransformerEncoderLayerCustom(
			d_model = args.AU_metric_dim, nhead = args.n_heads, dim_feedforward = 1024,
			activation='gelu', batch_first=True
		)
		self.other_MHA = TransformerEncoderLayerCustom(
			d_model = args.AU_metric_dim, nhead = args.n_heads, dim_feedforward = 1024,
			activation='gelu', batch_first=True
		)
		
		# self.transformation_matrices = []
		# for i_au in range(self.AU_nums):
		# 	matrix = nn.Linear(self.AU_metric_dim, self.AU_metric_dim, bias=False)
		# 	self.transformation_matrices.append(matrix)
		# self.transformation_matrices = nn.ModuleList(self.transformation_matrices)
		# # self.transformation_matrices = utils.get_parameters((1, self.AU_nums, self.AU_metric_dim, self.AU_metric_dim))

		# self.emotion_classifiers = {}
		# for task in ['EXPR', 'AU', 'VA']:
		# 	if task =='EXPR':
		# 		self.emotion_classifiers[task] = nn.Linear(args.AU_metric_dim, args.emotion_class_nums)
		# 	elif task == 'AU':
		# 		self.emotion_classifiers[task] = AUClassifier(args.AU_metric_dim, 12) # AU_nums
		# 	elif task == 'VA':
		# 		self.emotion_classifiers[task] = nn.Linear(args.AU_metric_dim, 2)
	
	def set_dict_cuda(self):
		for key in self.emotion_classifiers.keys():
			self.emotion_classifiers[key] = self.emotion_classifiers[key].cuda()

	def forward(self, feature_maps):
		x1 = self.AU_attention_convs(feature_maps) # (batch, AU_nums, 5, 5)
		atten_maps = self.AU_attention_map_module(x1) # (batch, AU_nums, 5, 5)
		atten_preds = self.AU_attention_classification_module(x1) # (batch, AU_nums)

		metrics = {}
		AU_metrics = []
		au_names = []
		outputs = {}
		for i_au, au_module in enumerate(self.AU_Metric_Modules):
			atten_map = atten_maps[:, i_au, ...].unsqueeze(1)
			au_metric = au_module(feature_maps, atten_map)
			# au_name = au_module.name
			AU_metrics.append(au_metric)
			# au_names.append(au_name)
		
		AU_metrics = torch.stack(AU_metrics, dim=1) # (batch, AU_nums, 16)
		input_seq = self.positional_encoding(AU_metrics)
		AU_output_seq = self.AU_MHA(input_seq)[0]
		
		''''''''''''
		other_output_seq = self.other_MHA(input_seq)[0]
		outputs = {}
		outputs['AU_metrics'] = AU_metrics
		outputs['AU_output_seq'] = AU_output_seq
		outputs['other_output_seq'] = other_output_seq
		return outputs

		# AU_metrics_with_labels = output_seq[:, :12, :]

		# outputs['AU'] = self.emotion_classifiers['AU'](AU_metrics_with_labels) # (batch, AU_nus)
		# metrics['AU'] = AU_metrics_with_labels

		# EXPR_VA_metrics = []
		# for i_au in range(self.AU_nums):
		# 	au_metric = AU_metrics[:, i_au, ...]
		# 	projected = self.transformation_matrices[i_au](au_metric)
		# 	EXPR_VA_metrics.append(projected)
		# EXPR_VA_metrics = torch.stack(EXPR_VA_metrics, dim=1) # bs, numeber of regions, dim
		# # EXPR_VA_metrics = (AU_metrics.unsqueeze(2)@self.transformation_matrices).squeeze(2) # 进行一次线性转换 (batch, AU_nums, 16)

		# if not self.avg_features:
		# 	bs, length = EXPR_VA_metrics.size(0), EXPR_VA_metrics.size(1)
        #     # EXPR classifier
		# 	outputs['EXPR'] = self.emotion_classifiers['EXPR'](EXPR_VA_metrics.view((bs*length, -1))).view((bs, length, -1))
		# 	metrics['EXPR'] = EXPR_VA_metrics

        #     # VA classifier
		# 	outputs['VA'] = self.emotion_classifiers['VA'](EXPR_VA_metrics.view((bs*length, -1))).view((bs, length, -1))
		# 	metrics['VA'] = EXPR_VA_metrics
		# else:
		# 	EXPR_VA_metrics = EXPR_VA_metrics.mean(1) # (batch, dim)
        #     # EXPR classifier
		# 	outputs['EXPR'] = self.emotion_classifiers['EXPR'](EXPR_VA_metrics)
		# 	metrics['EXPR'] = EXPR_VA_metrics

        #     # VA classifier
		# 	outputs['VA'] = self.emotion_classifiers['VA'](EXPR_VA_metrics)
		# 	metrics['VA'] = EXPR_VA_metrics
		# return outputs, metrics
