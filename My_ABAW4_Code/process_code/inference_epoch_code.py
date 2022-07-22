from sklearn.metrics import f1_score
import torch
from torch import nn
import numpy as np

@torch.no_grad()
def test_one_epoch(model, data, args):
    model.eval()
    
    minibatch_step = args.max_time_step
    # AU (batch, AU_nums)
    # VA (batch, 2)
    # EXPR (batch, 8)
    ans = {}
    for ret, (name, x) in enumerate(data):
        x = x.squeeze(0).cuda()
        # name = name.squeeze(0)

        if x.shape[0] > minibatch_step:
            i = 0
            output = None
            init_state = None
            while i < x.shape[0]:
                mini_x = x[i:min(i+minibatch_step, x.shape[0]), ...]
                mini_output, init_state = model(mini_x, init_state)
                if output != None:
                    output['VA'] = torch.concat((output['VA'], mini_output['VA']))
                    output['AU'] = torch.concat((output['AU'], mini_output['AU']))
                    output['EXPR'] = torch.concat((output['EXPR'], mini_output['EXPR']))
                else:
                    output = mini_output
                i = i+minibatch_step
        else:
            output, metrics = model(x)

        AU_pred_result = list(nn.Sigmoid()(output['AU']).cpu().detach().numpy())
        EXPR_pred_result = np.array(list(output['EXPR'].cpu().detach().numpy())).argmax(axis=1)
        for i in range(len(name)):
            tem_ans = {}
            tem_ans['valence'] = float(output['VA'][:,0][i])
            tem_ans['arousal'] = float(output['VA'][:,1][i])
            tem_ans['AU'] = list((AU_pred_result[i]>0.5)+0)
            tem_ans['EXPR'] = int(EXPR_pred_result[i])

            ans[name[i]] = tem_ans
    return ans