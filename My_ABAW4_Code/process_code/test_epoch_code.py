from sklearn.metrics import f1_score
import torch
from torch import nn
import numpy as np



EPS = 1e-8
def cal_CCC_value(x, y):
    vx = x - torch.mean(x) 
    vy = y - torch.mean(y) 
    rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + EPS)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + EPS)
    return float(ccc)

def cal_F1_value(x, y, threshold):
    x = (x>threshold)+0
    return f1_score(y, x)

def cal_EXPR_F1_value(x, y, class_index):
    x = (x==class_index)+0
    y = (y==class_index)+0
    return f1_score(y, x)


@torch.no_grad()
def test_one_epoch(model, data, args):
    model.eval()
    
    minibatch_step = args.max_time_step
    # AU (batch, AU_nums)
    # VA (batch, 2)
    # EXPR (batch, 8)
    V_CCC = []
    A_CCC = []
    AU_pred_result = []
    AU_true_result = []
    EXPR_pred_result = []
    EXPR_true_result = []
    for ret, (x, ans_valence, ans_arousal, ans_expression, ans_aus) in enumerate(data):
        x = x.squeeze(0).cuda()
        ans_valence = ans_valence.squeeze(0).cuda()
        ans_arousal = ans_arousal.squeeze(0).cuda()
        ans_expression = ans_expression.squeeze(0)
        ans_aus = ans_aus.squeeze(0)

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

        V_CCC.append(cal_CCC_value(output['VA'][:,0], ans_valence))
        A_CCC.append(cal_CCC_value(output['VA'][:,1], ans_arousal))
        AU_pred_result += list(nn.Sigmoid()(output['AU']).cpu().detach().numpy())
        AU_true_result += list(ans_aus.numpy())

        EXPR_pred_result += list(output['EXPR'].cpu().detach().numpy())
        EXPR_true_result += list(ans_expression.numpy())
        # break
        print(f'{ret} / {len(data)} has finished')
    mean_V_CCC = sum(V_CCC)/len(V_CCC)
    mean_A_CCC = sum(A_CCC)/len(A_CCC)
    AU_pred_result = np.array(AU_pred_result)
    AU_true_result = np.array(AU_true_result).astype(int)
    AU_F1_values = []
    for i in range(12):
        AU_F1_values.append(cal_F1_value(AU_pred_result[:,i], AU_true_result[:,i], 0.5))
    AU_F1 = sum(AU_F1_values)/12


    temp_EXPR_pred_result, temp_EXPR_true_result = [], []
    for i in range(len(EXPR_true_result)):
        if EXPR_true_result[i] != -1:
            temp_EXPR_pred_result.append(EXPR_pred_result[i])
            temp_EXPR_true_result.append(EXPR_true_result[i])

    EXPR_pred_result = np.array(temp_EXPR_pred_result).argmax(axis=1)
    EXPR_true_result = np.array(temp_EXPR_true_result)
    EXPR_F1_values = []
    for i in range(8):
        EXPR_F1_values.append(cal_EXPR_F1_value(EXPR_pred_result, EXPR_true_result, i))
    EXPR_F1 = sum(EXPR_F1_values)/8
    result_score = (mean_V_CCC+mean_A_CCC)/2 + AU_F1 + EXPR_F1
    print(f'|| V CCC = {mean_V_CCC:.3f} || A CCC = {mean_A_CCC:.3f} || AU F1 = {AU_F1:.3f} || EXPR F1 = {EXPR_F1:.3f} || score = {result_score:.3f}')
    return result_score