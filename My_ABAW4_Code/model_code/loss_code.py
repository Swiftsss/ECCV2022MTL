from numpy import NaN
import torch
from torch import nn
EPS = 1e-8
class CCCloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, y):
        vx = x - torch.mean(x) 
        vy = y - torch.mean(y) 
        rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + EPS)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + EPS)
        return 1.-ccc

class MTL_loss(nn.Module):
    def __init__(self, loss_mode):
        super().__init__()
        self.loss_mode = loss_mode
        # 8种
        expression_crossentropy_weight = [1.0366230082589472, 5.455982436882547, 7.84469696969697, 7.960281870595772, 1.3703887510339123, 3.266132211854383, 4.7536342769701605, 1.0]
        # 12类  每类0-1两种
        aus_pos_weight = [4.520974723454283, 10.814065180102915, 4.078851637007177, 2.375722921091325, 1.4993105450322954, 1.6985503460885465, 2.8574468879513124, 40.99756097560976, 33.669127516778524, 19.65040975414751, 0.4637235594973294, 8.605243584975828]
        
        self.AU_loss = nn.BCEWithLogitsLoss(weight=torch.tensor(aus_pos_weight).cuda())
        self.EXPR_loss = nn.CrossEntropyLoss(weight=torch.tensor(expression_crossentropy_weight).cuda())
        self.VA_loss = CCCloss()
        # self.VA_loss = nn.L1Loss()
    
    def forward(self, output, ans_valence, ans_arousal, ans_expression, ans_aus):
        l = [torch.tensor(0.).cuda()]*4
        if self.loss_mode == 'AU' or self.loss_mode == 'MTL':
            l[0] = self.AU_loss(output['AU'], ans_aus.float())
            
        if self.loss_mode == 'EXPR' or self.loss_mode == 'MTL':
            l[1] = self.EXPR_loss(output['EXPR'], ans_expression)
            
        if self.loss_mode == 'VA' or self.loss_mode == 'MTL':
            l[2] = self.VA_loss(output['VA'][:,0], ans_valence)
            l[3] = self.VA_loss(output['VA'][:,1], ans_arousal)
            
        sl = 0.
        for i in range(4):
            if not torch.isnan(l[i]):
                sl += l[i]
        return sl

def get_loss(args):
    if args.loss_name == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif args.loss_name == 'MTL':
        return MTL_loss(args.loss_mode)