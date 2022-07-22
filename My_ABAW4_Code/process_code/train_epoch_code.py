import utils
import time
def train_one_epoch(epoch, model, optimizer, loss, data, args, optimizer2, print_freq=100):
    model.train()
    
    start_time = time.time()
    for ret, (x, ans_valence, ans_arousal, ans_expression, ans_aus) in enumerate(data, start=epoch * len(data)):
        x = x.cuda().squeeze(0)
        ans_valence = ans_valence.cuda().squeeze(0)
        ans_arousal = ans_arousal.cuda().squeeze(0)
        ans_expression = ans_expression.cuda().squeeze(0)
        ans_aus = ans_aus.cuda().squeeze(0)
        lr = utils.adjust_learning_rate(args, optimizer, data, ret)
        if optimizer2 is not None:
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr*0.1
            optimizer2.zero_grad()
        optimizer.zero_grad()
        output, metrics = model(x)
        l = loss(output, ans_valence, ans_arousal, ans_expression, ans_aus)
        if l != 0.:
            l.backward(retain_graph=True)
        optimizer.step()
        if optimizer2 is not None:
            optimizer2.step()
        if (ret+1) % print_freq == 0:
            print(f'|| epoch {epoch} || {(ret+1)%len(data)}/{len(data)} || loss = {float(l):.4f} || lr {lr}')
        # break
    end_time = time.time()
    print(f'one epoch use {end_time-start_time:.4f}s')
