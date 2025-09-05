import numpy as np
import torch 
import torch.nn as nn
import util

def convert_value(value):
        if isinstance(value, (np.ndarray, torch.Tensor)):  # If NumPy array or PyTorch tensor
            return value.item()  # Convert to a Python float
        elif isinstance(value, np.generic):  # If NumPy scalar
            return float(value)  # Convert to float
        return value  # Otherwise, keep as is

###########################################################################################################################################
    
def evaluate_model(model, test_set, device):
    print('Evaluation of the model\n')
    #model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    accu = 0
    loss = 0
    num_batch = 0
    with torch.no_grad():
        for x,y,_ in test_set:
            x = x.to(device)
            y_true = y.to(device)
            # jet = jet.to(device)
            
            y_pred = model(x)
            
            accu += torch.sum(y_pred.max(dim=1)[1] == y_true.max(dim=1)[1]) / len(y_true)
            loss += criterion(y_pred, torch.argmax(y_true, dim=1)).item()
            num_batch += 1

    print(util.tcols.OKBLUE,'Test Accuracy of the model : {} %'.format(accu/num_batch), util.tcols.ENDC, '\n')
    print('The Test loss of the model : {} '.format(loss/num_batch))
    
    test_accu = accu/num_batch
    test_loss = loss/num_batch
    return test_accu, test_loss

###########################################################################################################################################

def plot_evaluation_metric(name, model, test_set, plotdir, device):
    y_true_all = []
    y_pred_all = []
    for x,y,_ in test_set:
        x = x.to(device)
        y_true = y.to(device)
        y_pred = nn.Softmax(dim = 1)(model(x))
        y_true = y_true.cpu().detach()
        y_pred = y_pred.cpu().detach()
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
    roc_curve_name = 'roc_curves_' + name
    y_pred_all = torch.cat(y_pred_all, dim = 0)
    y_true_all = torch.cat(y_true_all, dim = 0)
    _, _, aucs, _  = util.roc_curves(plotdir, y_pred_all, y_true_all, roc_curve_name)
    util.dnn_output(plotdir, y_pred_all, name)
    return aucs