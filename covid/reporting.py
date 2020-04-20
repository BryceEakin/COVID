from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as T
from IPython.display import display

from .constants import MODE_NAMES
from .model import run_model
from .utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

ConfusionMatrix = namedtuple('ConfusionMatrix', ['tp', 'fp', 'fn', 'tn'])

__ALL__ = [
    'get_performance_plots',
    'calculate_average_loss_and_accuracy',
    'ConfusionMatrix'
]

def plot_loss(ax, loss_x, loss_y, valid_x, valid_y, period=None):
    x,y = loss_x, loss_y
    
    if period is None:
        period = max(1, int(len(x)//500))
        
    x = np.reshape(x[:((len(x)//period)*period)],(-1,period)).mean(1)
    y_grouped = np.reshape(y[:((len(y)//period)*period)],(-1,period))

    y = y_grouped.mean(1)
    yerr = np.abs(np.percentile(y_grouped, axis=1, q=[0.05,0.95]) - y)

    ax.errorbar(
        x, 
        y,
        lw=1, 
        yerr=yerr, 
        elinewidth=0.75, 
        capsize=3, 
        capthick=0.75, 
        errorevery=10,
        label='model (training)'
    )

    x, y = valid_x, valid_y

    ax.plot(x, y, label='model (validation)', c='C1')

    ax.plot((x[0], x[-1]), (0.422, 0.422), c='g', ls=':', lw=1.5, label='random baseline')
    ax.legend()
    ax.set_ylim((0.0, 0.5))

def plot_stat(ax, x, stat, title, ylim=(0.0,1.0)):
    for i, name in enumerate(MODE_NAMES):
        ax.plot(x, stat[:,i], lw=(2.0 if name == 'Inhibition' else 0.5), label=name)
        ax.set_ylim(ylim)
    ax.legend(loc='best')
    ax.set_ylabel(title)

def get_performance_plots(losses, validation_stats, period=None):
    loss_x, loss_y = zip(*losses)
    valid_x, valid_y, valid_acc, valid_conf = zip(*validation_stats)
    
    with plt.style.context('bmh'):
        fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12,12))
        
        plot_loss(axes[0,0], loss_x, loss_y, valid_x, valid_y, period)
        axes[0,0].set_ylabel('Loss')
        
        tp, fp, fn, tn = zip(*valid_conf)
        tp, fp, fn, tn = np.stack(tp), np.stack(fp), np.stack(fn), np.stack(tn)

        #accuracy = np.stack(valid_acc)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        plot_stat(axes[0,1], valid_x, accuracy, 'Accuracy')
        
        precision = tp/np.maximum(tp+fp, 1e-5)
        plot_stat(axes[1,0], valid_x, precision, 'Precision')
        
        recall = tp/np.maximum(tp+fn, 1e-5)
        plot_stat(axes[1,1], valid_x, recall, 'Recall')

        f1 = 2*tp/np.maximum(2*tp + fp + fn, 1e-5)
        plot_stat(axes[2,0], valid_x, f1, 'F1')

        mcc = (tp*tn-fp*fn)/np.sqrt(np.maximum((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn),1e-5))
        plot_stat(axes[2,1], valid_x, mcc, "MCC / Pearson's Phi")
        
    return fig

def calculate_average_loss_and_accuracy(model, dl, device):
    model.eval()
    
    total_loss = 0.0
    total_div = 0
    
    accuracy_acc = np.zeros(5)
    accuracy_div = np.zeros(5)
    
    tp = np.zeros(5)
    fp = np.zeros(5)
    fn = np.zeros(5)
    tn = np.zeros(5)

    out_blocks = []
    
    with T.no_grad():
        for batch in tqdm(dl):
            pair_names, chem_graphs, chem_features, proteins, target = batch
            model.zero_grad()
            result, target, loss, weight = run_model(model, batch, device)
            
            total_loss += loss.item() * result.shape[0]
            total_div += result.shape[0]
            
            accuracy_acc += (((result > 0.5) == target) * weight).sum(0).cpu().numpy()
            accuracy_div += weight.sum(0).cpu().numpy()
            
            tp += (((result >= 0.5) * target) * weight).sum(0).cpu().numpy()
            fp += (((result >= 0.5) * (1 - target)) * weight).sum(0).cpu().numpy()
            fn += (((result < 0.5) * target) * weight).sum(0).cpu().numpy()
            tn += (((result < 0.5) * (1 - target)) * weight).sum(0).cpu().numpy()

            name_blk = pd.DataFrame(pair_names, columns=['chem', 'protein'])
            target_blk = pd.DataFrame(target.detach().cpu().numpy(), columns=MODE_NAMES)
            result_blk = pd.DataFrame(result.detach().cpu().numpy(), columns=['pred_'+n for n in MODE_NAMES])

            out_blocks.append(pd.concat([name_blk, target_blk, result_blk], axis=1))
        
    model.train()
        
    return (
        total_loss / total_div, 
        accuracy_acc / accuracy_div, 
        ConfusionMatrix(tp,fp,fn,tn), 
        pd.concat(out_blocks, axis=0).reset_index(drop=True)
    )
