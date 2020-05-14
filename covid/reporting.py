from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as T
from IPython.display import display

from .constants import MODE_NAMES
from .model import run_model
from .utils import is_notebook

import sklearn.metrics as metrics

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

ConfusionMatrix = namedtuple('ConfusionMatrix', ['tp', 'fp', 'fn', 'tn'])

__ALL__ = [
    'get_performance_plots',
    'get_performance_stats',
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
    ax.set_ylim((0.0, 1.0))
    ax.legend()
    ax.set_ylabel('Loss')

def plot_learning_rate(ax, lr_x, lr_y):
    ax = ax.twinx()
    ax.set_yscale('log')
    ax.plot(lr_x, lr_y, ls=':', lw=1.0, c='darkturquoise', label='lr (right axis)')
    ax.set_ylim((1e-8, 1e-3))
    #ax.set_ylabel("Learning Rate")
    ax.legend()

def plot_stat(ax, x, stat, title, ylim=(0.0,1.0)):
    period = max(1, int(len(x)//50))
        
    if period > 1:
        x = np.reshape(x[:((len(x)//period)*period)],(-1,period)).mean(1)

    for i, name in enumerate(MODE_NAMES):
        if period > 1:
            y = stat[:,i]
            y_grouped = np.reshape(y[:((len(y)//period)*period)],(-1,period))
            y = y_grouped.mean(1)
            yerr = np.abs(np.percentile(y_grouped, axis=1, q=[0.05,0.95]) - y)
            ax.errorbar(
                x, 
                y,
                lw=(2.0 if name == 'Inhibition' else 0.5), 
                yerr=yerr, 
                elinewidth=(0.75 if name == 'Inhibition' else 0.25), 
                capsize=3, 
                capthick=(0.75 if name == 'Inhibition' else 0.25), 
                errorevery=1,
                label=name
            )
        else:
            ax.plot(x, stat[:,i], lw=(2.0 if name == 'Inhibition' else 0.5), label=name)

        ax.set_ylim(ylim)
    ax.legend(loc='best')
    ax.set_ylabel(title)

def plot_auc(ax, pred_df):
    for mode in MODE_NAMES:
        y = pred_df[mode]
        y_hat = pred_df['pred_' + mode]
        fpr, tpr, threshold = metrics.roc_curve(y, y_hat)
        roc_auc = metrics.auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=(2.0 if mode == 'Indibition' else 0.5), alpha=0.6, label=f"{mode}={roc_auc:0.2f}")

    ax.plot((0,1), (0,1), lw=0.25, ls=':', color='black')
    ax.set_ylim((0,1))
    ax.set_xlim((0,1))
    ax.set_ylabel('tpr')
    ax.set_xlabel('fpr')


def get_performance_stats(validation_stats):
    valid_x, valid_y, valid_acc, valid_conf = zip(*validation_stats)
    tp, fp, fn, tn = zip(*valid_conf)
    tp, fp, fn, tn = np.stack(tp), np.stack(fp), np.stack(fn), np.stack(tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp/np.maximum(tp+fp, 1e-5)
    recall = tp/np.maximum(tp+fn, 1e-5)
    f1 = 2*tp/np.maximum(2*tp + fp + fn, 1e-5)
    mcc = (tp*tn-fp*fn)/np.sqrt(np.maximum((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn),1e-5))

    return {
        'epoch': valid_x, 
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall,
        'f1': f1, 
        'mcc': mcc
    }


def get_performance_plots(losses, validation_stats, learning_rates = None, pred_df = None, period=None):
    loss_x, loss_y = zip(*losses)
    valid_x, valid_y, valid_acc, valid_conf = zip(*validation_stats)
    
    with plt.style.context('bmh'):
        fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12,12))
        
        plot_loss(axes[0,0], loss_x, loss_y, valid_x, valid_y, period)
        
        if learning_rates is not None:
            lr_x, lr_y = zip(*learning_rates)
            plot_learning_rate(axes[0,0], lr_x, lr_y)

        stats = get_performance_stats(validation_stats)
        valid_x = stats['epoch']
        
        if pred_df is None:
            plot_stat(axes[0,1], valid_x, stats['accuracy'], 'Accuracy')
        else:
            plot_auc(axes[0,1], pred_df)

        plot_stat(axes[1,0], valid_x, stats['precision'], 'Precision')
        plot_stat(axes[1,1], valid_x, stats['recall'], 'Recall')
        plot_stat(axes[2,0], valid_x, stats['f1'], 'F1')
        plot_stat(axes[2,1], valid_x, stats['mcc'], "MCC / Pearson's Phi", ylim=(-0.5, 1.0))
        
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
