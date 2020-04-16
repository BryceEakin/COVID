import matplotlib.pyplot as plt
import numpy as np

from .constants import MODE_NAMES
from IPython.display import display

__ALL__ = ['get_performance_plots']

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

def plot_stat(ax, x, stat, title):
    for i, name in enumerate(MODE_NAMES):
        ax.plot(x, stat[:,i], lw=(2.0 if name == 'Inhibition' else 0.5), label=name)
    ax.legend(loc='best')
    ax.set_ylabel(title)

def get_performance_plots(losses, validation_stats, period=None):
    loss_x, loss_y = zip(*losses)
    valid_x, valid_y, valid_acc, valid_conf = zip(*validation_stats)
    
    with plt.style.context('bmh'):
        fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12,12))
        
        plot_loss(axes[0,0], loss_x, loss_y, valid_x, valid_y, period)
        axes[0,0].set_ylabel('Loss')
        
        tp, fp, tn, fn = zip(*valid_conf)
        tp, fp, tn, fn = np.stack(tp), np.stack(fp), np.stack(tn), np.stack(fn)

        accuracy = np.stack(valid_acc)
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