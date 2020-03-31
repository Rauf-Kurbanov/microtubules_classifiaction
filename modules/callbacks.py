import itertools
import re
from collections import defaultdict
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import wandb
from catalyst.dl import Callback, CallbackOrder, State
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from modules.utils import Mode, fig_to_pil


class ConfusionMatrixCallback(Callback):
    def __init__(self, mode, loader_name="valid"):
        super().__init__(CallbackOrder.Internal)
        mode_to_class = {Mode.ZERO_VS_ZERO_ONE: ["Control", "01Taxol"],
                         Mode.ZERO_VS_ONE: ["Control", "1Taxol"],
                         Mode.ZERO_ONE_VS_ONE: ["01Taxol", "1Taxol"],
                         Mode.ZERO_VS_ZERO_ONE_VS_ONE: ["Control", "01Taxol", "1Taxol"]}
        self._class_names = mode_to_class[mode]
        self.loader_name = loader_name

    def on_loader_start(self, state):
        """Prepare tensorboard writers for the current stage"""
        if state.logdir is None:
            return

        lm = state.loader_name
        log_dir = state.logdir / f"{lm}_log"
        self.logger = SummaryWriter(log_dir)

    def on_epoch_start(self, state: State):
        self.preds = []
        self.gts = []

    def on_batch_end(self, state: State):  # TODO use metrics
        if state.loader_name == self.loader_name:
            pred = state.batch_out["logits"].argmax(dim=1).cpu().numpy()
            pred = [self._class_names[x] for x in pred]
            gt = state.batch_in['targets'].cpu().numpy()
            gt = [self._class_names[x] for x in gt]

            self.preds.extend(pred)
            self.gts.extend(gt)

    def on_epoch_end(self, state: State):
        f = self.plot_confusion_matrix(self.gts, self.preds, labels=self._class_names)
        self.logger.add_figure("confusion_matrix", f, global_step=state.global_epoch)

        self.logger.flush()
        wandb.log({"confusion_matrix": [wandb.Image(fig_to_pil(f), caption="Label")]})

    @staticmethod
    def plot_confusion_matrix(correct_labels, predict_labels, labels, normalize=False):
        cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
        if normalize:
            cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm, copy=True)
            cm = cm.astype('int')

        np.set_printoptions(precision=2)
        # fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
        fig = Figure(facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, cmap='Oranges')

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
        classes = ['\n'.join(wrap(l, 40)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=4, va ='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
        fig.set_tight_layout(True)
        return fig


class EmbedPlotCallback(Callback):

    def __init__(self, mode):
        super().__init__(CallbackOrder.Internal)
        mode_to_class = {Mode.ZERO_VS_ZERO_ONE: ["Control", "01Taxol"],
                         Mode.ZERO_VS_ONE: ["Control", "1Taxol"],
                         Mode.ZERO_ONE_VS_ONE: ["01Taxol", "1Taxol"],
                         Mode.ZERO_VS_ZERO_ONE_VS_ONE: ["Control", "01Taxol", "1Taxol"]}
        self._class_names = mode_to_class[mode]
        self._n_classes = len(self._class_names)
        self._colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                        '#bcbd22', '#17becf']
        self._pca = PCA()

    def on_loader_start(self, state):
        """Prepare tensorboard writers for the current stage"""
        if state.logdir is None:
            return

        lm = state.loader_name
        log_dir = state.logdir / f"{lm}_log"
        self.logger = SummaryWriter(log_dir)

    def on_epoch_start(self, state: State):
        self.batch_outs = defaultdict(list)
        self.targets = defaultdict(list)

    def on_batch_end(self, state: State):
        ln = state.loader_name
        self.batch_outs[ln].append(state.batch_out["logits"].cpu().detach().numpy())
        self.targets[ln].append(state.batch_in['targets'].cpu().numpy())

    def on_epoch_end(self, state: State):
        train_epoch_out = np.concatenate(self.batch_outs["train"])
        self._pca.fit(train_epoch_out)

        val_epoch_out = np.concatenate(self.batch_outs["valid"])
        val_epoch_targets = np.concatenate(self.targets["valid"])
        val_epoch_embeds = self._pca.transform(val_epoch_out)

        f = self.plot_embeddings(val_epoch_embeds, val_epoch_targets)
        self.logger.add_figure("decomposition", f, global_step=state.global_epoch)
        self.logger.flush()
        wandb.log({"decomposition":  [wandb.Image(fig_to_pil(f), caption="Label")]})

    def plot_embeddings(self, embeddings, targets, xlim=None, ylim=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(self._n_classes):
            inds = np.where(targets == i)[0]
            ax.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=self._colors[i])
        if xlim:
            ax.xlim(xlim[0], xlim[1])
        if ylim:
            ax.ylim(ylim[0], ylim[1])
        ax.legend(self._class_names)

        return fig
