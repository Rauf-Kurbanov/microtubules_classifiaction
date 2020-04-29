import itertools
import re
import cv2
import random
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
from pathlib import Path
from modules.utils import Mode, fig_to_pil
from typing import Dict
from catalyst.dl import utils


MODE_TO_CLASS = {Mode.ZERO_VS_ZERO_ONE: ["01Taxol", "Control"],
                 Mode.ZERO_VS_ONE: ["1Taxol", "Control"],
                 Mode.ZERO_ONE_VS_ONE: ["01Taxol", "1Taxol"],
                 Mode.ZERO_VS_ZERO_ONE_VS_ONE: ["01Taxol", "1Taxol", "Control"]}


class ConfusionMatrixCallback(Callback):
    def __init__(self, mode, loader_name="valid"):
        super().__init__(CallbackOrder.Internal)
        self._class_names = MODE_TO_CLASS[mode]
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
        wandb.log({"confusion_matrix": [wandb.Image(fig_to_pil(f), caption="Label")]},
                  step=state.global_step)

    @staticmethod
    def plot_confusion_matrix(correct_labels, predict_labels, labels, normalize=False):
        cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
        if normalize:
            cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm, copy=True)
            cm = cm.astype('int')

        np.set_printoptions(precision=2)
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
        self._class_names = MODE_TO_CLASS[mode]
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
        wandb.log({"decomposition":  [wandb.Image(fig_to_pil(f), caption="Label")]},
                  step=state.global_step)

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


class MissCallback(Callback):
    def __init__(self, mode, origin_ds, n_examples=5):
        super().__init__(CallbackOrder.Internal)
        self.n_examples = n_examples
        self._class_names = MODE_TO_CLASS[mode]
        self._missclassified = []  # TODO memory
        self.origin_ds = origin_ds

    def on_batch_end(self, state: State):
        targets = state.batch_in['targets']
        predicted = state.batch_out['logits'].argmax(1).cpu().numpy()
        images = state.batch_in['original'][:self.n_examples]
        images = images.cpu().numpy()
        t_images = state.batch_in['features'][:self.n_examples].permute(0, 2, 3, 1).cpu().numpy()
        print('images.shape', images.shape)

        for img, t_image, targ, pred, fname in zip(images, t_images, targets, predicted, state.batch_in['name']):
            if pred != targ:
                self._missclassified.append({'image': img, 't_image': t_image, 'target': targ, 'pred': pred,
                                             'name': fname})

    def on_epoch_end(self, state: State):

        n_to_sample = min(self.n_examples, len(self._missclassified))
        miss_sample = random.sample(self._missclassified, n_to_sample)

        fig, axes = plt.subplots(nrows=3, ncols=n_to_sample, figsize=(30, 15))
        if n_to_sample == 1:
            axes = np.expand_dims(axes, axis=1)

        for i in range(n_to_sample):
            miss_d = miss_sample[i]
            axes[0, i].imshow(miss_d['image'])
            axes[0, 0].set_ylabel('Cropped')
            axes[0, i].set_title(f'{self._class_names[miss_d["pred"]]} instead of {self._class_names[miss_d["target"]]}')
            p = Path(miss_d['name'])
            axes[0, i].set_xlabel(f'{p.relative_to(p.parent.parent.parent)}')

            t_image = miss_d['t_image']
            m = cv2.UMat(np.zeros_like(t_image))
            ntimage = cv2.normalize(t_image, m, 0, 255, cv2.NORM_MINMAX).get()
            ntimage = ntimage.astype(np.int)
            axes[1, i].imshow(ntimage)
            axes[1, 0].set_ylabel('Transformed')
            axes[2, 0].set_ylabel('Origin')

            # TODO function
            p = Path(miss_d['name'])
            name_parts = p.name.split('_')
            ext = name_parts[-1].split('.')[1]
            new_name = f"{'_'.join(name_parts[:-1])}.{ext}"
            class_name = p.parent.stem
            origin_file = self.origin_ds / class_name / new_name
            origin_img = plt.imread(origin_file)
            axes[2, i].imshow(origin_img, cmap=plt.cm.gray)
            axes[2, i].set_xlabel(f'{origin_file.relative_to(origin_file.parent.parent.parent)}')

        wandb.log({"Miss Examples":  [wandb.Image(fig_to_pil(fig), caption="Label")]},
                  step=state.global_step)
        self._missclassified = []


class WandbCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.External)

        self.batch_log_suffix = "_batch"
        self.epoch_log_suffix = "_epoch"

    @staticmethod
    def _log_metrics(
        metrics: Dict, mode: str, step, suffix: str = "", commit: bool = True,
    ):
        def key_locate(key: str):
            """
            Wandb uses first symbol _ for it service purposes
            because of that fact, we can not send original metric names

            Args:
                key: metric name
            Returns:
                formatted metric name
            """
            if key.startswith("_"):
                return key[1:]
            return key

        metrics = {
            f"{key_locate(key)}/{mode}{suffix}": value
            for key, value in metrics.items()
        }
        print('\nlogging')
        print(metrics)
        wandb.log(metrics, commit=commit, step=step)

    def on_batch_end(self, state: State):
        mode = state.loader_name
        metrics = state.batch_metrics
        print('\n state.global_step', state.global_step)
        self._log_metrics(
            metrics=metrics,
            mode=mode,
            step=state.global_step,
            suffix=self.batch_log_suffix,
            commit=True
        )

    def on_epoch_end(self, state: State):
        mode_metrics = utils.split_dict_to_subdicts(
            dct=state.epoch_metrics,
            prefixes=list(state.loaders.keys()),
            extra_key="_base",
        )

        for mode, metrics in mode_metrics.items():
            print('mode', mode)
            self._log_metrics(
                metrics=metrics,
                mode=mode,
                step=state.global_step + 1,
                suffix=self.epoch_log_suffix,
                commit=False
            )

        wandb.log(commit=True)
