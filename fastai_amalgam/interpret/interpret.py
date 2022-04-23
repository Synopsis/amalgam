# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/interpret_classification-interpretation.ipynb (unless otherwise specified).

__all__ = ["ClassificationInterpretationEx"]

# Cell
from typing_extensions import Literal
from fastai.vision.all import *
from fastai.metrics import *
import PIL

# Cell
from ..utils import *
from ..show_data import *

# Cell
import fastai


def _get_truths(vocab, label_idx, is_multilabel):
    if is_multilabel:
        return ";".join([vocab[i] for i in torch.where(label_idx == 1)][0])
    else:
        return vocab[label_idx]


class ClassificationInterpretationEx(ClassificationInterpretation):
    """
    Extend fastai2's `ClassificationInterpretation` to analyse model predictions in more depth
    See:
      * self.preds_df
      * self.plot_label_confidence()
      * self.plot_confusion_matrix()
      * self.plot_accuracy()
      * self.get_fnames()
      * self.plot_top_losses_grid()
      * self.print_classification_report()
    """

    def __init__(self, dl, inputs, preds, targs, decoded, losses):
        super().__init__(dl, inputs, preds, targs, decoded, losses)
        self.vocab = self.dl.vocab
        if is_listy(self.vocab):
            self.vocab = self.vocab[-1]
        if self.targs.__class__ == fastai.torch_core.TensorMultiCategory:
            self.is_multilabel = True
        else:
            self.is_multilabel = False
        self.compute_label_confidence()
        self.determine_classifier_type()

    def determine_classifier_type(self):
        if self.targs[0].__class__ == fastai.torch_core.TensorCategory:
            self.is_multilabel = False
        if self.targs[0].__class__ == fastai.torch_core.TensorMultiCategory:
            self.is_multilabel = True
            self.thresh = self.dl.loss_func.thresh

    def compute_label_confidence(self, df_colname: Optional[str] = "filepath"):
        """
        Collate prediction confidence, filenames, and ground truth labels
        in DataFrames, and store them as class attributes
        `self.preds_df` and `self.preds_df_each`

        If the `DataLoaders` is constructed from a `pd.DataFrame`, use
        `df_colname` to specify the column name with the filepaths
        """
        if not isinstance(self.dl.items, pd.DataFrame):
            self._preds_collated = [
                # (item, self.dl.vocab[label_idx], *preds.numpy()*100)\
                (
                    item,
                    _get_truths(self.dl.vocab, label_idx, self.is_multilabel),
                    *preds.numpy() * 100,
                )
                for item, label_idx, preds in zip(self.dl.items, self.targs, self.preds)
            ]
        ## need to extract fname from DataFrame
        elif isinstance(self.dl.items, pd.DataFrame):
            self._preds_collated = [
                # (item[df_colname], self.dl.vocab[label_idx], *preds.numpy()*100)\
                (
                    item[df_colname],
                    _get_truths(self.dl.vocab, label_idx, self.is_multilabel),
                    *preds.numpy() * 100,
                )
                for (_, item), label_idx, preds in zip(
                    self.dl.items.iterrows(), self.targs, self.preds
                )
            ]

        self.preds_df = pd.DataFrame(
            self._preds_collated, columns=["fname", "truth", *self.dl.vocab]
        )
        self.preds_df.insert(2, column="loss", value=self.losses.numpy())
        self.preds_df.insert(2, column="predicted_label", value=self.get_pred_labels())

        if self.is_multilabel:
            return  # preds_df_each doesnt make sense for multi-label
        self._preds_df_each = {
            l: self.preds_df.copy()[self.preds_df.truth == l].reset_index(drop=True)
            for l in self.dl.vocab
        }
        self.preds_df_each = defaultdict(dict)

        sort_desc = lambda x, col: x.sort_values(col, ascending=False).reset_index(
            drop=True
        )
        for label, df in self._preds_df_each.items():
            filt = df[label] == df[self.dl.vocab].max(axis=1)
            self.preds_df_each[label]["accurate"] = df.copy()[filt]
            self.preds_df_each[label]["inaccurate"] = df.copy()[~filt]

            # fmt: off
            self.preds_df_each[label]['accurate']   = sort_desc(self.preds_df_each[label]['accurate'], label)
            self.preds_df_each[label]['inaccurate'] = sort_desc(self.preds_df_each[label]['inaccurate'], label)
            assert len(self.preds_df_each[label]['accurate']) + len(self.preds_df_each[label]['inaccurate']) == len(df)
            # fmt: on

    def get_pred_labels(self) -> list:
        if self.is_multilabel:
            pred_idxs = list(map(lambda x: torch.where(x == 1)[0], self.decoded))
            pred_labels = list(map(lambda i: self.vocab[i], pred_idxs))
            return pred_labels
        else:
            return self.vocab[self.decoded]

    def get_fnames(
        self,
        label: str,
        mode: Literal["accurate", "inaccurate"],
        conf_level: Union[int, float, tuple],
    ) -> np.ndarray:
        """
        Utility function to grab filenames of a particular label `label` that were classified
        as per `mode` (accurate|inaccurate).
        These filenames are filtered by `conf_level` which can be above or below a certain
        threshold (above if `mode` == 'accurate' else below), or in confidence ranges
        """
        assert label in self.dl.vocab
        if not hasattr(self, "preds_df_each"):
            self.compute_label_confidence()
        df = self.preds_df_each[label][mode].copy()
        if mode == "accurate":
            if isinstance(conf_level, tuple):
                filt = df[label].between(*conf_level)
            if isinstance(conf_level, (int, float)):
                filt = df[label] > conf_level
        if mode == "inaccurate":
            if isinstance(conf_level, tuple):
                filt = df[label].between(*conf_level)
            if isinstance(conf_level, (int, float)):
                filt = df[label] < conf_level
        return df[filt].fname.values


# Cell
from palettable.scientific.sequential import Davos_3_r


@patch
def plot_confusion_matrix(
    self: ClassificationInterpretationEx,
    normalize=True,
    title="Confusion matrix",
    cmap=None,
    norm_dec=2,
    plot_txt=True,
    return_fig=False,
    dpi=100,
    figsize=(5, 5),
    **kwargs,
):
    """
    Plot the confusion matrix

    A near exact replica of fastai's method, with the added option
    of `return_fig`, to be able to save the image to disk and a
    different default colormap
    """
    if self.is_multilabel:
        raise NotImplementedError(
            f"Confusion matrices for multi-label problems aren't implemented"
        )
    # This function is mainly copied from the sklearn docs
    cm = self.confusion_matrix()
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(dpi=dpi, figsize=figsize, **kwargs)
    if cmap is None:
        cmap = Davos_3_r.mpl_colormap
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(self.vocab))
    plt.xticks(tick_marks, self.vocab, rotation=90)
    plt.yticks(tick_marks, self.vocab, rotation=0)

    if plot_txt:
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            coeff = f"{cm[i, j]:.{norm_dec}f}" if normalize else f"{cm[i, j]}"
            plt.text(
                j,
                i,
                coeff,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax = fig.gca()
    ax.set_ylim(len(self.vocab) - 0.5, -0.5)

    plt.tight_layout()
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.grid(False)
    if return_fig:
        return fig


# Cell
@patch
def plot_accuracy(
    self: ClassificationInterpretationEx,
    width=0.9,
    figsize=(6, 6),
    return_fig=False,
    title="Accuracy Per Label",
    ylabel="Accuracy (%)",
    style: Optional[str] = None,
    color="#2a467e",
    vertical_labels=True,
):
    "Plot a bar plot showing accuracy per label"
    if not hasattr(self, "preds_df_each"):
        raise NotImplementedError
    if style:
        plt.style.use(style)
    if not hasattr(self, "preds_df_each"):
        self.compute_label_confidence()
    self.accuracy_dict = defaultdict()

    for label, df in self.preds_df_each.items():
        total = len(df["accurate"]) + len(df["inaccurate"])
        self.accuracy_dict[label] = 100 * len(df["accurate"]) / total

    fig, ax = plt.subplots(figsize=figsize)

    x = self.accuracy_dict.keys()
    y = [v for k, v in self.accuracy_dict.items()]

    rects = ax.bar(x, y, width, color=color)
    for rect in rects:
        ht = rect.get_height()
        ax.annotate(
            s=f"{ht:.02f}",
            xy=(rect.get_x() + rect.get_width() / 2, ht),
            xytext=(0, 3),  # offset vertically by 3 points
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set_ybound(lower=0, upper=100)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(x, rotation="vertical" if vertical_labels else "horizontal")
    plt.suptitle(title)
    plt.tight_layout()
    if return_fig:
        return fig


# Cell
@patch
def plot_label_confidence(
    self: ClassificationInterpretationEx,
    bins: int = 5,
    fig_width: int = 12,
    fig_height_base: int = 4,
    title: str = "Accurate vs. Inaccurate Predictions Confidence (%) Levels Per Label",
    return_fig: bool = False,
    label_bars: bool = True,
    style: Optional[str] = None,
    dpi=150,
    accurate_color="#2a467e",
    inaccurate_color="#dc4a46",
):
    """Plot label confidence histograms for each label
    Key Args:
      * `bins`:       No. of bins on each side of the plot
      * `return_fig`: If True, returns the figure that can be easily saved to disk
      * `label_bars`: If True, displays the % of samples that fall into each bar
      * `style`:      A matplotlib style. See `plt.style.available` for more
      * `accurate_color`:   Color of the accurate bars
      * `inaccurate_color`: Color of the inaccurate bars
    """
    if not hasattr(self, "preds_df_each"):
        raise NotImplementedError
    if style:
        plt.style.use(style)
    fig, axes = plt.subplots(
        nrows=len(self.preds_df_each.keys()),
        ncols=2,
        dpi=dpi,
        figsize=(fig_width, fig_height_base * len(self.dl.vocab)),
    )
    for i, (label, df) in enumerate(self.preds_df_each.items()):
        height = 0
        # find max height
        for mode in ["inaccurate", "accurate"]:
            len_bins, _ = np.histogram(df[mode][label], bins=bins)
            if len_bins.max() > height:
                height = len_bins.max()

        for mode, ax in zip(["inaccurate", "accurate"], axes[i]):
            range_ = (50, 100) if mode == "accurate" else (0, 50)
            color = accurate_color if mode == "accurate" else inaccurate_color
            num, _, patches = ax.hist(
                df[mode][label], bins=bins, range=range_, rwidth=0.95, color=color
            )
            num_samples = len(df["inaccurate"][label]) + len(df["accurate"][label])
            pct_share = len(df[mode][label]) / num_samples
            if label_bars:
                for rect in patches:
                    ht = rect.get_height()
                    ax.annotate(
                        s=f"{round((int(ht) / num_samples) * 100, 1) if ht > 0 else 0}%",
                        xy=(rect.get_x() + rect.get_width() / 2, ht),
                        xytext=(0, 3),  # offset vertically by 3 points
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )
            ax.set_ybound(upper=height + height * 0.3)
            ax.set_xlabel(
                f"{label}: {mode.capitalize()} ({round(pct_share * 100, 2)}%)"
            )
            ax.set_ylabel(
                f"Num. {mode.capitalize()} ({len(df[mode][label])} of {num_samples})"
            )
    fig.suptitle(title, y=1.0)
    plt.subplots_adjust(top=0.9, bottom=0.01, hspace=0.25, wspace=0.2)
    plt.tight_layout()
    if return_fig:
        return fig


# Cell
from ..utils import *


@patch
def plot_top_losses_grid(
    self: ClassificationInterpretationEx,
    k=16,
    ncol=4,
    __largest=True,
    font_path=None,
    font_size=12,
    use_dedicated_layout=True,
) -> PIL.Image.Image:
    """Plot top losses in a grid

    Uses fastai'a `ClassificationInterpretation.plot_top_losses` to fetch
    predictions, and makes a grid with the ground truth labels, predictions,
    prediction confidence and loss ingrained into the image

    By default, `use_dedicated_layout` is used to plot the loss (bottom),
    truths (top-left), and predictions (top-right) in dedicated areas of the
    image. If this is set to `False`, everything is printed at the bottom of the
    image
    """
    # all of the pred fetching code is copied over from
    # fastai's `ClassificationInterpretation.plot_top_losses`
    # and only plotting code is added here
    losses, idx = self.top_losses(k, largest=__largest)
    if not isinstance(self.inputs, tuple):
        self.inputs = (self.inputs,)
    if isinstance(self.inputs[0], Tensor):
        inps = tuple(o[idx] for o in self.inputs)
    else:
        inps = self.dl.create_batch(
            self.dl.before_batch([tuple(o[i] for o in self.inputs) for i in idx])
        )
    b = inps + tuple(
        o[idx] for o in (self.targs if is_listy(self.targs) else (self.targs,))
    )
    x, y, its = self.dl._pre_show_batch(b, max_n=k)
    b_out = inps + tuple(
        o[idx] for o in (self.decoded if is_listy(self.decoded) else (self.decoded,))
    )
    x1, y1, outs = self.dl._pre_show_batch(b_out, max_n=k)
    # if its is not None:
    #    _plot_top_losses(x, y, its, outs.itemgot(slice(len(inps), None)), self.preds[idx], losses,  **kwargs)
    plot_items = (
        its.itemgot(0),
        its.itemgot(1),
        outs.itemgot(slice(len(inps), None)),
        self.preds[idx],
        losses,
    )

    def draw_label(x: TensorImage, labels):
        return PILImage.create(x).draw_labels(
            labels, font_path=font_path, font_size=font_size, location="bottom"
        )

    # return plot_items
    results = []
    for x, truth, preds, preds_raw, loss in zip(*plot_items):
        if self.is_multilabel:
            preds = preds[0]
        probs_i = np.array([self.dl.vocab.o2i[o] for o in preds])
        pred2prob = [
            f"{pred} ({round(prob.item()*100,2)}%)"
            for pred, prob in zip(preds, preds_raw[probs_i])
        ]
        if use_dedicated_layout:
            # draw loss at the bottom, preds on top-right
            # and truths on the top
            img = PILImage.create(x)
            if isinstance(truth, Category):
                truth = [truth]
            truth.insert(0, "TRUTH: ")
            pred2prob.insert(0, "PREDS: ")
            loss_text = f"{'LOSS: '.rjust(8)} {round(loss.item(), 4)}"
            img.draw_labels(
                truth, location="top-left", font_size=font_size, font_path=font_path
            )
            img.draw_labels(
                pred2prob,
                location="top-right",
                font_size=font_size,
                font_path=font_path,
            )
            img.draw_labels(
                loss_text, location="bottom", font_size=font_size, font_path=font_path
            )
            results.append(img)
        else:
            # draw everything at the bottom
            out = []
            out.append(f"{'TRUTH: '.rjust(8)} {truth}")
            bsl = "\n"  # since f-strings can't have backslashes
            out.append(f"{'PRED: '.rjust(8)} {bsl.join(pred2prob)}")
            if self.is_multilabel:
                out.append("\n")
            out.append(f"{'LOSS: '.rjust(8)} {round(loss.item(), 4)}")
            results.append(draw_label(x, out))
    return make_img_grid(results, img_size=None, ncol=ncol)


# Cell
@patch
@delegates(to=ClassificationInterpretationEx.plot_top_losses_grid, but=["largest"])
def plot_lowest_losses_grid(self: ClassificationInterpretationEx, **kwargs):
    """Plot the lowest losses. Exact opposite of `ClassificationInterpretationEx.plot_top_losses`"""
    return self.plot_top_losses_grid(__largest=False, **kwargs)


# Cell
import sklearn.metrics as skm


@patch
def print_classification_report(self: ClassificationInterpretationEx, as_dict=False):
    "Get scikit-learn classification report"
    # `flatten_check` and `skm.classification_report` don't play
    # nice together for multi-label
    # d,t = flatten_check(self.decoded, self.targs)
    d, t = self.decoded, self.targs
    if as_dict:
        return skm.classification_report(
            t,
            d,
            labels=list(self.vocab.o2i.values()),
            target_names=[str(v) for v in self.vocab],
            output_dict=True,
        )
    else:
        return skm.classification_report(
            t,
            d,
            labels=list(self.vocab.o2i.values()),
            target_names=[str(v) for v in self.vocab],
            output_dict=False,
        )
