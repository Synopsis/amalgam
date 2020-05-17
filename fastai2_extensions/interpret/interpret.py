# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_classification-interpretation.ipynb (unless otherwise specified).

__all__ = ['ClassificationInterpretationEx']

# Cell
from fastai2.vision.all import *
from fastai2.metrics import *
from nbdev.showdoc import show_doc

# Cell
class ClassificationInterpretationEx(ClassificationInterpretation):
    """
    Extend fastai2's `ClassificationInterpretation` to analyse model predictions in more depth
    """
    def __init__(self, dl, inputs, preds, targs, decoded, losses):
        store_attr(self, "dl,inputs,preds,targs,decoded,losses")

    def compute_label_confidence(self):
        """
        Collate prediction confidence, filenames, and ground truth labels
        in DataFrames, and store them as class attributes
        `self.preds_df` and `self.preds_df_each`
        """
        self._preds_collated = [
            (item, self.dl.vocab[label_idx], *preds.numpy()*100)\
            for item,label_idx,preds in zip(self.dl.items,
                                            self.targs,
                                            self.preds)
        ]

        self.preds_df       = pd.DataFrame(self._preds_collated, columns = ['fname','truth', *self.dl.vocab])
        self._preds_df_each = {l:self.preds_df.copy()[self.preds_df.truth == l].reset_index(drop=True) for l in self.dl.vocab}
        self.preds_df_each  = defaultdict(dict)


        sort_desc = lambda x,col: x.sort_values(col, ascending=False).reset_index(drop=True)
        for label,df in self._preds_df_each.items():
            filt = df[label] == df[self.dl.vocab].max(axis=1)
            self.preds_df_each[label]['accurate']   = df.copy()[filt]
            self.preds_df_each[label]['inaccurate'] = df.copy()[~filt]

            self.preds_df_each[label]['accurate']   = sort_desc(self.preds_df_each[label]['accurate'], label)
            self.preds_df_each[label]['inaccurate'] = sort_desc(self.preds_df_each[label]['inaccurate'], label)
            assert len(self.preds_df_each[label]['accurate']) + len(self.preds_df_each[label]['inaccurate']) == len(df)

    def plot_label_confidence(self, bins=10, fig_width=12, fig_height_base=4, return_fig=False,
                              title='Accurate vs. Inaccurate Predictions Confidence (%) Levels Per Label',
                              accurate_color='mediumseagreen', inaccurate_color='tomato'):
        'Plot label confidence histograms for each label'
        if not hasattr(self, 'preds_df_each'): self.compute_label_confidence()
        fig, axes = plt.subplots(nrows = len(self.preds_df_each.keys()), ncols=2,
                                 figsize = (fig_width, fig_height_base * len(self.dl.vocab)))
        for i, (label, df) in enumerate(self.preds_df_each.items()):
            for mode,ax in zip(['inaccurate', 'accurate'], axes[i]):
                range_ = (50,100) if mode == 'accurate' else (0,50)
                color  = accurate_color if mode == 'accurate' else inaccurate_color
                ax.hist(df[mode][label], bins=bins, range=range_, rwidth=.95, color=color)
                ax.set_xlabel(f'{label}: {mode.capitalize()}')
                ax.set_ylabel(f'No. {mode.capitalize()} = {len(df[mode][label])}')
        fig.suptitle(title)
        plt.subplots_adjust(top = 0.9, bottom=0.01, hspace=0.25, wspace=0.2)
        if return_fig: return fig

    def get_fnames(self, label:str,
                   mode:('accurate','inaccurate'),
                   conf_level:Union[int,float,tuple]) -> np.ndarray:
        """
        Utility function to grab filenames of a particular label `label` that were classified
        as per `mode` (accurate|inaccurate).
        These filenames are filtered by `conf_level` which can be above or below a certain
        threshold (above if `mode` == 'accurate' else below), or in confidence ranges
        """
        assert label in self.dl.vocab
        if not hasattr(self, 'preds_df_each'): self.compute_label_confidence()
        df = self.preds_df_each[label][mode].copy()
        if mode == 'accurate':
            if isinstance(conf_level, tuple):       filt = df[label].between(*conf_level)
            if isinstance(conf_level, (int,float)): filt = df[label] > conf_level
        if mode == 'inaccurate':
            if isinstance(conf_level, tuple):       filt = df[label].between(*conf_level)
            if isinstance(conf_level, (int,float)): filt = df[label] < conf_level
        return df[filt].fname.values