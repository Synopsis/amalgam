# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/interpret_gradcam.ipynb (unless otherwise specified).

__all__ = ['Hook', 'HookBwd', 'create_test_img', 'compute_gcam_items', 'compute_gcam_map', 'plt_decoded', 'plot_gcam',
           'GradCam', 'PathLike']

# Cell
try:
    from fastai.vision.all import *
except:
    from fastai2.vision.all import *
from typing import List, Tuple, Callable, Union, Optional, Any

# Cell
class Hook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)
    def hook_func(self, m, inp, out):  self.stored = out.detach().clone()
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.hook.remove()

class HookBwd():
    def __init__(self,m):
        self.hook = m.register_backward_hook(self.hook_func)
    def hook_func(self, model, grad_in, grad_out): self.stored = grad_out[0].detach().clone()
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.hook.remove()

# Cell
def create_test_img(learn, f, return_img=True):
    img = PILImage.create(f)
    x = first(learn.dls.test_dl([f]))
    x = x[0]
    if return_img: return img,x
    return x

# Cell
def compute_gcam_items(learn, x, label) -> Tuple[torch.Tensor]:
    'Compute gradient and activations of `model` for `x` with respect to `label`'

    label_idx = learn.dls.vocab.o2i[label]

    with HookBwd(learn.model[0]) as hook_g:
        with Hook(learn.model[0]) as hook:
            preds       = learn.model.eval()(x)
            activations = hook.stored
        preds[0, label_idx].backward()
        gradients = hook_g.stored

    preds = getattr(learn.loss_func, 'activation', noop)(preds)

    # remove leading batch_size axis
    gradients   = gradients  [0]
    activations = activations[0]

    return gradients,activations,preds.detach().cpu().numpy().flatten()

# Cell
def compute_gcam_map(gradients, activations):
    '(mean(gradients) * activations).sum() to return tiny grad-cam map'
    # Mean over the feature maps. If you don't use `keepdim`, it returns
    # a value of shape (1280) which isn't amenable to `*` with the activations
    gcam_weights = gradients.mean(dim=[1,2], keepdim=True) # (1280,7,7)   --> (1280,1,1)
    gcam_map     = (gcam_weights * activations) # (1280,1,1) * (1280,7,7) --> (1280,7,7)
    gcam_map     = gcam_map.sum(0)              # (1280,7,7) --> (7,7)
    return gcam_map

# Cell
def plt_decoded(learn, x, ctx):
    'Processed tensor --> plottable image, return `extent`'
    x_decoded = TensorImage(learn.dls.train.decode((x,))[0][0])
    extent = (0, x_decoded.shape[1], x_decoded.shape[2], 0)
    x_decoded.show(ctx=ctx)
    return extent

def plot_gcam(img:PILImage, x:tensor, gcam_map:tensor, plt_axis,
              full_size=True, alpha=0.6, learn=None,
              interpolation='bilinear', cmap='magma'):
    'Plot gradcam on `plt_axis`'
    if full_size:
        extent = (0, img.width,img.height, 0)
        show_image(img, ctx=plt_axis)
    else:
        extent = plt_decoded(learn, x, plt_axis)

    show_image(gcam_map.detach().cpu(), ctx=plt_axis,
               alpha=alpha, extent=extent,
               interpolation=interpolation, cmap=cmap)

# Cell
import math
from typing import List
PathLike = Union[str,Path]

class GradCam():
    "Class interface to facilitate computing and siplaying Grad-CAM"
    def __init__(self, learn:Learner, fname:PathLike, labels:Union[str,List[str],None]):
        """
        Compute Grad-CAM maps for all `labels`.
        If `labels` is None, compute for the predicted class (more expensive)
        """
        self.learn = learn
        self.fname = fname
        self.img, self.x = create_test_img(self.learn, self.fname)
        if labels is None:
            self.labels = [self.learn.predict(fname)[0]]
        else:
            self.labels = [labels] if isinstance(labels,str) else labels
        self.compute_gcams()

    def compute_gcams(self):
        self.gradcams = defaultdict()
        for label in self.labels:
            gradients, activations, self.preds = compute_gcam_items(self.learn,self.x,label)
            gcam_map = compute_gcam_map(gradients, activations)
            self.gradcams[label] = gcam_map
            self.preds_dict = {
                lab:pred for pred,lab in zip(self.preds, self.learn.dls.vocab)
            }

    def plot(self, max_ncols=None, full_size=True, alpha=0.6,
             interpolation='bilinear', cmap='magma',
             figsize=(12,12), return_fig=False, plot_original=False):
        """
        Plot the computed Grad-CAMs.

        Key Arguments
        -------------
        * full_size: If True, plots the images in their original size, else
                     in the size that the `Learner` resizes them to
        * plot_original: if True, plots the original image without any overlays
                         in addition to the heatmaps
        * max_ncols: Use this to manipulate the number of rows you'd like your
                     plot to have. Useful for classifiers with a large no. of
                     outputs. Enter `None` to plot everything in one row.
        """
        label_idx = 0
        total = len(self.labels)+1 if plot_original else len(self.labels)

        if max_ncols is None:
            max_ncols=len(self.labels)+1 if plot_original else len(self.labels)

        if total > max_ncols:
            nrows  = math.ceil(total/max_ncols)
            fig,ax = plt.subplots(nrows=nrows, ncols=max_ncols, figsize=figsize)
            plt.axis('off')

            for i in range(nrows):
                for j in range(max_ncols):
                    if plot_original:
                        if i==0 and j==0:
                            if full_size:
                                show_image(self.img, ctx=ax[0,0])
                            else:
                                x = TensorImage(learn.dls.train.decode((self.x,))[0][0])
                                x.show(ctx=ax[0,0])
                            ax[0][0].set_title('original')
                            continue
                    plot_gcam(img=self.img, x=self.x, full_size=full_size,
                              gcam_map=self.gradcams[self.labels[label_idx]],
                              plt_axis=ax[i,j], alpha=alpha, learn=self.learn,
                              interpolation=interpolation, cmap=cmap)
                    title = self.labels[label_idx]
                    ax[i][j].set_title(f'{title}, {self.preds_dict[title] * 100:.02f}%')
                    label_idx += 1
                    if label_idx >= len(self.labels): break
        else:
            fig,ax = plt.subplots(nrows=1, ncols=max_ncols, figsize=figsize)
            plt.axis('off')

            for i in range(len(ax)):
                if plot_original:
                    if i==0 and full_size:
                        show_image(self.img, ctx=ax[0])
                        ax[0].set_title('original')
                        continue
                    elif i==0 and not full_size:
                        _ = plt_decoded(self.learn, self.x, ctx=ax[0])
                        ax[0].set_title('original')
                        continue
                plot_gcam(img=self.img, x=self.x, full_size=full_size,
                          gcam_map=self.gradcams[self.labels[label_idx]],
                          plt_axis=ax[i], alpha=alpha, learn=self.learn,
                          interpolation=interpolation, cmap=cmap)
                title = self.labels[label_idx]
                ax[i].set_title(f'{title}, {self.preds_dict[title] * 100:.02f}%')
                label_idx += 1