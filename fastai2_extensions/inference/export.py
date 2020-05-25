# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_inference_export-to-other-frameworks.ipynb (unless otherwise specified).

__all__ = ['to_cuda', 'preprocess_one', 'preprocess_batch', 'PathLike', 'open_image', 'torch_to_onnx', 'torch_to_numpy',
           'onnx_to_tf', 'get_nn_spec']

# Cell
import torchvision.transforms.functional as TTF
import torch.nn as nn
to_cuda = lambda x: x.cuda() if torch.cuda.is_available() else x

# Cell
PathLike = Union[str,Path]

def preprocess_one(fname:PathLike):
    x = open_image(fname)
    x = TTF.to_tensor(x)
    x = TTF.normalize(x, mean=[0.485, 0.456, 0.406], # imagenet mean
                         std =[0.229, 0.224, 0.225]) # imagenet sdev
    x = to_cuda(x)
    x = x.unsqueeze(0)
    return x

def preprocess_batch(fnames:Union[PathLike,Collection]):
    batch = [preprocess_one(f) for f in fnames]
    return torch.cat(batch)

# Cell
from pprint import pprint
import PIL

open_image = lambda f,size=(224,224): PIL.Image.open(f).convert('RGB').resize(size, PIL.Image.BILINEAR)

# Cell
import onnx
import onnx.utils
from onnx import optimizer

# Cell
def torch_to_onnx(model:nn.Module,
                  activation:nn.Module=None,
                  save_path:str     = '../exported-models/',
                  model_fname:str   = 'onnx-model',
                  input_shape:tuple = (1,3,224,224),
                  input_name:str    = 'input_image',
                  output_names:Union[str,list] = 'output',
                  **export_args) -> None:
    """
    Export a `nn.Module` -> ONNX

    This function exports the model with support for batching,
    checks that the export was done properly, and polishes the
    model up (removes unnecessary fluff added during conversion)

    Key Arguments
    =============
    * activation:  If not None, append this to the end of your model.
                   Typically a `nn.Softmax(-1)` or `nn.Sigmoid()`
    * input_shape: Shape of the inputs to the model
    """
    save_path = Path(save_path)
    if isinstance(output_names, str): output_names = [output_names]
    if activation: model = nn.Sequential(*[model, activation])
    model.eval()
    x = torch.randn(input_shape, requires_grad=True)
    x = x.cuda() if torch.cuda.is_available() else x
    model(x)
    dynamic_batch = {0: 'batch'}
    dynamic_axes  = {input_name : dynamic_batch}
    for out in output_names: dynamic_axes[out] = dynamic_batch
    torch.onnx._export(model, x, f"{save_path/model_fname}.onnx",
                       export_params=True, verbose=False,
                       input_names=[input_name], output_names=output_names,
                       dynamic_axes=dynamic_axes, keep_initializers_as_inputs=True,
                       **export_args)
    print(f"Loading, polishing, and optimising exported model from {save_path/model_fname}.onnx")
    onnx_model = onnx.load(f'{save_path/model_fname}.onnx')
    model = onnx.utils.polish_model(onnx_model)
    #onnx.checker.check_model(model)

    # removing unused parts of the model
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(onnx_model, passes)

    onnx.save(optimized_model, f'{save_path/model_fname}.onnx')
    print('Exported successfully')

# Cell
def torch_to_numpy(x:tensor):
    if x.requires_grad: return x.detach().cpu().numpy()
    else:               return x.cpu().numpy()

# Cell
def onnx_to_tf(onnx_model:PathLike, output_path:PathLike):
    """
    Simplest wrapper around https://github.com/onnx/onnx-tensorflow/blob/master/example/onnx_to_tf.py
    Ensure `output_path` includes `.pb` as the file extension
    """
    onnx_model = onnx.load(onnx_model_path)
    tf_rep     = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(output_path)  # export the model
    return tf_rep

# Cell
import copy
import coremltools
import onnx_coreml
import os

from onnx_coreml import convert

# Cell
# borrowed from the CoreML Survival Guide, written by Matthijs Hollemans
def get_nn_spec(spec):
    "spec is of type `Model_pb2.Model`, accessed via coreml_model.get_spec()"
    if   spec.WhichOneof("Type") == 'neuralNetwork': return spec.neuralNetwork
    elif spec.WhichOneof("Type") == 'neuralNetworkClassifier': return spec.neuralNetworkClassifier
    elif spec.WhichOneof("Type") == 'neuralNetworkRegressor':  return spec.neuralNetworkRegressor
    return None