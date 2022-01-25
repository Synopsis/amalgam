__all__ = ["torch_to_onnx"]


import torch.nn as nn
import torch
from typing import Union
from pathlib import Path

import os
import onnx
import onnx.utils
from typing import *

# Source: https://github.com/microsoft/onnxruntime/blob/master/tools/python/remove_initializer_from_input.py
def remove_initializer_from_input(model_path):
    model = onnx.load(model_path)
    if model.ir_version < 4:
        print(
            "Model with ir_version below 4 requires to include initilizer in graph input"
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, model_path)


def torch_to_onnx(
    model: nn.Module,
    activation: nn.Module = None,
    save_path: str = "../exported-models/",
    model_fname: str = "onnx-model",
    input_shape: tuple = (1, 3, 224, 224),
    input_name: str = "input_image",
    output_names: Union[str, list] = "output",
    verbose: bool = True,
    opset_version: Optional[int] = None,
    simplify: bool = True,
    **export_args,
) -> os.PathLike:
    """
    Export a `nn.Module` -> ONNX
    This function exports the model with support for batching,
    checks that the export was done properly, and polishes the
    model up (removes unnecessary fluff added during conversion)

    The path to the saved model is returned
    Key Arguments
    =============
    * activation:  If not None, append this to the end of your model.
                   Typically a `nn.Softmax(-1)` or `nn.Sigmoid()`
    * input_shape: Shape of the inputs to the model
    """
    save_path = Path(save_path)
    if isinstance(output_names, str):
        output_names = [output_names]
    if activation:
        model = nn.Sequential(*[model, activation])
    model.eval()

    # Dummy inputs, dry run
    x = torch.randn(input_shape, requires_grad=True)
    x = x.cuda() if torch.cuda.is_available() else x
    model(x)

    # Dynamic axes
    dynamic_batch = {0: "batch"}
    dynamic_axes = {input_name: dynamic_batch}
    for out in output_names:
        dynamic_axes[out] = dynamic_batch

    torch.onnx._export(
        model,
        x,
        f"{save_path/model_fname}.onnx",
        export_params=True,
        verbose=False,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=True,
        opset_version=opset_version,
        **export_args,
    )
    if verbose:
        print(
            f"Loading, polishing, and optimising exported model from {save_path/model_fname}.onnx"
        )
    onnx_model = onnx.load(f"{save_path/model_fname}.onnx")

    if simplify:
        from onnxsim import simplify

        onnx_model, check = simplify(
            onnx_model,
            input_shapes={input_name: input_shape},
            dynamic_input_shape=True,
        )

    onnx.save(onnx_model, f"{save_path/model_fname}.onnx")
    print("<Exported ONNX model successfully>")  # print regardless
    return f"{save_path/model_fname}.onnx"
