__all__ = [
    "KorniaBase",
    "MotionBlur",
    "ColorJitter",
    "Rotate",
    "MedianBlur",
    "HFlip",
    "VFlip",
    "Grayscale",
    "PerspectiveWarp",
    "Invert",
    "Equalize",
    "Fisheye",
    "Sharpness",
    "ChannelShuffle",
    # Redefined / Modified Kornia transforms
    "RandomMedianBlur",
    "RandomInvertLazy",
]


from typing import *

from beartype import beartype
from beartype.vale import Is
from fastai.vision.all import *
from PIL import Image
from typing_extensions import Annotated

import kornia as K
from kornia.constants import BorderType, Resample
from torch import Tensor

"""
We only check that the `p` being inputted is between 0-1. Kornia handles checking
valid values for all other parameters
"""


def _unit_interval(x):
    return 0.0 <= x <= 1


FloatProbability = Annotated[float, Is[_unit_interval]]


class KorniaBase(RandTransform):
    """
    Pass in a kornia function, module, list of modules, or nn.Sequential
    containers to `kornia_tfm`.
    If passing functions, you can pass in function arguments as keyword
    args (**kwargs), which can also be random number generators.

    Example
    =======
    * KorniaBase(kornia.adjust_hue, hue_factor=1.2)
    * KorniaBase(kornia.augmentation.ColorJitter(.2,.3,.1,.2))
    * KorniaBase(nn.Sequential(*[kornia.augmentation.ColorJitter()]))
    * KorniaBase([
        kornia.augmentation.ColorJitter(.2),
        kornia.augmentation.RandomMotionBlur(3, 5., 1.)
    ]))
    """

    order = 10
    split_idx = 0

    @beartype
    def __init__(self, kornia_tfm: nn.Module, **kwargs):
        super().__init__(p=1.0)  # Delegate handling of transforms to individual tfms
        self.tfm = kornia_tfm
        # assert isinstance(self.tfm, nn.Module)

    @property
    def to_tensor(self):
        return Pipeline([ToTensor(), IntToFloatTensor()])

    # fmt: off
    def _encode(self, o:TensorImage):  return TensorImage(self.tfm(o)) if self.do else o
    def encodes(self, o:Tensor): return self._encode(o)
    def encodes(self, o:Image.Image):  return self._encode(self.to_tensor(PILImage(o)))
    def encodes(self, o:TensorImage):  return self._encode(o)
    def encodes(self, o:PILImage):     return self._encode(self.to_tensor(o))
    def encodes(self, o:(str,Path)):   return self._encode(self.to_tensor(PILImage.create(o)))
    def encodes(self, o:(TensorCategory,TensorMultiCategory)): return o
    # fmt: on

    def __repr__(self):
        return f"KorniaBase({self.tfm.__repr__()})"


class MotionBlur(KorniaBase):
    "kornia.augmentation.RandomMotionBlur"
    order = 110

    # FIXME: Also accept Tuple[int, int] after this issue is closed: https://github.com/kornia/kornia/issues/1540
    @beartype
    def __init__(
        self,
        p: FloatProbability = 0.2,
        kernel_size: int = 7,
        angle=(15.0, 15.0),
        direction=(-1.0, 1.0),
    ):
        tfm = K.augmentation.RandomMotionBlur(
            kernel_size=kernel_size, angle=angle, direction=direction, p=p
        )
        super().__init__(tfm)


class ColorJitter(KorniaBase):
    "kornia.augmentation.ColorJitter"
    order = 20

    @beartype
    def __init__(
        self,
        p: FloatProbability = 0.2,
        jitter_brightness=0.1,
        jitter_contrast=0.1,
        jitter_saturation=(0.1, 0.9),
        jitter_hue=0.2,
    ):
        tfm = K.augmentation.ColorJitter(
            brightness=jitter_brightness,
            contrast=jitter_contrast,
            saturation=jitter_saturation,
            hue=jitter_hue,
            p=p,
        )
        super().__init__(tfm)


class Rotate(KorniaBase):
    "kornia.augmentation.RandomRotation"
    order = 13

    @beartype
    def __init__(self, p: FloatProbability = 0.2, rotate_degrees=10):
        tfm = K.augmentation.RandomRotation(rotate_degrees, p=p)
        super().__init__(tfm)


class RandomMedianBlur(K.filters.MedianBlur):
    def __init__(self, kernel_size: Tuple[int, int], p: float = 0.5) -> None:
        super().__init__(kernel_size)
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        if random.random() < self.p:
            return super().forward(input)
        else:
            return input


class MedianBlur(KorniaBase):
    "kornia.filters.MedianBlur"
    order = 14

    @beartype
    def __init__(self, p: FloatProbability = 0.2, kernel_size=(5, 5)):
        tfm = RandomMedianBlur(kernel_size, p)
        super().__init__(tfm)


class HFlip(KorniaBase):
    "kornia.augmentation.RandomHorizontalFlip"
    order = 15

    @beartype
    def __init__(self, p: FloatProbability = 0.5):
        tfm = K.augmentation.RandomHorizontalFlip(p=p)
        super().__init__(tfm)


class VFlip(KorniaBase):
    "kornia.augmentation.RandomVerticalFlip"
    order = 16

    @beartype
    def __init__(self, p: FloatProbability = 0.5):
        tfm = K.augmentation.RandomVerticalFlip(p=p)
        super().__init__(tfm)


class Grayscale(KorniaBase):
    "kornia.augmentation.RandomGrayscale"
    order = 17

    @beartype
    def __init__(self, p: FloatProbability = 0.2):
        tfm = K.augmentation.RandomGrayscale(p=p)
        super().__init__(tfm)


class PerspectiveWarp(KorniaBase):
    "kornia.augmentation.RandomPerspective"
    order = 18

    @beartype
    def __init__(
        self,
        p: FloatProbability = 0.2,
        distortion_scale=0.5,
        resample=Resample.BILINEAR,
    ):
        tfm = K.augmentation.RandomPerspective(
            p=p, distortion_scale=distortion_scale, resample=resample
        )
        super().__init__(tfm)


# Redefine
class RandomInvertLazy(K.augmentation.RandomInvert):
    """
    Like kornia's RandomInvert but dynamically determines the `max_val` based
    on the input tensor. This is useful because some models, like YOLOX, may
    not normalise inputs and instead rely on BatchNorm to adapt to the wider
    input range instead
    """

    def __init__(
        self,
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        max_val = None
        super().__init__(max_val, return_transform, same_on_batch, p, keepdim)
        self.flags = dict(max_val="<dynamic>")

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        self.flags = dict(max_val=input.max())
        return super().apply_transform(input, params, transform)


class Invert(KorniaBase):
    "Lazy version of K.augmentation.RandomInvert"
    order = 19

    @beartype
    def __init__(self, p: FloatProbability = 0.25, max_val: float = 1.0):
        tfm = RandomInvertLazy(p=p)
        super().__init__(tfm)


class Equalize(KorniaBase):
    "K.augmentation.RandomEqualize"
    order = 20

    @beartype
    def __init__(self, p: FloatProbability = 0.25):
        tfm = K.augmentation.RandomEqualize(p=p)
        super().__init__(tfm)


class Fisheye(KorniaBase):
    "K.augmentation.RandomFisheye"
    order = 21

    @beartype
    def __init__(
        self,
        p: FloatProbability = 0.25,
        center_x: Tensor = torch.tensor([-0.3, 0.3]),
        center_y: Tensor = torch.tensor([-0.3, 0.3]),
        gamma: Tensor = torch.tensor([0.9, 1.0]),
    ):
        tfm = K.augmentation.RandomFisheye(center_x, center_y, gamma, p=p)
        super().__init__(tfm)


class Sharpness(KorniaBase):
    "K.augmentation.RandomSharpness"
    order = 22

    def __init__(self, p: FloatProbability, sharpness=(3.0, 33.0)):
        tfm = K.augmentation.RandomSharpness(sharpness, p=p)
        super().__init__(tfm)


class ChannelShuffle(KorniaBase):
    "K.augmentation.RandomChannelShuffle"
    order = 23

    @beartype
    def __init__(self, p: FloatProbability):
        tfm = K.augmentation.RandomChannelShuffle(p=p)
        super().__init__(tfm)
