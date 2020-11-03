# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/augment_albumentations.ipynb (unless otherwise specified).

__all__ = ['albu_augment', 'AlbumentationsWrapper', 'BlurringTfms', 'StyleTfms', 'WeatherTfms', 'NoiseTfms',
           'ColorTonesTfms', 'ColorChannelTfms', 'LightingTfms', 'OtherTfms', 'Tfms']

# Cell
try:
    from fastai.vision.all import *
except:
    from fastai2.vision.all import *
import PIL
from typing import List, Tuple, Callable, Union, Optional, Any

# Cell
import albumentations as A
from albumentations import (
    RandomBrightness, RandomContrast, CLAHE,      # Lighting Transforms
    ToSepia, ToGray,                              # Color Tones
    RGBShift, HueSaturationValue, ChannelShuffle, # Color Channels
    Blur, MotionBlur, GaussianBlur,               # Blurs
    MedianBlur, IAAEmboss,                        # Style Transfer-Esque
    GaussNoise, IAAAdditiveGaussianNoise,         # Noise
    JpegCompression, Posterize,                   # Noise
    RandomSunFlare,                               # Weather
    FancyPCA,                                     # Other
    OneOf, Compose
)

# Cell
def albu_augment(aug, img): return aug(image=np.array(img))['image']

##https://dev.fast.ai/tutorial.pets
class AlbumentationsWrapper(Transform):
    "Wrapper function for any Albumentations Transform"
    order=0 # apply before `ToTensor`
    def __init__(self, aug): self.aug=aug
    def __repr__(self): return self.aug.__repr__()

    def _encodes(self, img:PILImage):
        aug_img = albu_augment(self.aug, img)
        return PILImage.create(aug_img)

    def encodes(self, img:PIL.Image.Image): return self._encodes(img)
    def encodes(self, img:PILImage): return self._encodes(img)
    def encodes(self, o:(str,Path)): return self._encodes(PILImage.create(o))

# Cell

# lean heavily towards Motion Blur
BlurringTfms = OneOf([
    MotionBlur(blur_limit=(15,15), p=0.8),
    Blur(p=0.2)
], p=0.5)

StyleTfms = OneOf([
    MedianBlur(blur_limit=(3,7), p=0.6),
    IAAEmboss(strength=(0.2,0.99), p=0.4)
], p=0.3)

WeatherTfms = RandomSunFlare(
    src_radius=80, p=0.1
)

NoiseTfms = OneOf([
    GaussNoise(p=0.6),
    IAAAdditiveGaussianNoise(p=0.4), # stronger
    JpegCompression(quality_lower=25, quality_upper=55, p=0.2)
], p=0.25)

ColorTonesTfms = OneOf([
    ToSepia(),
    ToGray()
], p=0.3)

ColorChannelTfms = OneOf([
    ChannelShuffle(),
    HueSaturationValue(val_shift_limit=5),
    RGBShift()
], p=0.3)

LightingTfms = OneOf([
    RandomContrast(p=0.1),
    RandomBrightness(p=0.1),
    CLAHE(p=0.8)
], p=0.3)

OtherTfms = FancyPCA(alpha=0.4, p=0.4)

# Cell
Tfms = Compose([
    BlurringTfms,
    StyleTfms,
    WeatherTfms,
    NoiseTfms,
    ColorTonesTfms,
    ColorChannelTfms,
    LightingTfms,
    OtherTfms
])