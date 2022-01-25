## WIP Changes

- **`fastai_amalgam.augment.kornia`**
  - **Changes**
    - Updated for `kornia>=0.6.2`
    - Move away from nbdev based workflow
    - Use `beartype` to validate probabilities are between 0-1
    - Use kornia's `p` parameter (now available on ~all~ most augmentations) instead of handling that internally inside `KorniaBase`
      - Where `kornia` doesn't provide `p`, create a `Random{}` class that behaves like a native kornia transform
    - Don't prefix transforms to be used in fastai data block pipeline with `Random` (all augmentations used need to have a random element anywayrs!). Also leads to lesser chances of namespace collision
    - `KorniaBase` is now more opinionated - it only works with a transform inherited from `nn.Module`
  - **New Features**
    - Added a bunch of new wrappers around:
      - `K.augmentation.RandomChannelShuffle`
      - `K.augmentation.RandomFisheye`
      - `K.augmentation.RandomEqualize`
    - `RandomInvertLazy` - A variant of `K.augmentation.RandomInvert` that computes `max_val` dynamically (helpful if inputs are not normalised, like in YOLOX models)

- Delete `fastai_amalgam.albumentations`. Will only be supporting `kornia` moving forward

- Update `fastai_amalgam.export.onnx` for newer ONNX versions (>= ...?); use `onnxsim` for simplifying / optimising model
