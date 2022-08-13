#!/usr/bin/env python

import numpy as np

__all__ = ['Albumentations']

from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform
import albumentations as A


class AlbumentationsTransform(Transform):
    def __init__(self, augmentor):
        """
        Args:
            augmentor (albumentations.Compose):
        """
        if not isinstance(augmentor, A.Compose):
            raise ValueError("augmentor parameter should be Compose")
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        return self.augmentor(img)

    def apply_coords(self, coords):
        raise NotImplementedError("AlbumentationsTransform should implement `apply_coords`")

    def inverse(self):
        raise NotImplementedError("AlbumentationsTransform should implement `inverse`")

    def apply_segmentation(self, segmentation):
        raise NotImplementedError("AlbumentationsTransform should implement `apply_segmentation`")


class Albumentations(Augmentation):
    def __init__(self, augmentor):
        """
        Args:
            augmentor (albumentations.Compose):
        """
        super(Albumentations, self).__init__()
        self._aug = augmentor

    def get_transform(self, img):
        return AlbumentationsTransform(self._aug)
# Mostly adopted from https://github.com/tensorpack/tensorpack/blob/8e6f930b5f021c21f785619278e7ec9beacc0a1b/tensorpack/dataflow/imgaug/external.py#L66-L97

