# BOP DATASET: MVTec ITODD [1]


## Dataset parameters

* Objects: 28
* Object models: Mesh models
* Training images: None (must be rendered from the CAD models)
* Test images: XXXXX
* Distribution of the ground truth poses in test images:
    * Range of object distances: 601 - 1102 mm
    * Full rotational range (all rotations are possible)


## Training images

The dataset does not contain dedicated training images, and none of its test
images should be used for training.

The dataset models industrial applications, where it is often not applicable to
acquire and label many training images. Instead, methods must be trained based
on the CAD model and the pose range alone, for example, by rendering the
model.


## Dataset format

General information about the dataset format can be found in:
https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md


## References

[1] Drost, Bertram, et al. "Introducing MVTec ITODD-a dataset for 3D object
    recognition in industry." ICCV 2017,
    web: https://www.mvtec.com/company/research/datasets/mvtec-itodd/
