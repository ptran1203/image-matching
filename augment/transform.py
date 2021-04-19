import albumentations as A
from augment.gridmask import GridMask


def get_transforms(image_size, stage=1, norm=True):
    if stage == 1:
        transforms_train = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.25,
                scale_limit=0.25,
                rotate_limit=20,
                border_mode=0,
                p=0.3,
            ),
            A.OneOf(
                [
                    A.MedianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                    A.GaussNoise(),
                ],
                p=0.3,
            ),
            A.OneOf([A.GridDistortion(), A.OpticalDistortion(),], p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ]
    elif stage == 2:
        transforms_train = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.25,
                scale_limit=0.25,
                rotate_limit=30,
                border_mode=0,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.MedianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                    A.GaussNoise(),
                ],
                p=0.5,
            ),
            A.OneOf([A.GridDistortion(), A.OpticalDistortion(),], p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            GridMask(num_grid=(5, 7), p=0.5),
        ]
    else:
        transforms_train = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.25,
                scale_limit=0.25,
                rotate_limit=30,
                border_mode=0,
                p=0.7,
            ),
            A.OneOf(
                [
                    A.MedianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                    A.GaussNoise(),
                ],
                p=0.7,
            ),
            A.OneOf([A.GridDistortion(), A.OpticalDistortion(),], p=0.7),
            A.RandomBrightnessContrast(p=0.7),
            GridMask(num_grid=(3, 7), p=0.7),
        ]

    transforms_val = [
        A.Resize(image_size, image_size),
    ]

    if norm:
        transforms_train.append(A.Normalize())
        transforms_val.append(A.Normalize())
    else:
        transforms_train.append(A.Normalize(mean=0, std=1))
        transforms_val.append(A.Normalize(mean=0, std=1))

    return A.Compose(transforms_train), A.Compose(transforms_val)
