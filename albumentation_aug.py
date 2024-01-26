import albumentations as A
import numpy as np
from PIL import Image


class AlbumentationsAugmentation(object):
    """exapmple:
    dict(
        type='AlbumentationsAugmentation',
        transforms=[
            dict(type='RandomCrop', width=256, height=256),
            dict(type='HorizontalFlip', p=0.5),
            dict(type='RandomBrightnessContrast', p=0.2),
        ])
    """
    def __init__(self, transforms):
        transforms_list = [
            getattr(
                A,
                item['type'])(**{k: v
                                 for k, v in item.items() if k != 'type'})
            for item in transforms
        ]
        self.transform = A.Compose(transforms_list)

    def __call__(self, img, seg=None):
        if seg is None:
            augmented = self.transform(image=img)
            img = augmented['image']
            return img
        augmented = self.transform(image=img, mask=seg)
        img = augmented['image']
        seg = augmented['mask']

        return img, seg

    def __repr__(self):
        return self.__class__.__name__


if __name__ == '__main__':
    palette = [[0, 0, 0], [255, 255, 255]]
    transformers = [
        dict(type='RandomCrop', width=256, height=256),
        dict(type='HorizontalFlip', p=0.5),
        dict(type='RandomBrightnessContrast', p=0.2),
    ]
    transformers = [
        dict(type='RandomCrop', width=256, height=256),
        dict(
            type='ShiftScaleRotate',
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=30,
            p=0.5,
        ),
        dict(type='RGBShift',
             r_shift_limit=25,
             g_shift_limit=25,
             b_shift_limit=25,
             p=0.5),
        dict(
            type='RandomBrightnessContrast',
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        dict(type='CLAHE', clip_limit=2),
        dict(type='Normalize',
             mean=(0.485, 0.456, 0.406),
             std=(0.229, 0.224, 0.225)),
    ]
    aug = AlbumentationsAugmentation(transformers)
    img_path = '/nas/tsgil/dataset/SpaceNet3/mmstyle/img_dir/test/AOI_2_Vegas_22.png'  # noqa
    aug_path = '/nas/tsgil/gil/test.png'
    seg_path = '/nas/tsgil/dataset/SpaceNet3/mmstyle/ann_dir/test/AOI_2_Vegas_22.png'  # noqa
    new_seg_path = '/nas/tsgil/gil/test_seg.png'
    img = Image.open(img_path)
    img = np.array(img)
    seg = Image.open(seg_path)
    seg = np.array(seg)

    # 이미지 어레이가 들어오면 어그멘티드된 이미지 어레이 리턴
    # new_img = aug(img)
    # if new_img.dtype != 'uint8':
    #     # Assuming new_img is your input array
    #     min_val = new_img.min()
    #     max_val = new_img.max()

    #     # Normalize to 0-1
    #     normalized_img = (new_img - min_val) / (max_val - min_val)
    #     new_img = (normalized_img * 255).astype('uint8')
    # Image.fromarray(new_img).save(aug_path)

    # 이미지와 세그 어레이가 들어오면 어그멘티드된 이미지와 세그 어레이 리턴
    new_img, new_seg = aug(img, seg)

    if new_img.dtype != 'uint8':
        # Assuming new_img is your input array
        min_val = new_img.min()
        max_val = new_img.max()

        # Normalize to 0-1
        normalized_img = (new_img - min_val) / (max_val - min_val)
        new_img = (normalized_img * 255).astype('uint8')
    Image.fromarray(new_img).save(aug_path)

    # new_seg가 0과 1로 이루어져 있어서 팔레트에서 찾아서 색을 보여줌
    pil_image = Image.fromarray(new_seg).convert('P')
    pil_image.putpalette(np.array(palette, dtype=np.uint8))
    pil_image.save(new_seg_path)
