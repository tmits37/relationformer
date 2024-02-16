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
            getattr(A,item['type'])(**{k: v for k, v in item.items() if k != 'type'})
            for item in transforms
        ]
        self.transform = A.Compose(transforms_list)

    def __call__(self, img):
        # gt_seg = results['gt_semantic_seg'] # seg용

        # 리사이즈 기능도 쓸거기때문에 seg도 인자로 받을 수 있게
        # transform 짜기 1개 받거나 두개 받거나 가능하게
        # augmented = self.transform(image=img, mask=gt_seg) # seg용
        augmented = self.transform(image=img)
        img = augmented['image']
        # results['gt_semantic_seg'] = augmented['mask'] # seg용

        return img

    def __repr__(self):
        return self.__class__.__name__

if __name__ == '__main__':
    transformers = [
        dict(type='RandomCrop', width=256, height=256),
        dict(type='HorizontalFlip', p=0.5),
        dict(type='RandomBrightnessContrast', p=0.2),
    ]
    aug = AlbumentationsAugmentation(transformers)
    img_path = '/nas/tsgil/dataset/SpaceNet3/mmstyle/img_dir/test/AOI_2_Vegas_22.png'
    aug_path = '/nas/tsgil/gil/test.png'
    img = Image.open(img_path)
    img = np.array(img)
    new_img = aug(img) # 이미지 어레이가 들어오면 어그멘티드된 이미지 어레이 리턴
    print(new_img)
    Image.fromarray(new_img).save(aug_path)
    

