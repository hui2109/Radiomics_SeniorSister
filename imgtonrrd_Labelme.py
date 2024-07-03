import pathlib
import pickle

import SimpleITK as sitk
import numpy as np
from PIL import Image


def image_to_nrrd(imgs_path: str, masks_path: str, cls: str):
    finding_list = []
    images_path = pathlib.Path(imgs_path)
    masks_path = pathlib.Path(masks_path)

    finding_list_path = pathlib.Path(f'./resources/extract_data')
    converted_imgs_path = pathlib.Path(f'./resources/converted/{cls}/imgs')
    converted_masks_path = pathlib.Path(f'./resources/converted/{cls}/masks')
    finding_list_path.mkdir(parents=True, exist_ok=True)
    converted_imgs_path.mkdir(parents=True, exist_ok=True)
    converted_masks_path.mkdir(parents=True, exist_ok=True)

    for i in images_path.iterdir():
        img = Image.open(i)
        grey_img = img.convert('L')
        grey_img_array = np.array(grey_img)
        grey_img_3d_array = np.array([grey_img_array])
        img = sitk.GetImageFromArray(grey_img_3d_array)

        label_path = masks_path / (i.stem + '.png')
        label = Image.open(label_path)
        grey_label = label.convert('L')
        grey_label_array = np.array(grey_label)
        grey_label_3d_array = np.array([grey_label_array])

        # 判断病灶类型，38是阴性，设为0
        lesion_set = set(grey_label_3d_array[grey_label_3d_array != 0])
        if 38 in lesion_set:
            lesion_type = 0
        else:
            lesion_type = 1

        grey_label_3d_array[grey_label_3d_array != 0] = 1
        label = sitk.GetImageFromArray(grey_label_3d_array)

        # 将数组写入文件
        sitk.WriteImage(img, converted_imgs_path / (i.stem + '.nrrd'))
        sitk.WriteImage(label, converted_masks_path / (i.stem + '.nrrd'))

        # 制作成组表,方便查找
        group_dict = {
            'img': converted_imgs_path / (i.stem + '.nrrd'),
            'mask': converted_masks_path / (i.stem + '.nrrd'),
            'label': lesion_type,
            'stem': i.stem
        }
        finding_list.append(group_dict)

    with open(finding_list_path / f'finding_list_{cls}.pkl', 'wb') as f:
        pickle.dump(finding_list, f)


if __name__ == '__main__':
    image_to_nrrd('./resources/source_data/ceus_new/dataset_voc_ceus/JPEGImages',
                  './resources/source_data/ceus_new/dataset_voc_ceus/SegmentationClassPNG', 'ceus_new')
