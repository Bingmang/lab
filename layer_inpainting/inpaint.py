"""
层间图像修复（传统方法）

Usage: python inpaint.py --input datasets/imgs --mask datasets/masks --gap 2 --out results

"""

import os
import argparse
import multiprocessing
import cv2
from skimage.measure import compare_mse

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, type=str, help='The input image directory.')
parser.add_argument('--mask', required=True, type=str, help='The mask image directory.')
parser.add_argument('--out', type=str, default='./out', help='The output directory.')
parser.add_argument('--gap', required=True, type=int, help='The number of blank line between two lines.')

opt = parser.parse_args()
if not opt.out.endswith('/'):
    opt.out += '/'
print(opt)

GAP_RADIUS_MAP = {
    1: 2,
    2: 2,
    3: 1,
    4: 20,
    5: 24,
    6: 26,
}


def mkoutdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def layer_inpainting(img, mask, gap, mode='NS'):
    radius = GAP_RADIUS_MAP[gap]
    if mode == 'NS':
        return cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
    elif mode == 'TELEA':
        return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    else:
        raise TypeError('The inpaint method is not supported')


def get_flist(path):
    flist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('jpeg') or file.endswith('.png'):
                flist.append(os.path.join(root, file))
    return flist


def inpaint_worker(img, mask):
    name, ext = os.path.basename(img).split('.')
    img = cv2.imread(img)
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    dst = layer_inpainting(img, mask, opt.gap)
    cv2.imwrite(opt.out + '{}.{}'.format(name, ext), dst)


if __name__ == '__main__':
    mkoutdir(opt.out)
    img_list = get_flist(opt.input)
    mask_list = get_flist(opt.mask)

    # 多线程
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for i in range(len(img_list)):
        pool.apply_async(inpaint_worker, (img_list[i], mask_list[i]))
    pool.close()
    pool.join()
