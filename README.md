# Lab

## 1. layer_inpainting

层间图像修复

#### inpaint.py

```bash
python inpaint.py --input datasets/imgs --mask datasets/masks --gap 2 --out results
```

注：输入为文件夹路径，`gap`为两行已知像素中的空行数量。
