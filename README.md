# 使用yolo-v7训练自己的数据集

## 数据集

- 遥感图像-目标检测数据集[LOD](https://universe.roboflow.com/satellite-images-i8zj5/landscape-object-detection-on-satellite-images-with-ai/dataset/3)
    - coco json格式 [label x y width height]

## train/predict

- train

```shell
python train.py
```

- predict

```shell
python predict.py
```

# 友情链接

精简后修改自 https://github.com/bubbliiiing/yolov7-pytorch