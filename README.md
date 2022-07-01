# DETR3D

This repo is to build a lightweight DETR3D without relying on MMDET3D. 

## Prepare Dataset

Follow [MMDET3D](https://github.com/open-mmlab/mmdetection3d/blob/32a4328b16b85aae26d08d81157ab74b58edcdb1/docs/en/data_preparation.md) to prepare dataset.

Current supported datasets: nuScenes

## Download Pretrained Model

Downloads the [pretrained backbone weights fcos3d](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN) to `ckpts/`.

## Train

### With mmdet3d

```bash
python tools/train.py configs/detr3d/detr3d_res101_gridmask.py
```

### Without mmdet3d

```bash
python tools/train_wo_mmdet3d.py configs/detr3d/detr3d_res101_gridmask.py
```
