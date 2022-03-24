# Applying SimViT to Object Detection

Our detection code is developed on top of [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0) and [PVT](https://github.com/whai362/PVT).

For details see [SimViT: Exploring a Simple Vision Transformer with sliding windows](https://arxiv.org/pdf/2112.13085.pdf). 

If you use this code for a paper please cite:

SimViT
```
@misc{li2021simvit,
      title={SimViT: Exploring a Simple Vision Transformer with sliding windows}, 
      author={Gang Li and Di Xu and Xing Cheng and Lingyu Si and Changwen Zheng},
      year={2021},
      eprint={2112.13085},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Usage

Install [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).

or

```
pip install mmdet==2.13.0 --user
```

Apex (optional):
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext --user
```

If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the configuration files:
```
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Data preparation

Prepare COCO according to the guidelines in [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).


## Results and models

- SimViT on COCO


| Method     | Backbone | Pretrain    | Lr schd | Aug | box AP | mask AP | Config                                               | Download |
|------------|----------|-------------|:-------:|:---:|:------:|:-------:|------------------------------------------------------|----------|
| RetinaNet  | PVTv2-b0 | ImageNet-1K |    1x   |  No |  37.2  |    -    | [config](configs/tmp/retinanet_pvt_v2_b0_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1190iSH3oas_71DPEVjMK9JTn59RYdF3T/view?usp=sharing) & [model](https://drive.google.com/file/d/1K6OkU3CYVglnLSDSvsY8HpDcISB6eKzM/view?usp=sharing) |
| RetinaNet  | PVTv2-b1 | ImageNet-1K |    1x   |  No |  41.2  |    -    | [config](configs/tmp/retinanet_pvt_v2_b1_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/19Wsg25yvKdiqMjXWIFEe-DcaHlkw3iSv/view?usp=sharing) & [model](https://drive.google.com/file/d/1UyBfxAyQygVgAtBeynXrG2iCJj70kiP9/view?usp=sharing) |
| RetinaNet  | PVTv2-b2-li | ImageNet-1K |    1x   |  No |  43.6  |    -    | [config](configs/tmp/retinanet_pvt_v2_b3_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1PRSG3q0M_ZztMMbTxB_961ta4I626T-T/view?usp=sharing) & [model](https://drive.google.com/file/d/1v3j4D1FZuasPi6lGHoHM3bok7PM8F1sg/view?usp=sharing) |
| RetinaNet  | PVTv2-b2 | ImageNet-1K |    1x   |  No |  44.6  |    -    | [config](configs/tmp/retinanet_pvt_v2_b2_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1AMkXwopXLJtW71zT3MjXo0YdoojUbxpQ/view?usp=sharing) & [model](https://drive.google.com/file/d/1VqrLiQ0329HpqiG3BU3q0LoXi6ncS1_k/view?usp=sharing) |
| RetinaNet  | PVTv2-b3 | ImageNet-1K |    1x   |  No |  45.9  |    -    | [config](configs/tmp/retinanet_pvt_v2_b3_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1L59JWC2jepRMT-l5lo8bqygSUGixfGsr/view?usp=sharing) & [model](https://drive.google.com/file/d/1Lz4qRtDoYT8RvDpVxJCvstM3qTtHlLqL/view?usp=sharing) |
| RetinaNet  | PVTv2-b4 | ImageNet-1K |    1x   |  No |  46.1  |    -    | [config](configs/tmp/retinanet_pvt_v2_b4_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1uzFo1W0ARXZfIxBOAGDRi56LpUVIXQNu/view?usp=sharing) & [model](https://drive.google.com/file/d/1GCiE6tniZrG36vumnGi9d79xQEXCS-l2/view?usp=sharing) |
| RetinaNet  | PVTv2-b5 | ImageNet-1K |    1x   |  No |  46.2  |    -    | [config](configs/tmp/retinanet_pvt_v2_b5_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/10bxZEXFQSTVWOWjx2WOgKpoohJS51iPd/view?usp=sharing) & [model](https://drive.google.com/file/d/10cUAXpajabSpAJVSPRNywkOiUlgIjN0e/view?usp=sharing) |
| Mask R-CNN | PVTv2-b0 | ImageNet-1K |    1x   |  No |  38.2  |   36.2  | [config](configs/tmp/mask_rcnn_pvt_v2_b0_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1F4ILBaFnsjwB-_3H8R9lAuf6aZXf8i-V/view?usp=sharing) & [model](https://drive.google.com/file/d/1eRDCU0Erv-kWwCFCwU_VljdjpHk9ktAY/view?usp=sharing) |
| Mask R-CNN | PVTv2-b1 | ImageNet-1K |    1x   |  No |  41.8  |   38.8  | [config](configs/tmp/mask_rcnn_pvt_v2_b1_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1huXFBdZjAtW2PjRUGq5ByNFiuNvy79Va/view?usp=sharing) & [model](https://drive.google.com/file/d/1Y4xgILkl7bh3-DV3rksZTGlN_VTLAIkO/view?usp=sharing) |
| Mask R-CNN | PVTv2-b2-li | ImageNet-1K |    1x   |  No |  44.1  |   40.5  | [config](configs/tmp/mask_rcnn_pvt_v2_b3_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1_SsUAOrH73OpSkhx0g15i6bxCn9I826Y/view?usp=sharing) & [model](https://drive.google.com/file/d/1DWUryElNqWzaNuPafWL0GuOSQxqvqxba/view?usp=sharing) |
| Mask R-CNN | PVTv2-b2 | ImageNet-1K |    1x   |  No |  45.3  |   41.2  | [config](configs/tmp/mask_rcnn_pvt_v2_b2_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1s0L3Dfk7GIKpSmIX0zFRAArADaJy9MeM/view?usp=sharing) & [model](https://drive.google.com/file/d/1F3-FBjDLkskZFvhECO3sSkHwBg8Mx_j0/view?usp=sharing) |
| Mask R-CNN | PVTv2-b3 | ImageNet-1K |    1x   |  No |  47.0  |   42.5  | [config](configs/tmp/mask_rcnn_pvt_v2_b3_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/17fDDXDVim6rFKcrDZCp8ox0_RbeN4yPd/view?usp=sharing) & [model](https://drive.google.com/file/d/1Uq9KpUSLt1-B_6tgVTc1-neuxYBMTI1S/view?usp=sharing) |
| Mask R-CNN | PVTv2-b4 | ImageNet-1K |    1x   |  No |  47.5  |   42.7  | [config](configs/tmp/mask_rcnn_pvt_v2_b4_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1FVWU1ohn19DBOuCZPb8A9DFfrMa5yYm2/view?usp=sharing) & [model](https://drive.google.com/file/d/1IpdgEHAe3XNlIldk6drzOWRcE-wFCv7v/view?usp=sharing) |
| Mask R-CNN | PVTv2-b5 | ImageNet-1K |    1x   |  No |  47.4  |   42.5  | [config](configs/tmp/mask_rcnn_pvt_v2_b5_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/19LN-8TWsrVKsF5aBzXiqKva5mQrAusDw/view?usp=sharing) & [model](https://drive.google.com/file/d/1BvI5XXaGbv3tbLrXbVQ5K45gFVEHbBGX/view?usp=sharing) |



## Evaluation
To evaluate PVT-Small + RetinaNet (640x) on COCO val2017 on a single node with 8 gpus run:
```
dist_test.sh configs/retinanet_pvt_s_fpn_1x_coco_640.py /path/to/checkpoint_file 8 --out results.pkl --eval bbox
```
This should give
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.593
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.408
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.212
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.416
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.544
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.545
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.545
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.545
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.329
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.583
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.721
```

## Training
To train PVT-Small + RetinaNet (640x) on COCO train2017 on a single node with 8 gpus for 12 epochs run:

```
dist_train.sh configs/retinanet_pvt_s_fpn_1x_coco_640.py 8
```

## Demo
```
python demo.py demo.jpg /path/to/config_file /path/to/checkpoint_file
```


## Calculating FLOPS & Params

```
python get_flops.py configs/gfl_pvt_v2_b2_fpn_3x_mstrain_fp16.py
```
This should give
```
Input shape: (3, 1280, 800)
Flops: 260.65 GFLOPs
Params: 33.11 M
```

