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


| Method     | Backbone | Pretrain    | Lr schd | Aug | box AP | Config                                               | Download |
|------------|----------|-------------|:-------:|:---:|:------:|------------------------------------------------------|----------|
| RetinaNet  | SimViT-Small | ImageNet-1K |    1x   |  No |  46.3 |  [config](configs/tmp/retinanet_pvt_v2_b0_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1190iSH3oas_71DPEVjMK9JTn59RYdF3T/view?usp=sharing) & [model](https://drive.google.com/file/d/1K6OkU3CYVglnLSDSvsY8HpDcISB6eKzM/view?usp=sharing) |
| ATSS| SimViT-Small | ImageNet-1K |    1x   |  No | 49.6  | [config](configs/tmp/mask_rcnn_pvt_v2_b4_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1FVWU1ohn19DBOuCZPb8A9DFfrMa5yYm2/view?usp=sharing) & [model](https://drive.google.com/file/d/1IpdgEHAe3XNlIldk6drzOWRcE-wFCv7v/view?usp=sharing) |
|GFL | SimViT-Small | ImageNet-1K |    1x   |  No |  49.9 | [config](configs/tmp/mask_rcnn_pvt_v2_b5_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/19LN-8TWsrVKsF5aBzXiqKva5mQrAusDw/view?usp=sharing) & [model](https://drive.google.com/file/d/1BvI5XXaGbv3tbLrXbVQ5K45gFVEHbBGX/view?usp=sharing) |



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

