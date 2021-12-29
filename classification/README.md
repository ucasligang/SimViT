# SimViT: Exploring a Simple Vision Transformer with sliding windows

Our classification code is developed on top of [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [deit](https://github.com/facebookresearch/deit) and [PVT](https://github.com/whai362/PVT).

We will open our models soon.

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

First, clone the repository locally:
```
git clone https://github.com/ucasligang/SimViT.git
```
Then, install PyTorch 1.6.0+ and torchvision 0.7.0+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Model Zoo

## Model Zoo

- SimViT on ImageNet-1K

| Method           | Size | Acc@1 | #Params (M) | Config                                   | Download                                                                                   |
|------------------|:----:|:-----:|:-----------:|------------------------------------------|--------------------------------------------------------------------------------------------|
| SimViT-micro        |  224 |  71.1 |     3.3     | [config](configs/SimViT/simvit_micro.py)    | 14M [[Google]](https://drive.google.com/file/d/1qnqChpm93vtXULeTuCT_0mJ2ZKIDc-Qo/view?usp=sharing) [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth) |

## Evaluation
To evaluate a pre-trained SimViT-micro on ImageNet val with a 4 GPUs run:
```
sh dist_eval.sh configs/SimViT/simvit_micro.py 4 /path/to/imagenet /path/to/checkpoint_file
```
This should give
```
* Acc@1 71.118 Acc@5 90.516 loss 1.297
Accuracy of the network on the 50000 test images: 71.1%
```

## Training
To train SimViT-micro on ImageNet on a single node with 8 gpus for 300 epochs run:

```
sh dist_train.sh configs/SimViT/simvit_micro.py 8 --data-path /path/to/imagenet
```

## Calculating FLOPS & Params

```
python get_flops.py simvit_micro
```
This should give
```
Input shape: (3, 224, 224)
Flops: 0.65 GFLOPs
Params: 3.33 M
```

