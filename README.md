# Using Unreliable Pseudo Labels

Modified model from [Semi-Supervised Semantic Segmentation Using Unreliable Pseudo Labels](https://arxiv.org/abs/2203.03884), CVPR 2022.

Refer to [U2PL GitHub](https://github.com/ponoma1202/U2PL_copy/blob/main/README.md) for the official U2PL model code. 

## Installation Guide
After cloning the repository,
```
conda create -n u2pl python=3.8.16
pip install -r requirements.txt
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Setting Up the Data
The U2PL model is trained on both Citiscapes and PASCAL VOC 2012 datasets.

<details>
  <summary><b>For Cityscapes</b></summary>

1. Download [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/downloads/)

2. Download [gtFine.zip](https://drive.google.com/file/d/10tdElaTscdhojER_Lf7XlytiyAkk7Wlg/view?usp=sharing) from Google Drive

3. Unzip `gtFine` and `leftImg8bit_trainvaltest` into a new folder named `citiscapes`. 

4. Move `citiscapes` folder into `data` folder.

Note: both `gtFine` and `leftImg8bit_trainvaltest` contain:
  - `train`
  - `test`
  - `val`

</details>

<details>
  <summary><b>For ResNet</b></summary>

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012/code)

2. Download [SegmentationClassAug.zip](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)

3. Unzip the `archive.zip` file into `data`. Unzipped file should be called `VOC2012`.

4. Move `SegmentationClassAug` into the VOC2012 folder.

The path to the `VOC2012` should be `U2PL/data/VOC2012`. File directory should look like this:
- `data/VOC2012`
    - `Annotations`
    - `ImageSets`
    - `JPEGImages`
    - `SegmentationClass`
    - `SegmentationClassAug`
    - `SegmentationObject`

</details>

The `data` folder should now contain at least two of the following folders:
- `cityscapes`
- `splits`
- `VOC2012`

### Download Pre-trained ResNet101

1. Download [resnet101.pth](https://drive.google.com/file/d/1nzSX8bX3zoRREn6WnoEeAPbKYPPOa-3Y/view?usp=sharing) file

2. Replace `/path/to/resnet101.pth` at the top of the `u2pl/models/resnet.py` file under the `model_urls = {"resnet101": "/path/to/resnet101.pth"}` variable with file path of `resnet101.pth`.

### Directory Guide within U2PL project folder
```angular2html
U2PL
├───data
│   └───**splits**
│       ├───cityscapes
│       │   ├───1488
│       │   ├───186
│       │   ├───372
│       │   └───744
│       └───pascal
│           ├───1323
│           ├───1464
│           ├───183
│           ├───2646
│           ├───366
│           ├───5291
│           ├───662
│           ├───732
│           └───92
├───experiments
│   ├───cityscapes
│   │   └───744
│   │       ├───**ours**
│   │       └───suponly
│   └───pascal
│       └───1464
│           ├───**ours
│           └───suponly
├───**pytorch_utils**
├───u2pl
    ├───**dataset**
    ├───models
    └───utils

```

- `data/splits` contains all labeled.txt and unlabeled.txt splits.
- `experiments/pascal/1464/ours/config.yaml` contains config file for semi-supervised model using the PASCAL VOC dataset. Follow similar structure to access config files for Cityscapes.
- `pytorch_utils/lr_scheduler` contains learning rate scheduler with early stopping
- `pytroch_utils/metadata.py` is a tracker for metadata such as training accuracy, learning rate, loss, etc
- `u2pl/dataset/pascal_voc.py` is the DataSet class for the PASCAL VOC dataset. Similar structure for Cityscapes dataset.
- TODO: edit hierarchy of dataset and dataloader constructors to make everything more concise (basically follow Mike's tips)

## Training the Model

### Input Arguments
- **config**: specify file path for configuration file (.yaml)
- **seed**: set to 2 in original U2PL model
- **output_dirpath**: specify file path for output directory for plots of tracked parameters and copy of the dictionary of the trained model

### Using Sbatch Files

### Python Command


## Inferencing
Use the infer.py file with the following arguments:
- **config**: specify file path for configuration file (.yaml) used during training
- **model_path**: path to the model-state-dict.py file which should be located in the `output_dirpath` folder, specified when running the model (3rd input argument).
- **save_folder**: path to folder into which inferencing images will be saved

To compare to the original U2PL model results, download the model checkpoints from the U2PL GitHub README file.


## Usage

U<sup>2</sup>PL is evaluated on both Cityscapes and PASCAL VOC 2012 dataset.
### Prepare Data

<details>
  <summary>For Cityscapes</summary>

Download "leftImg8bit_trainvaltest.zip" from: https://www.cityscapes-dataset.com/downloads/

Download "gtFine.zip" from: https://drive.google.com/file/d/10tdElaTscdhojER_Lf7XlytiyAkk7Wlg/view?usp=sharing

Next, unzip the files to folder ```data``` and make the dictionary structures as follows:

```angular2html
data/cityscapes
├── gtFine
│   ├── test
│   ├── train
│   └── val
└── leftImg8bit
    ├── test
    ├── train
    └── val
```

</details>


<details>
  <summary>For PASCAL VOC 2012</summary>

Refer to [this link](https://github.com/zhixuanli/segmentation-paper-reading-notes/blob/master/others/Summary%20of%20the%20semantic%20segmentation%20datasets.md) and download ```PASCAL VOC 2012 augmented with SBD``` dataset.

And unzip the files to folder ```data``` and make the dictionary structures as follows:

```angular2html
data/VOC2012
├── Annotations
├── ImageSets
├── JPEGImages
├── SegmentationClass
├── SegmentationClassAug
└── SegmentationObject
```
</details>

Finally, the structure of dictionary ```data``` should be as follows:

```angular2html
data
├── cityscapes
│   ├── gtFine
│   └── leftImg8bit
├── splits
│   ├── cityscapes
│   └── pascal
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```

### Prepare Pretrained Backbone

Before training, please download ResNet101 pretrained on ImageNet-1K from one of the following:
  - [Google Drive](https://drive.google.com/file/d/1nzSX8bX3zoRREn6WnoEeAPbKYPPOa-3Y/view?usp=sharing)
  - [Baidu Drive](https://pan.baidu.com/s/1FDQGlhjzQENfPp4HTYfbeA) Fetch Code: 3p9h

After that, modify ```model_urls``` in ```semseg/models/resnet.py``` to ```</path/to/resnet101.pth>```

### Train a Fully-Supervised Model

For instance, we can train a model on PASCAL VOC 2012 with only ```1464``` labeled data for supervision by:
```bash
cd experiments/pascal/1464/suponly
# use torch.distributed.launch
sh train.sh <num_gpu> <port>

# or use slurm
# sh slurm_train.sh <num_gpu> <port> <partition>
```
Or for Cityscapes, a model supervised by only ```744``` labeled data can be trained by:
```bash
cd experiments/cityscapes/744/suponly
# use torch.distributed.launch
sh train.sh <num_gpu> <port>

# or use slurm
# sh slurm_train.sh <num_gpu> <port> <partition>
```
After training, the model should be evaluated by
```bash
sh eval.sh
```
### Train a Semi-Supervised Model

We can train a model on PASCAL VOC 2012 with ```1464``` labeled data and ```9118``` unlabeled data for supervision by:
```bash
cd experiments/pascal/1464/ours
# use torch.distributed.launch
sh train.sh <num_gpu> <port>

# or use slurm
# sh slurm_train.sh <num_gpu> <port> <partition>
```
Or for Cityscapes, a model supervised by ```744``` labeled data and ```2231``` unlabeled data can be trained by:
```bash
cd experiments/cityscapes/744/ours
# use torch.distributed.launch
sh train.sh <num_gpu> <port>

# or use slurm
# sh slurm_train.sh <num_gpu> <port> <partition>
```
After training, the model should be evaluated by
```bash
sh eval.sh
```

### Train a Semi-Supervised Model on Cityscapes with AEL

First, you should switch the branch:
```bash
git checkout with_AEL
```
Then, we can train a model supervised by ```744``` labeled data and ```2231``` unlabeled data by:
```bash
cd experiments/city_744
# use torch.distributed.launch
sh train.sh <num_gpu> <port>

# or use slurm
# sh slurm_train.sh <num_gpu> <port> <partition>
```
After training, the model should be evaluated by
```bash
sh eval.sh
```

### Note
```<num_gpu>``` means the number of GPUs for training.

To reproduce our results, we recommend you follow the settings:
- Cityscapes: ```4 * V100 (32G)``` for SupOnly and ```8 * V100 (32G)``` for Semi-Supervised
- PASCAL VOC 2012: ```2 * V100 (32G)``` for SupOnly and ```4 * V100 (32G)``` for Semi-Supervised

If you got ```CUDA Out of Memory``` error, please try training our method in [fp16](https://github.com/NVIDIA/apex) mode.
Or, change the ```lr``` in ```config.yaml``` in a linear manner (*e.g.*, if you want to train a SupOnly model on Cityscapes with 8 GPUs, 
you are recommended to change the ```lr``` to ```0.02```).

If you want to train a model on other split, you need to modify ```data_list``` and ```n_sup``` in ```config.yaml```.

Due to the randomness of function ```torch.nn.functional.interpolate``` when ```mode="bilinear"```, 
the results of semantic segmentation will not be the same EVEN IF a fixed random seed is set.

Therefore, we recommend you run 3 times and get the average performance.

## License

This project is released under the [Apache 2.0](LICENSE) license.

## Acknowledgement

The contrastive learning loss and strong data augmentation (CutMix, CutOut, and ClassMix) 
are borrowed from **ReCo**.
We reproduce our U<sup>2</sup>PL based on **AEL** on branch ```with_AEL```.
- ReCo: https://github.com/lorenmt/reco
- AEL: https://github.com/hzhupku/SemiSeg-AEL

Thanks a lot for their great work!

## Citation
```bibtex
@inproceedings{wang2022semi,
    title={Semi-Supervised Semantic Segmentation Using Unreliable Pseudo Labels},
    author={Wang, Yuchao and Wang, Haochen and Shen, Yujun and Fei, Jingjing and Li, Wei and Jin, Guoqiang and Wu, Liwei and Zhao, Rui and Le, Xinyi},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```

## Contact

- Yuchao Wang, 44442222@sjtu.edu.cn
- Haochen Wang, wanghaochen2022@ia.ac.cn
- Jingjing Fei, feijingjing1@sensetime.com
- Wei Li, liwei1@sensetime.com
