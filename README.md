# Using Unreliable Pseudo Labels (modified)

Modified model from [Semi-Supervised Semantic Segmentation Using Unreliable Pseudo Labels](https://arxiv.org/abs/2203.03884), CVPR 2022.

Refer to [U2PL GitHub](https://github.com/ponoma1202/U2PL_copy/blob/main/README.md) for the official U2PL model code. 

## Installation Guide
After cloning the repository,
```
git clone https://github.com/ponoma1202/U2PL_copy.git && cd U2PL
conda create -n u2pl python=3.8.16
pip install -r requirements.txt
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
```
Used pytorch version 2.0.1 and torchvision version 0.15.2.

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

2. Download and unzip [SegmentationClassAug.zip](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)

3. Unzip the `archive.zip` file into `data` from Kaggle download. Unzipped `archive` file should contain the `VOC2012` folder. Delete the extra `archive/VOC2012/VOC2012` folder.

4. Move the unzipped `SegmentationClassAug/SegmentationClassAug` folder into the VOC2012 folder.

Move `VOC2012` into the `U2PL` project folder. The path should be `U2PL/data/VOC2012`. File directory should look like this:
- `data/VOC2012`
    - `Annotations`
    - `ImageSets`
    - `JPEGImages`
    - `SegmentationClass`
    - `SegmentationClassAug`
    - `SegmentationObject`

</details>

The `data` folder should now contain at least one of the following folders in addition to the `splits` folder:
- `cityscapes`
- `VOC2012`

### Download Pre-trained ResNet101

1. Download [resnet101.pth](https://drive.google.com/file/d/1nzSX8bX3zoRREn6WnoEeAPbKYPPOa-3Y/view?usp=sharing) file

2. Replace `/path/to/resnet101.pth` at the top of the `u2pl/models/resnet.py` file under the `model_urls = {"resnet101": "/path/to/resnet101.pth"}` variable with file path of `resnet101.pth`.

### Additional Set-Up Notes
1. Replace the relative paths for `data_root` and `data_list` values in the configuration files with explicit paths.

2. Edit the TODO's in each sbatch shell script. 

The U2PL model used a batch size of 16, however, the replicated U2PL model could only use a batch size of 14 before running out of 80 GB of memory on an A100 GPU.

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

## Training the Model

### Input Arguments
- **config**: specify file path for configuration file (.yaml)
- **seed**: set to 2 in original U2PL model
- **output_dirpath**: specify file path for output directory for plots of tracked parameters and copy of the dictionary of the trained model

### Accuracy Metrics Tracked
The original U2PL model uses intersection over union (IoU) as its benchmark for accuracy. In the modified U2PL model, IoU remains
the benchmark for accuracy, however, other accuracy metrics are also tracked. All plots for accuracy metrics, along with a csv file
of all tracked metrics are generated in the folder specified by `output_dirpath` argument.

- **IoU**: intersection over union. Tracked only for validation.
- **accuracy**: number of pixels classified correctly / total number of pixels in image. Tracked for both training and validation
- **ARI**: adjusted random score

### Using Sbatch Files
Each sbatch file is located in the main `U2PL` project directory. Edit TODO's before running.

## Inferencing
Use the infer.py file with the following arguments:
- **config**: specify file path for configuration file (.yaml) used during training
- **model_path**: path to the model-state-dict.pt file located in the `output_dirpath` folder (3rd input argument for model training)
- **save_folder**: path to folder into which inferencing images will be saved

To compare to the original U2PL model results, download the model checkpoints from the U2PL GitHub README file.

## Acknowledgement

The `pytorch_utils` folder containing: `plateau_scheduler`, used for stopping the training of the model early if learning rate plateaus,
and `training_stats`, used to track metadata, were taken from Michael Majursky's https://github.com/usnistgov/semantic-segmentation-unet/tree/pytorch.
