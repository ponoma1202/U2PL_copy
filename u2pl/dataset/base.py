import logging
import os.path

import skimage
from PIL import Image
from torch.utils.data import Dataset
import mrcfile


class BaseDataset(Dataset):
    def __init__(self, d_list, **kwargs):
        # parse the input list
        self.parse_input_list(d_list, **kwargs)

    def parse_input_list(self, d_list, max_sample=-1, start_idx=-1, end_idx=-1):
        logger = logging.getLogger("global")
        assert isinstance(d_list, str)
        if "cityscapes" in d_list:
            self.list_sample = [
                [
                    line.strip(),
                    "gtFine/" + line.strip()[12:-15] + "gtFine_labelTrainIds.png",
                ]
                for line in open(d_list, "r")
            ]
        elif "pascal" in d_list or "VOC" in d_list:
            self.list_sample = [
                [
                    "JPEGImages/{}.jpg".format(line.strip()),
                    "SegmentationClassAug/{}.png".format(line.strip()),
                ]
                for line in open(d_list, "r")
            ]
        elif "v4" in d_list:            # added cryoem to dataset
            self.list_sample = [
                [
                    "images/{}".format(line.strip()),
                    "gold_truth/{}".format(line.replace(".mrc", ".tif").strip())
                ]
                for line in open(d_list, "r")
            ]
        else:
            raise "unknown dataset!"

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        logger.info("# samples: {}".format(self.num_sample))

    def img_loader(self, path, mode):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert(mode)

    def cryoem_img_loader(self, path):
        with mrcfile.open(path) as mrc:
            data = mrc.data.astype('float32')
            return data

    def cryoem_label_loader(self, path):
        img = skimage.io.imread(path, as_gray=True)     # as gray makes pixel values either 0 or 1 by default
        img = (img > 0)
        return img.astype('float32')

    def __len__(self):
        return self.num_sample
