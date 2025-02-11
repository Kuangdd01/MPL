# Momentum Pseudo-Labeling for Weakly Supervised Phrase Grounding
Implementation of MPL: Momentum Pseudo-Labeling for Weakly Supervised Phrase Grounding.
[ [Paper](-) | [Appendix](assets/appendix.pdf) ]

Some of our code is based on MAF & CLEM & volta. Thanks to their excellent works!!

## Prepare
> The all image features are extracted by a Faster R-CNN pre-trained on the Visual Genome with a ResNet-101.
1. For the Flickr image features, we adopted the extracted features from [MAF](https://github.com/qinzzz/Multimodal-Alignment-Framework).
2. For the RefCOCO/RefCOCO+/RefCOCOg image features, we recommend to follow [volta](https://github.com/e-bug/volta/blob/main/data/README.md) instructions to obtain corresponding image features.

## Install
``` shell
conda create -n <your_env> python==3.9
conda activate <your_env>
pip install -r requirements.txt
```
## QuickStart
```shell
# put corresponding data following:
mat_root
├── test_detection_dict.json
├── test_features_compress.hdf5
├── test_imgid2idx.pkl
├── train_detection_dict.json
├── train_features_compress.hdf5
├── train_imgid2idx.pkl
├── val_detection_dict.json
├── val_features_compress.hdf5
└── val_imgid2idx.pkl

dataroot
├── Annotations
├── Sentences
├── annotations.zip
├── cache
├── test.txt
├── train.txt
└── val.txt

referoot
├── images
├── refcoco
├── refcoco+
├── refcocog
└── refer

features_path
├── refcoco
├── refcoco+
├── refcocog
├── uniter.bin
├── vilbert.bin
└── visualbert.bin

bash run.sh
```