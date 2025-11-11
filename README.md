# TriNet: A Triple-Stream Deformable Pyramid Network for Unsupervised Medical Image Registration

This paper proposes a Triple-stream Deformable Pyramid Network (TriNet) for unsupervised medical image registration, aiming to address the issues of existing deep learning-based registration methods, such as the lack of explicit modeling of spatial correspondences in encoders and detail loss caused by decoder upsampling.

![TriNet Architecture](https://github.com/Jincheng05/TriNet/blob/main/TriNet.png)



## Requirements

- <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-ff69b4.svg" /></a>
- <a href= "https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-2.1.1+-lightgreen.svg" /></a>
- <a href= "https://docs.pytorch.org/vision/stable/index.html"> <img src="https://img.shields.io/badge/TorchVision-0.16.1+-yellow.svg" /></a>
- NumPy
- SciPy
- Matplotlib
- nibabel
- scikit-image
- tensorboard
- natsort
- ml_collections
- pystrum

## Dataset
LONI Probabilistic Brain Atlas (LPBA) [Link](https://resource.loni.usc.edu/resources/atlases-downloads/)

Open Access Series of Imaging Studies (OASIS):
- This dataset was made available by [Andrew Hoopes](https://www.nmr.mgh.harvard.edu/user/3935749) and [Adrian V. Dalca](http://www.mit.edu/~adalca/) for the [HyperMorph](https://arxiv.org/abs/2101.01035).    
- OASIS in `.pkl` format processed by the Transmorph's team. [Link](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph_on_OASIS.md)

IXI:
- Official [Link](https://brain-development.org/ixi-dataset/)
- IXI in `.pkl` format processed by the Transmorph's team. [Link](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md)

Please note that the IXI and OASIS datasets we use are all files in `.pkl` format processed by [Transmorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration).


## Training

To train the model, run:

```bash
python train_TriNet_IXI.py
```

Key training parameters can be adjusted in the `train_TriNet_IXI.py` file:

- `batch_size`: Training batch size
- `lr`: Learning rate
- `max_epoch`: Maximum number of training epochs
- `weights`: Loss function weights

The training script will automatically create experiment directories and log files.

## Inference

To perform inference using a trained model:

```bash
python infer_TriNet_IXI.py
```

Make sure to update the model path in the inference script to point to your trained model weights.

## Results

TriNet achieves state-of-the-art performance on the IXI, OASIS and LPBA datasets for medical image registration. The model produces accurate displacement fields that can align brain MRI scans with high precision.

## License <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

For any questions or issues, please open an issue on the GitHub repository.
