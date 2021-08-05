## Vision Transformers for Dense Prediction

This repository contains code and models for our [paper](https://arxiv.org/abs/2103.13413):

> Vision Transformers for Dense Prediction  
> René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
> ICCV 2021


### Changelog 
* [August 2021] Models refactored to support model scripting and tracing
* [March 2021]  Initial release of inference code and models

### Setup 

1) Download the model weights and place them in the `weights` folder:


Monodepth:
- dpt_hybrid-midas-d889a10e.pt, [Mirror](https://drive.google.com/file/d/1H9EWydg6iasnlLLyPrVP_KYe4oa_3NxP/view?usp=sharing)
- dpt_large-midas-b53ba79e.pt, [Mirror](https://drive.google.com/file/d/1bmo-jMyuuIc_uZPTub_n3mLYnPT33ro7/view?usp=sharing)

Segmentation:
- dpt_hybrid-ade20k-a7d10e8d.pt, [Mirror](https://drive.google.com/file/d/1owEjmYwTI7kadXt77iwQTbMxKSts8ldO/view?usp=sharing)
- dpt_large-ade20k-078062de.pt, [Mirror](https://drive.google.com/file/d/1vCxbb8oNlSI-RSzMCXDWfI1FiqTPLgF2/view?usp=sharing)
  
2) Set up dependencies: 

    ```shell
    pip install -r requirements.txt
    ```

   The code was tested with Python 3.7, PyTorch 1.9.0, OpenCV 4.5.1, and timm 0.4.9

### Usage 

1) Place one or more input images in the folder `input`.

2) Run a monocular depth estimation model:

    ```shell
    python run_monodepth.py
    ```

    Or run a semantic segmentation model:

    ```shell
    python run_segmentation.py
    ```

3) The results are written to the folder `output_monodepth` and `output_semseg`, respectively.

Use the flag `-t` to switch between different models. Possible options are `dpt_hybrid` (default) and `dpt_large`.


**Additional models:**

- Monodepth finetuned on KITTI: dpt_hybrid-kitti-e7069aae.pt [Mirror](https://drive.google.com/file/d/1h9M_KPI43iEc7uuKkGlEcQiAVyMTXNTw/view?usp=sharing)
- Monodepth finetuned on NYUv2: dpt_hybrid-nyu-b3a2ef48.pt [Mirror](https://drive.google.com/file/d/1CgIW_u1vXM2sfx1tkN7GUDOPXe3MbUfd/view?usp=sharing)

Run with 

```shell
python run_monodepth -t [dpt_hybrid_kitti|dpt_hybrid_nyu] 
```

### Evaluation

Hints on how to evaluate monodepth models can be found here: https://github.com/intel-isl/DPT/blob/main/EVALUATION.md


### Citation

Please cite our papers if you use this code or any of the models. 
```
@inproceedings{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	booktitle = {ICCV},
	year      = {2021},
}
```

```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```

### Acknowledgements

Our work builds on and uses code from [timm](https://github.com/rwightman/pytorch-image-models) and [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding). We'd like to thank the authors for making these libraries available.

### License 

MIT License 
