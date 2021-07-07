## Vision Transformers for Dense Prediction

This repository contains code and models for our [paper](https://arxiv.org/abs/2103.13413):

> Vision Transformers for Dense Prediction  
> René Ranftl, Alexey Bochkovskiy, Vladlen Koltun


### Changelog 
* [March 2021] Initial release of inference code and models

### Setup 

1) Download the model weights and place them in the `weights` folder:


Monodepth:
- [dpt_hybrid-midas-501f0c75.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt), [Mirror](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view?usp=sharing)
- [dpt_large-midas-2f21e586.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt), [Mirror](https://drive.google.com/file/d/1vnuhoMc6caF-buQQ4hK0CeiMk9SjwB-G/view?usp=sharing)

Segmentation:
 - [dpt_hybrid-ade20k-53898607.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-ade20k-53898607.pt), [Mirror](https://drive.google.com/file/d/1zKIAMbltJ3kpGLMh6wjsq65_k5XQ7_9m/view?usp=sharing)
 - [dpt_large-ade20k-b12dca68.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-ade20k-b12dca68.pt), [Mirror](https://drive.google.com/file/d/1foDpUM7CdS8Zl6GPdkrJaAOjskb7hHe-/view?usp=sharing)
  
2) Set up dependencies: 

    ```shell
    pip install -r requirements.txt
    ```

   The code was tested with Python 3.7, PyTorch 1.8.0, OpenCV 4.5.1, and timm 0.4.5

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

- Monodepth finetuned on KITTI: [dpt_hybrid_kitti-cb926ef4.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_kitti-cb926ef4.pt) [Mirror](https://drive.google.com/file/d/1-oJpORoJEdxj4LTV-Pc17iB-smp-khcX/view?usp=sharing)
- Monodepth finetuned on NYUv2: [dpt_hybrid_nyu-2ce69ec7.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt) [Mirror](https\://drive.google.com/file/d/1NjiFw1Z9lUAfTPZu4uQ9gourVwvmd58O/view?usp=sharing)

Run with 

```shell
python run_monodepth -t [dpt_hybrid_kitti|dpt_hybrid_nyu] 
```

### Evaluation

Hints on how to evaluate monodepth models can be found here: https://github.com/intel-isl/DPT/blob/main/EVALUATION.md


### Citation

Please cite our papers if you use this code or any of the models. 
```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
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
