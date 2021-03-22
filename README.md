## Vision Transformers for Dense Prediction

This repository contains code and models for our [paper](TODO):

> Vision Transformers for Dense Prediction  
> René Ranftl, Alexey Bochkovskiy, Vladlen Koltun


### Changelog 
* [March 2021] Initial release of inference code and models

### Setup 

1) Download the model weights and place them in the `weights` folder:


Monodepth:
- [dpt_hybrid-midas-501f0c75.pt](TODO), [Mirror](TODO)
- [dpt_large-midas-2f21e586.pt](TODO), [Mirror](TODO)


Segmentation:
 - [dpt_hybrid-ade20k-53898607.pt](TODO), [Mirror](TODO)
 - [dpt_large-ade20k-b12dca68.pt](TODO), [Mirror](TODO)
  
2) Set up dependencies: 

    ```shell
    conda install pytorch torchvision opencv 
    pip install timm
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

3) The results are written to the folder `output_monodepth` and `output_segmentation`, respectively.

Use the flag `-t` to switch between different models. Possible options are `dpt_hybrid` (default) and `dpt_large`.


### Citation

Please cite our paper if you use this code or any of the models:
```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}
```

### Acknowledgements

Our work builds on [timm](https://github.com/rwightman/pytorch-image-models) and [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding). We'd like to thank the authors for making these libraries available.

### License 

MIT License 
