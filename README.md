## Vision Transformers for Dense Prediction

This repository contains code and models for our [paper](TODO):

>Vision Transformers for Dense Prediction
René Ranftl, Alexey Bochkovskiy, Vladlen Koltun


### Changelog 
* [March 2021] Initial release of inference code and models

### Setup 

1) Download the model weights and place them in the `weights` folder:


Monodepth:
- [dpt_hybrid-midas-501f0c75.pt](TODO)
- [dpt_large-midas-2f21e586.pt](TODO) 


Segmentation:
 - [dpt_hybrid-ade20k-53898607.pt](TODO)
 - [dpt_large-ade20k-XXXXXXXX.pt](TODO)
  
2) Set up dependencies: 

    ```shell
    conda install pytorch torchvision opencv
    ```

   The code was tested with Python 3.7, PyTorch 1.8.0, and OpenCV 4.5.1.

    
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

You can use the flag `-t` to switch between different models. Possible options are `dpt_hybrid` (default) and `dpt_large`.


### Citation

Please cite our paper if you use this code or any of the models:
```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiV Preprint},
	year      = {2021},
}
```

### License 

MIT License 
