import os
import argparse
import shutil
import sys
from subprocess import call
from types import SimpleNamespace
from random import randint
import gradio as gr
import run_segmentation
import run_monodepth
from PIL import Image

def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)

run_cmd("wget --continue https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt -O weights/dpt_large-midas-2f21e586.pt")
run_cmd("wget --continue https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-ade20k-b12dca68.pt -O weights/dpt_large-ade20k-b12dca68.pt")

def process(img, inference_type):
    _id = randint(1, 10000)
    INPUT_DIR = "/tmp/input_image" + str(_id) + "/"
    OUTPUT_DIR = "/tmp/output_image" + str(_id) + "/"
    run_cmd("rm -rf " + INPUT_DIR)
    run_cmd("rm -rf " + OUTPUT_DIR)
    run_cmd("mkdir " + INPUT_DIR)
    run_cmd("mkdir " + OUTPUT_DIR)
    basewidth = 512
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(INPUT_DIR + "1.jpg", "JPEG")
    opts = SimpleNamespace(input_folder=INPUT_DIR, output_folder=OUTPUT_DIR)
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if inference_type == "monodepth":
      run_monodepth.run(INPUT_DIR, OUTPUT_DIR, "weights/dpt_large-midas-2f21e586.pt", "dpt_large")
    else:
      run_segmentation.run(INPUT_DIR, OUTPUT_DIR, "weights/dpt_large-ade20k-b12dca68.pt", "dpt_large")

    return os.path.join(OUTPUT_DIR, "1.png")

title = "DPT"
description = "demo for Dense Prediction Transformers for dense prediction tasks using vision transformers. To use it, simply upload your image, or click one of the examples to load them and select one of the options for monodepth or segmentation. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2103.13413'>Vision Transformers for Dense Prediction</a> | <a href='https://github.com/intel-isl/DPT'>Github Repo</a></p>"

examples = [
    ["elephant.jpg"],
    ["kangaroo.jpg"]
]

gr.Interface(
    process, 
    [gr.inputs.Image(type="pil"), gr.inputs.Radio(["monodepth", "segmentation"], type="value", label="Inference type")], 
    "image", title=title, description=description, examples=examples, article=article).launch(debug=True)

