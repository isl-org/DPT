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

def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)

def monodepth(img, inference_type):
    _id = randint(1, 10000)
    INPUT_DIR = "/tmp/input_image" + str(_id) + "/"
    OUTPUT_DIR = "/tmp/output_image" + str(_id) + "/"
    run_cmd("rm -rf " + INPUT_DIR)
    run_cmd("rm -rf " + OUTPUT_DIR)
    run_cmd("mkdir " + INPUT_DIR)
    run_cmd("mkdir " + OUTPUT_DIR)
    img.save(INPUT_DIR + "1.jpg", "JPEG")
    opts = SimpleNamespace(input_folder=INPUT_DIR, output_folder=OUTPUT_DIR)
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if inference_type == "monodepth":
      run_monodepth.run(INPUT_DIR, OUTPUT_DIR, "weights/dpt_large-midas-2f21e586.pt", "dpt_large")
    else:
      run_segmentation.run(INPUT_DIR, OUTPUT_DIR, "weights/dpt_large-ade20k-b12dca68.pt", "dpt_large")

    return os.path.join(OUTPUT_DIR, "1.png")

iface = gr.Interface(
    monodepth, 
    [gr.inputs.Image(type="pil"), gr.inputs.Radio(["monodepth", "segmentation"], type="value", label="Inference type")], 
    "image",
   )
iface.launch(debug=True)

