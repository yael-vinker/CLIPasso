import argparse
import os
import re

import imageio
import matplotlib.pyplot as plt
import moviepy.editor as mvp
import numpy as np
import pydiffvg
import torch
from IPython.display import Image as Image_colab
from IPython.display import display, SVG
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--target_file", type=str,
                    help="target image file, located in <target_images>")
parser.add_argument("--num_strokes", type=int)
args = parser.parse_args()


def read_svg(path_svg, multiply=False):
    device = torch.device("cuda" if (
        torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        path_svg)
    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= 2
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
        torch.ones(img.shape[0], img.shape[1], 3,
                   device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    return img


abs_path = os.path.abspath(os.getcwd())
target_path = f"{abs_path}/target_images/{args.target_file}"
result_path = f"{abs_path}/output_sketches/{os.path.splitext(args.target_file)[0]}"
svg_files = os.listdir(result_path)
svg_files = [f for f in svg_files if "best.svg" in f and f"{args.num_strokes}strokes" in f]
svg_output_path = f"{result_path}/{svg_files[0]}"
sketch_res = read_svg(svg_output_path, multiply=True).cpu().numpy()
sketch_res = Image.fromarray((sketch_res * 255).astype('uint8'), 'RGB')

input_im = Image.open(target_path).resize((224,224))
display(input_im)
display(SVG(svg_output_path))

p = re.compile("_best")
best_sketch_dir = ""
for m in p.finditer(svg_files[0]):
    best_sketch_dir += svg_files[0][0: m.start()]


sketches = []
cur_path = f"{result_path}/{best_sketch_dir}"
sketch_res.save(f"{cur_path}/final_sketch.png")
print(f"You can download the result sketch from {cur_path}/final_sketch.png")

if not os.path.exists(f"{cur_path}/svg_to_png"):
    os.mkdir(f"{cur_path}/svg_to_png")
if os.path.exists(f"{cur_path}/config.npy"):
    config = np.load(f"{cur_path}/config.npy", allow_pickle=True)[()]
    inter = config["save_interval"]
    loss_eval = np.array(config['loss_eval'])
    inds = np.argsort(loss_eval)
    intervals = list(range(0, (inds[0] + 1) * inter, inter))
    for i_ in intervals:
        path_svg = f"{cur_path}/svg_logs/svg_iter{i_}.svg"
        sketch = read_svg(path_svg, multiply=True).cpu().numpy()
        sketch = Image.fromarray((sketch * 255).astype('uint8'), 'RGB')
        # print("{0}/iter_{1:04}.png".format(cur_path, int(i_)))
        sketch.save("{0}/{1}/iter_{2:04}.png".format(cur_path, "svg_to_png", int(i_)))
        sketches.append(sketch)
    imageio.mimsave(f"{cur_path}/sketch.gif", sketches)

print(cur_path)
