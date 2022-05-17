# sudo cog push r8.im/yael-vinker/clipasso

# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from cog import BasePredictor, Input, Path
import subprocess as sp
import os
import re

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pydiffvg
import torch
from PIL import Image
import multiprocessing as mp
from shutil import copyfile

import argparse
import math
import sys
import time
import traceback

import PIL
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchvision import models, transforms
from tqdm import tqdm

import config
import sketch_utils as utils
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.num_iter = 2001
        self.save_interval = 100
        self.num_sketches = 3
        self.use_gpu = True

    def predict(
        self,
        target_image: Path = Input(description="Input image (square, without background)"),
        num_strokes: int = Input(description="The number of strokes used to create the sketch, which determines the level of abstraction",default=16),
        trials: int = Input(description="It is recommended to use 3 trials to recieve the best sketch, but it might be slower",default=3),
        mask_object: int = Input(description="It is recommended to use images without a background, however, if your image contains a background, you can mask it out by using this flag with 1 as an argument",default=0),
        fix_scale: int = Input(description="If your image is not squared, it might be cut off, it is recommended to use this flag with 1 as input to automatically fix the scale without cutting the image",default=0),
    ) -> Path:
        
        self.num_sketches = trials
        target_image_name = os.path.basename(str(target_image))

        multiprocess = False
        abs_path = os.path.abspath(os.getcwd())

        target = str(target_image)
        assert os.path.isfile(target), f"{target} does not exists!"

        test_name = os.path.splitext(target_image_name)[0]
        output_dir = f"{abs_path}/output_sketches/{test_name}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("=" * 50)
        print(f"Processing [{target_image_name}] ...")
        print(f"Results will be saved to \n[{output_dir}] ...")
        print("=" * 50)

        if not torch.cuda.is_available():
            self.use_gpu = False
            print("CUDA is not configured with GPU, running with CPU instead.")
            print("Note that this will be very slow, it is recommended to use colab.")
        print(f"GPU: {self.use_gpu}")
        seeds = list(range(0, self.num_sketches * 1000, 1000))

        losses_all = {}
        
        for seed in seeds:
            wandb_name = f"{test_name}_{num_strokes}strokes_seed{seed}"
            sp.run(["python", "config.py", target,
                        "--num_paths", str(num_strokes),
                        "--output_dir", output_dir,
                        "--wandb_name", wandb_name,
                        "--num_iter", str(self.num_iter),
                        "--save_interval", str(self.save_interval),
                        "--seed", str(seed),
                        "--use_gpu", str(int(self.use_gpu)),
                        "--fix_scale", str(fix_scale),
                        "--mask_object", str(mask_object),
                        "--mask_object_attention", str(
                            mask_object),
                        "--display_logs", str(int(0))])
            config_init = np.load(f"{output_dir}/{wandb_name}/config_init.npy", allow_pickle=True)[()]
            args = Args(config_init)
            args.cog_display = True
            
            final_config = vars(args)
            try:
                configs_to_save = main(args)
            except BaseException as err:
                print(f"Unexpected error occurred:\n {err}")
                print(traceback.format_exc())
                sys.exit(1)
            for k in configs_to_save.keys():
                final_config[k] = configs_to_save[k]
            np.save(f"{args.output_dir}/config.npy", final_config)
            if args.use_wandb:
                wandb.finish()

            config = np.load(f"{output_dir}/{wandb_name}/config.npy",
                            allow_pickle=True)[()]
            loss_eval = np.array(config['loss_eval'])
            inds = np.argsort(loss_eval)
            losses_all[wandb_name] = loss_eval[inds][0]
            # return Path(f"{output_dir}/{wandb_name}/best_iter.svg") 


        sorted_final = dict(sorted(losses_all.items(), key=lambda item: item[1]))
        copyfile(f"{output_dir}/{list(sorted_final.keys())[0]}/best_iter.svg",
                f"{output_dir}/{list(sorted_final.keys())[0]}_best.svg")
        target_path = f"{abs_path}/target_images/{target_image_name}"
        svg_files = os.listdir(output_dir)
        svg_files = [f for f in svg_files if "best.svg" in f]
        svg_output_path = f"{output_dir}/{svg_files[0]}"
        sketch_res = read_svg(svg_output_path, multiply=True).cpu().numpy()
        sketch_res = Image.fromarray((sketch_res * 255).astype('uint8'), 'RGB')
        sketch_res.save(f"{abs_path}/output_sketches/sketch.png")
        return Path(svg_output_path)


class Args():
    def __init__(self, config):
        for k in config.keys():
            setattr(self, k, config[k])
        

def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_, mask


def main(args):
    loss_func = Loss(args)
    inputs, mask = get_target(args)
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    renderer = load_renderer(args, inputs, mask)

    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss, best_fc_loss = 100, 100
    best_iter, best_iter_fc = 0, 0
    min_delta = 1e-5
    terminate = False

    renderer.set_random_noise(0)
    img = renderer.init_image(stage=0)
    optimizer.init_optimizers()

    for epoch in tqdm(range(args.num_iter)):
        renderer.set_random_noise(epoch)
        if args.lr_scheduler:
            optimizer.update_lr(counter)

        start = time.time()
        optimizer.zero_grad_()
        sketches = renderer.get_image().to(args.device)
        losses_dict = loss_func(sketches, inputs.detach(
        ), renderer.get_color_parameters(), renderer, counter, optimizer)
        loss = sum(list(losses_dict.values()))
        loss.backward()
        optimizer.step_()
        if epoch % args.save_interval == 0:
            utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                             use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
            # if args.cog_display:
            #     yield Path(f"{args.output_dir}/svg_logs/svg_iter{epoch}.svg")


        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(
                ), renderer.get_points_parans(), counter, optimizer, mode="eval")
                loss_eval = sum(list(losses_dict_eval.values()))
                configs_to_save["loss_eval"].append(loss_eval.item())
                for k in losses_dict_eval.keys():
                    if k not in configs_to_save.keys():
                        configs_to_save[k] = []
                    configs_to_save[k].append(losses_dict_eval[k].item())
                if args.clip_fc_loss_weight:
                    if losses_dict_eval["fc"].item() < best_fc_loss:
                        best_fc_loss = losses_dict_eval["fc"].item(
                        ) / args.clip_fc_loss_weight
                        best_iter_fc = epoch
                # print(
                #     f"eval iter[{epoch}/{args.num_iter}] loss[{loss.item()}] time[{time.time() - start}]")

                cur_delta = loss_eval.item() - best_loss
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = epoch
                        terminate = False
                        utils.plot_batch(
                            inputs, sketches, args.output_dir, counter, use_wandb=args.use_wandb, title="best_iter.jpg")
                        renderer.save_svg(args.output_dir, "best_iter")

                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_loss_fc"] = best_fc_loss
                    wandb_dict = {"delta": cur_delta,
                                  "loss_eval": loss_eval.item()}
                    for k in losses_dict_eval.keys():
                        wandb_dict[k + "_eval"] = losses_dict_eval[k].item()
                    wandb.log(wandb_dict, step=counter)

                if abs(cur_delta) <= min_delta:
                    if terminate:
                        break
                    terminate = True

        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                             args.use_wandb, "{}/{}.jpg".format(
                                 args.output_dir, "attention_map"),
                             args.saliency_model, args.display_logs)

        if args.use_wandb:
            wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
            for k in losses_dict.keys():
                wandb_dict[k] = losses_dict[k].item()
            wandb.log(wandb_dict, step=counter)

        counter += 1

    renderer.save_svg(args.output_dir, "final_svg")
    path_svg = os.path.join(args.output_dir, "best_iter.svg")
    utils.log_sketch_summary_final(
        path_svg, args.use_wandb, args.device, best_iter, best_loss, "best total")

    return configs_to_save


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
