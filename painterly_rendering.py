from tqdm import tqdm
from torchvision import transforms, models

import torch
import argparse
import math
import wandb
import torch.nn.functional as F
import torch.nn as nn
import os
import sketch_utils as utils
import time
import config
import numpy as np

from torch.cuda.amp import autocast
from PIL import Image
from loss import Loss
from models.painter_params import Painter, PainterOptimizer
import sys 
import PIL


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
        new_image = Image.new("RGBA", target.size, "WHITE")  # Create a white rgba background
        new_image.paste(target, (0, 0), target)  # Paste the image on the background.
        target = new_image
    target = target.convert("RGB")
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize((args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(args.image_scale, interpolation=PIL.Image.BICUBIC))
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
        losses_dict = loss_func(sketches, inputs.detach(), renderer.get_color_parameters(), renderer, counter, optimizer)
        loss = sum(list(losses_dict.values()))
        loss.backward()
        optimizer.step_()
        if epoch % args.save_interval == 0:
            utils.plot_batch(inputs, sketches, args, counter, use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_svg(args.output_dir, f"svg_iter{epoch}")
        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(), renderer.get_points_parans(), counter, optimizer, mode="eval")
                loss_eval = sum(list(losses_dict_eval.values()))
                configs_to_save["loss_eval"].append(loss_eval.item())
                for k in losses_dict_eval.keys():
                    if k not in configs_to_save.keys():
                        configs_to_save[k] = []
                    configs_to_save[k].append(losses_dict_eval[k].item())
                if args.clip_fc_loss_weight:
                    if losses_dict_eval["fc"].item() < best_fc_loss:
                        best_fc_loss = losses_dict_eval["fc"].item() / args.clip_fc_loss_weight
                        best_iter_fc = epoch
                print(f"eval iter[{epoch}/{args.num_iter}] loss[{loss.item()}] time[{time.time() - start}]")
                
                cur_delta = loss_eval.item() - best_loss
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = epoch
                        terminate = False
                        utils.plot_batch(inputs, sketches, args, counter, use_wandb=args.use_wandb, title="best_iter.jpg")
                        renderer.save_svg(args.output_dir, "best_iter")

                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_loss_fc"] = best_fc_loss
                    wandb_dict = {"delta": cur_delta, "loss_eval": loss_eval.item()}
                    for k in losses_dict_eval.keys():
                        wandb_dict[k + "_eval"] = losses_dict_eval[k].item()
                    wandb.log(wandb_dict, step=counter)
                
                if abs(cur_delta) <= min_delta:
                    if terminate:
                        break
                    terminate = True

                    
        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(), 
                                args.use_wandb, "{}/{}.jpg".format(args.output_dir, "attention_map"), 
                                args.saliency_model, args.display_logs)


        if args.use_wandb:
            wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
            for k in losses_dict.keys():
                wandb_dict[k] = losses_dict[k].item()
            wandb.log(wandb_dict, step=counter)
        
        counter += 1

    renderer.save_svg(args.output_dir, "final_svg")
    path_svg = os.path.join(args.output_dir, "best_iter.svg")
    utils.log_sketch_summary_final(path_svg, args.use_wandb, args.device, best_iter, best_loss, "best total")
    
    return configs_to_save


if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    configs_to_save = main(args)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()
