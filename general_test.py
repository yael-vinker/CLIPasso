import subprocess as sp
import numpy as np
from shutil import copyfile
import os
from torch.nn.parallel import parallel_apply
import multiprocessing as mp

manager = mp.Manager()

sp.run(["pip", "install", "-U", "scikit-learn"])
test_name = "speedtest"

output_dir = f"/datasets/home/vinker/sketches/experiments_res/{test_name}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

num_iter = 2
save_interval = 1
use_wandb = 0

# targets = ["/datasets/home/vinker/data/background/horse/horse_416721310_08ac6a68e8.png",]
targets = ["/datasets/home/vinker/data/unique_classes/ehsan_images/animals_ontop.png"]
# targets = [""]
# # targets = ["/datasets/home/vinker/code/sketch_generator/data/xdog_res/animals/cat_10544.png"]
# target_names = ["horse0"]
target_names = ["animals_ontop"]
# targets = ["/datasets/home/vinker/data/user_study/women/08-14717382472_3119d35971_o_w.png",]
# targets = ["/datasets/home/vinker/code/sketch_generator/data/xdog_res/animals/cat_10544.png"]
# target_names = ["women1"]

wandb_project_name = "sketches-tests"
seeds = [0]  # verify randomness is ok
exit_codes = []
losses_all = manager.dict()
losses_all_fc = manager.dict()
train_with_clip=1
clip_weight=1
clip_conv_loss = 0
# clip_conv_layer_weights = "0,0,1.0,1.0,0"
clip_conv_layer_weights = "0,0,0,0,0"

clip_model_name="RN101"
# clip_model_name="ViT-B/32"
# clip_fc_loss_weight=0.1
clip_fc_loss_weight=0

augment_both=0
use_gpu=1
text_target="a horse"
clip_text_guide=0
augemntations="affine"

saliency_model="clip"
saliency_clip_model="ViT-B/32"
xdog_intersec=1
attention_init=1
percep_loss="none"
perceptual_weight=0
fix_scale=0
mask_object=0
mask_object_attention=0

def run(target, seed, wandb_name):
    exit_codes.append(sp.run(["python", "painterly_rendering.py", target, "--output_dir", output_dir,
                            "--use_wandb", str(use_wandb),  "--wandb_project_name", wandb_project_name, 
                            "--wandb_name", wandb_name, "--num_iter", str(num_iter), 
                            "--save_interval", str(save_interval), "--train_with_clip", str(train_with_clip), 
                            "--clip_weight", str(clip_weight), "--seed", str(seed), "--clip_conv_loss", str(clip_conv_loss),
                            "--clip_conv_layer_weights", clip_conv_layer_weights, "--clip_model_name", clip_model_name,
                            "--clip_fc_loss_weight", str(clip_fc_loss_weight), "--augment_both", str(augment_both),
                            "--use_gpu", str(use_gpu),
                            "--text_target", text_target, "--clip_text_guide", str(clip_text_guide),
                            "--augemntations", augemntations, "--saliency_model", saliency_model,
                            "--xdog_intersec", str(xdog_intersec), "--attention_init", str(attention_init),
                            "--percep_loss", percep_loss, "--saliency_clip_model", saliency_clip_model,
                            "--perceptual_weight", str(perceptual_weight), 
                            "--fix_scale", str(fix_scale), 
                            "--mask_object", str(mask_object),
                            "--mask_object_attention", str(mask_object_attention)]))
    config = np.load(f"{output_dir}/{wandb_name}/config.npy", allow_pickle=True)[()]
    loss_eval = np.array(config['loss_eval'])
    inds = np.argsort(loss_eval)
    losses_all[wandb_name] = loss_eval[inds][0]
    # intervals = np.arange(0, num_iter, save_interval)[inds]
    # copyfile(f"{output_dir}/{wandb_name}/best_iter.svg", f"{output_dir}/{wandb_name}/best_epoch{intervals[0]}.svg")
    # copyfile(f"{output_dir}/{wandb_name}/svg_stage0_iter{intervals[0]}.svg", f"{output_dir}/{wandb_name}/best_epoch.svg")
    # copyfile(f"{output_dir}/{wandb_name}/stage0_iter{intervals[0]}.jpg", f"{output_dir}/{wandb_name}/best_epoch.jpg")

    # if clip_fc_loss_weight:
    #     loss_fc_eval = np.array(config['fc'])
    #     inds = np.argsort(loss_fc_eval)
    #     losses_all_fc[wandb_name] = loss_fc_eval[inds][0] / clip_fc_loss_weight
    #     intervals = np.arange(0, num_iter, save_interval)[inds]
    #     copyfile(f"{output_dir}/{wandb_name}/svg_stage0_iter{intervals[0]}.svg", f"{output_dir}/{wandb_name}/fc_best_epoch{intervals[0]}.svg")
    #     copyfile(f"{output_dir}/{wandb_name}/svg_stage0_iter{intervals[0]}.svg", f"{output_dir}/{wandb_name}/fc_best_epoch.svg")
    #     copyfile(f"{output_dir}/{wandb_name}/stage0_iter{intervals[0]}.jpg", f"{output_dir}/{wandb_name}/fc_best_epoch.jpg")

    # print(losses_all)

save_fc_best = True

ncpus=10
import time
start = time.time()
# P = mp.Pool(ncpus) # Generate pool of workers
for target, target_name in zip(targets, target_names): # Generate processes
    for seed in seeds:
        saliency_clip_model_ = saliency_clip_model.replace("/","")
        wandb_name=f"{saliency_model}_{saliency_clip_model_}_{target_name}_seed{seed}"
        if xdog_intersec and saliency_model == "clip":
            wandb_name=f"{saliency_model}_xdog_{saliency_clip_model_}_{target_name}_seed{seed}"
        # P.apply_async(run,(target, seed, wandb_name)) # run simulation and ISF analysis in each process
        run(target, seed, wandb_name)

# P.close()
# P.join() # start processes 

print("==================")
print("multi process", time.time() - start)
sorted_final = dict(sorted(losses_all.items(), key=lambda item: item[1]))
copyfile(f"{output_dir}/{list(sorted_final.keys())[0]}/best_iter.svg", f"{output_dir}/{list(sorted_final.keys())[0]}_best.svg")
# copyfile(f"{output_dir}/{list(sorted_final.keys())[0]}/best_iter.jpg", f"{output_dir}/{list(sorted_final.keys())[0]}_best.jpg")

# if save_fc_best:
#     sorted_final = dict(sorted(losses_all_fc.items(), key=lambda item: item[1]))
#     copyfile(f"{output_dir}/{list(sorted_final.keys())[0]}/fc_best_epoch.svg", f"{output_dir}/{list(sorted_final.keys())[0]}_best_fc.svg")
#     copyfile(f"{output_dir}/{list(sorted_final.keys())[0]}/fc_best_epoch.jpg", f"{output_dir}/{list(sorted_final.keys())[0]}_best_fc.jpg")

