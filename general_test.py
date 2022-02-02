import subprocess as sp
import numpy as np
from shutil import copyfile
import os
from torch.nn.parallel import parallel_apply
import multiprocessing as mp

manager = mp.Manager()
exit_codes = []
losses_all = manager.dict()

sp.run(["pip", "install", "-U", "scikit-learn"])

test_name = "camel"
output_dir = f"output_sketches/{test_name}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

targets = ["target_images/camel.png"]
target_names = ["camel"]
num_iter = 2
save_interval = 1
use_gpu=1
seeds = [0]

# if you need to mask the input image and pad the aspect ratio
fix_scale=0
mask_object=0
mask_object_attention=0


def run(target, seed, wandb_name):
    exit_codes.append(sp.run(["python", "painterly_rendering.py", target, 
                                "--output_dir", output_dir,
                                "--wandb_name", wandb_name,
                                "--num_iter", str(num_iter),
                                "--save_interval", str(save_interval),
                                "--seed", str(seed),
                                "--use_gpu", str(use_gpu),
                                "--fix_scale", str(fix_scale), 
                                "--mask_object", str(mask_object),
                                "--mask_object_attention", str(mask_object_attention)]))
    config = np.load(f"{output_dir}/{wandb_name}/config.npy", allow_pickle=True)[()]
    loss_eval = np.array(config['loss_eval'])
    inds = np.argsort(loss_eval)
    losses_all[wandb_name] = loss_eval[inds][0]


ncpus=10
import time
start = time.time()
# P = mp.Pool(ncpus) # Generate pool of workers
for target, target_name in zip(targets, target_names): # Generate processes
    for seed in seeds:
        wandb_name=f"{target_name}_seed{seed}"
        # P.apply_async(run,(target, seed, wandb_name)) # run simulation and ISF analysis in each process
        run(target, seed, wandb_name)

# P.close()
# P.join() # start processes 

print("==================")
print("multi process", time.time() - start)
sorted_final = dict(sorted(losses_all.items(), key=lambda item: item[1]))
copyfile(f"{output_dir}/{list(sorted_final.keys())[0]}/best_iter.svg", f"{output_dir}/{list(sorted_final.keys())[0]}_best.svg")
