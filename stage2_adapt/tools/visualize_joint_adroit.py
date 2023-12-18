# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates
import argparse
import mj_envs
import click
import gym
from pathlib import Path
import pickle
home = str(Path.home())
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
import mjrl
from mjrl.policies import *
import numpy as np
import os
import rrl
import cv2
from PIL import Image
from natsort import natsorted
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import FinetuneUnsupLoss
from core.function import test_time_training
from utils.utils import get_optimizer

import dataset
import models
"""
Test and visulzie the trained policy for joint angle mapping
NOTE: unset LD_PRELOAD if having open-GL issues
Input: dataset path (with image and GT joint angle), model_path, env_name
Output: 
	1. Joint angle prediction and comparison.
	2. Image prediction and comparison (set the pred_joint angle in simulation and compare)
"""

_mj_envs = {'pen-v0', 'hammer-v0', 'door-v0', 'relocate-v0', 'tools-v0'}
_mjrl_envs = {'mjrl_peg_insertion-v0', 'mjrl_reacher_7dof-v0'}
DESC = '''
Helper script to create demos.\n
USAGE:\n
    Create demos on the env\n
    $ \n
'''
seed = 123
data_folder = "0" # TODO: add to parameters later

def render_obs(env, img_size=224, camera_name="vil_camera", device=0):
    # img = env.env.sim.render(width=img_size, height=img_size, \
	# 		mode='window', camera_name=camera_name, device_id=device) # offscreen
    img =  env.env.sim.render(width=img_size, height=img_size, \
			mode='offscreen', camera_name=camera_name, device_id=device)
    img = img[::-1, :, : ] # Image given has to be flipped
    return img

def read_img(data_numpy, img_path=None, transform=None, color_rgb=True):
    if img_path is not None:
        data_numpy = cv2.imread(
            img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if color_rgb:
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
    if data_numpy is None:
        raise ValueError('Fail to read {}'.format(img_path))
    img = Image.fromarray(data_numpy)
    if transform is not None:
        img = transform(img)
    return img

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--keypointnet_pretrain',
                        type=str,
                        default='none')

    parser.add_argument('--freeze_bn',
                        default=0,
                        type=int)

    parser.add_argument('--freeze_conv',
                        default=0,
                        type=int)

    parser.add_argument('--task_name',
                        type=str,
                        default='none')
    
    parser.add_argument('--use_entire_pretrain', default=0, type=int)
    parser.add_argument('--freeze_encoder', default=0, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--test_data', default="training_data", type=str)
    
    # wandb
    parser.add_argument('--use_wandb', default=0, type=int)
    parser.add_argument('--wandb_project', default='handae', type=str)
    parser.add_argument('--wandb_group', default='debug', type=str)
    parser.add_argument('--wandb_name', default='0', type=str)
	
	# policy load
    parser.add_argument('--policy', type=str, help='Location to the policy', required=True)
    parser.add_argument('--mode', type=str, help='Mode : evaluation, exploration', default="exploration")
    parser.add_argument('--img_size', type=int, help='Image size', default=256)
    parser.add_argument('--camera_name', type=str, help='Camera name', default="vil_camera")
    parser.add_argument('--gpu_id', type=int, help='GPU ID', default=0)
    parser.add_argument("--data_dir", type=str, help="Directory to save data", required=True)
    args = parser.parse_args()

    return args

# @click.command(help=DESC)
# @click.option('--data_dir', type=str, help='Directory to save data', required=True)
# @click.option('--env_name', type=str, help='environment to load', required= True)
# @click.option('--num_demos', type=int, help='Number of demonstrations', default=25)
# @click.option('--mode', type=str, help='Mode : evaluation, exploration', default="exploration")
# @click.option('--policy', type=str, help='Location to the policy', required=True)
# @click.option('--img_size', type=int, help='Image size', default=256)
# @click.option('--camera_name', type=str, help='Camera name', default="vil_camera")
# @click.option('--gpu_id', type=int, help='GPU ID', default=0)
# def main(data_dir, env_name, num_demos, mode, policy, img_size, camera_name, gpu_id):
	# parser to load the config file
	# parser = argparse.ArgumentParser(description='Train keypoints network')
	# parser.add_argument('--cfg', default="experiments/ttp.yaml")
def main():
    args = parse_args()
    update_config(cfg, args)
    
    data_dir = os.path.join(cfg.DATASET.ROOT, args.task_name)
    data_dir = os.path.join(data_dir, data_folder)
    print("Data Directory : ", data_dir)

    tgt_joint = pickle.load(open(os.path.join(data_dir, "obs.pkl"), 'rb'))

    resized_center_crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    transform = transforms.Compose([resized_center_crop, transforms.ToTensor()])
    
	# Build and load the model
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True, is_finetune=False, freeze_bn=False, freeze_encoder=False, joint_pred=True
    )
    try:
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True) # NOTE: load the original model
    except:
        # Poor man option for one GPU
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    model.cuda()
    model.eval()

    # Optional debug, read the input data, setup the env setting and dir
    obs_train = pickle.load(open(os.path.join(data_dir, "obs.pkl"), 'rb'))
    mode, img_size, camera_name, gpu_id = args.mode, args.img_size, args.camera_name, args.gpu_id	
    video_dir = os.path.join(data_dir, "../../hammer_visualize_fcn1216_sgd_sim") # TODO: change the name
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # setup the simulation environment
    e = GymEnv(args.task_name)
    e.set_seed(seed)
    done = False
    obs = e.reset()
    pi = pickle.load(open(args.policy, 'rb'))

    if args.test_data == "training_data":
        ######### read and test the files in the dataset ###########
        img_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        img_files = natsorted(img_files)
        for id in range(len(img_files)):
            # Read images from dataset which is not necessary
            if img_files[id].endswith(".pkl"):
                continue
            img = read_img(None, img_path=img_files[id], transform=transform, color_rgb=True)
            img = img.cuda()
            # plt.figure()
            # plt.imshow(img.permute(1,2,0).detach().cpu().numpy())
            # plt.show()
            img = torch.stack([img, img], dim=0)
            pred_images, pose_unsup, pose_sup, joint_pred = model(img.unsqueeze(0))
            joint_pred = joint_pred.to('cpu').detach().numpy()
            error = np.abs(joint_pred - tgt_joint[id][:26])

            # compare input images, sim image from ground truth joint angle, sim image from predicted joint angle
            old_state_dict = e.get_env_state()
            new_state_dict = copy.deepcopy(old_state_dict)
            new_state_dict['qpos'][:26] = joint_pred[0,:26] # TODO: test why the nails position is changed

            # test the id of the joints
            # new_state_dict['qpos'][:] = 0.0
            # new_state_dict['qpos'][25] = 0.1*id # id 26 is the nail position (1D), 27-32 is axe 6-D pos
            e.set_env_state(new_state_dict)
            img_pred = render_obs(e, img_size=img_size, camera_name=camera_name, device=gpu_id)
            # cv2.imwrite(os.path.join(video_dir, str(id) + ".png"), img_pred)
            
            new_state_dict['qpos'][:26] = tgt_joint[id][:26]
            e.set_env_state(new_state_dict)
            img_gt = render_obs(e, img_size=img_size, camera_name=camera_name, device=gpu_id)
		    # convert to BGR
            img = img[1,...].permute(1,2,0)*255 # grab the second target image
            img = img.to(torch.uint8).to('cpu').detach().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)
            
            plt.subplot(1,3,1)
            plt.imshow(img)
            plt.title("Input img")
            plt.subplot(1,3,2)
            plt.imshow(img_gt)
            plt.title("GT img")
            plt.subplot(1,3,3)
            plt.imshow(img_pred)
            plt.title("Pred img")
            plt.tight_layout()
            plt.savefig(os.path.join(video_dir, str(id) + ".png"))

            print("Error joint: ", error.mean(), error.max(), error.min())
    
    else:
        ################ simulation test #########################
        img_list, img_rebuild = [], []
        step = 0
        while not done:
            # TODO: add mode
            action = pi.get_action(obs)[0] if mode == 'exploration' else pi.get_action(obs)[1]['evaluation']
            next_obs, reward, done, info = e.step(action)
            obs = next_obs
            
            img_obs = render_obs(e, img_size=img_size, camera_name=camera_name, device=gpu_id)
            img_list.append(img_obs)
            
            # Get joint angle and compare reconstruct images
            img = read_img(img_obs, transform=transform, color_rgb=False)
            # plt.figure()
            # plt.subplot(2,2,1)
            # plt.imshow(img.permute(1,2,0).detach().cpu().numpy())
            # plt.show()
            # plt.title("Input img")
            # plt.subplot(2,2,2)
            # plt.imshow(img_obs)
            # plt.title("Input obs")

            # test the input images
            # dirr = os.path.join(data_dir, "test.png")
            # tt = img.permute(1,2,0)*255
            # cv2.imwrite(dirr, cv2.cvtColor(tt.to(torch.uint8).to('cpu').detach().numpy(), cv2.COLOR_RGB2BGR))
            img_stack = torch.stack([img, img], dim=0).to('cuda')

            pred_images, pose_unsup, pose_sup, joint_pred = model(img_stack.unsqueeze(0))
            old_state_dict = e.get_env_state()
            new_state_dict = copy.deepcopy(old_state_dict)
            new_state_dict['qpos'][:26] = joint_pred.to('cpu').detach().numpy()
            # new_state_dict['qpos'][:27] = obs_train[step][:27] # TODO: test input

            e.set_env_state(new_state_dict)
            img_reb = render_obs(e, img_size=img_size, camera_name=camera_name, device=gpu_id)

            # test the dataset
            # img_reb_input = read_img(img_reb, transform=transform, color_rgb=True)
            # img_input = torch.stack([img_reb_input, img_reb_input], dim=0).to('cuda')
            # pred_images, pose_unsup, pose_sup, joint_pred = model(img_input.unsqueeze(0))
            # cv2.imwrite(dirr, cv2.cvtColor(img_reb, cv2.COLOR_RGB2BGR)) # NOTE: test input
            tgt_joint = old_state_dict['qpos'][:26]
            error = np.abs(joint_pred.to('cpu').detach().numpy() - tgt_joint)
            print("Error joint: ", error.mean(), error.max(), error.min())

            img_rebuild.append(img_reb)
            e.set_env_state(old_state_dict) # set to original true state and continue

            step += 1

            # save the images
            plt.subplot(1,2,1)
            img = img.permute(1,2,0).to('cpu').detach().numpy()
            plt.imshow(img)
            plt.title("gt img")
            plt.subplot(1,2,2)
            plt.imshow(img_reb)
            plt.title("pred img")
            plt.tight_layout()
            plt.savefig(os.path.join(video_dir, str(step) + ".png"))
                 
        # for img_id in range(len(img_list)):
        #     img = img_list[img_id]
        #     img_rec = img_rebuild[img_id]
        #     img_path = os.path.join(video_dir, str(img_id) + ".png")
        #     img_rec_path = os.path.join(video_dir, str(img_id) + "_rec.png")
    	# 	# convert to BGR
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     img_rec = cv2.cvtColor(img_rec, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(img_path, img)
        #     cv2.imwrite(img_rec_path, img_rec)

	# num_demos = 1
	# for data_id in range(num_demos):
	# 	img_list = []
	# 	obs_list = []
	# 	obs = e.reset()

	# 	done = False
	# 	new_path = {}
	# 	ep_reward = 0
	# 	step = 0

	# 	video_dir = os.path.join(data_dir, str(data_id))
	# 	if not os.path.exists(video_dir):
	# 		os.makedirs(video_dir)

	# 	for img_id in range(len(img_list)):
	# 		img = img_list[img_id]
	# 		img_path = os.path.join(video_dir, str(img_id) + ".png")
	# 		# convert to BGR
	# 		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	# 		cv2.imwrite(img_path, img)

	# 	obs_path = os.path.join(video_dir, "obs.pkl")
	# 	pickle.dump(obs_list, open(obs_path, 'wb'))
	# 	print("Dumping video demos at : ", video_dir)

if __name__ == "__main__":
	main()
