## Setting
1. If OpenGL fails when running visualize_joint_adroit.py, run `unset LD_PRELOAD`.

## How to run the code
1. Fine tune the MLP to perform the

## Test and Log process
1. Training paramters mismatch.
   1. Time: 12-10
   2. Phenomina: loss getting smaller, but no retargetting effort -> each image in the batch has the same output during training -> the feature values get to 1e-15 which is not good.
   3. Test: compare the Frankmocap data and the adapted ckpt files we have and the training parameters we are using.
   4. Result: frankmocap and ckpt have the same params for the Conv1 in pose_net as well as when doing the adaptation traning at stage2. But I have different one in training time.
   5. Solution: check the loaded model (conv1 is not requesting grad, confirm) process.
   6. Remain question: why the loss is small, what is it learning?
   7. **Root reason**: I use load_dict() incorrectly: strict=false will skip all the mismatched values. So I used the random initialized values all the time.... Feel so dumb....

2. Fixing the parameters loading and test
   1. Time:12-15
   2. Observations: the position of the nail changes. So I set only to [:24]. The grasping pose is basically Ok, but the position of the hand is not good. More: for the qpos in env_state function, we get dim=33 vector. **0-25 is for joint angles** (0 and 1 are the dof outside the hand part for pitch and yaw, 2-25 are for joint angles) 26 is for nail, 27-32 is 6D pose for axe. So we have 25 dof for the hand??? Verify whether we get the same for the observation function (where I get the training data). 
   3. Small improvement: the image size of input img is 256 while the rest is 224

3. Try SGD:
   1. Time 12-16
   2. Observation: need to change learning rate to a small value. The result is descent. Max error (in training set) often occur in id=25 (thumb twist) with 0.1-0.2 rad (6-12deg) which is small enough for our application. Arg error around 1e-2.
   3. Observation for predict angle in real-time simulation: arg error around 0.1-1. Inspecting the input data, there is color difference between the simulation and training dataset. That's due to the channel changes of RGB in `read_img` function. Problem fixed.

## Ideas dumped:
- [x] try SGD
- [ ] Add temporal information, make the prediction smoother: instead of inputting independent images in batch, input a consequtive image sequence.
- [ ] more complicated network
- [ ] figure out why it is swing around
- [ ] add a video demo
  