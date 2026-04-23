#### Configure the motors :
 
lerobot-find-port

#### Set-up motors :

lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem585A0076841  # <- paste here the port found at previous step

lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0  


#### Calibration :

lerobot-calibrate   --robot.type=so101_follower   --robot.port=/dev/ttyACM0   --robot.id=jo_follow

lerobot-calibrate    --teleop.type=so101_leader   --teleop.port=/dev/ttyACM1  --teleop.id=jo_lead

#### Teleoperation :

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=jo_follow \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=jo_lead

#### Setup Camera :


lerobot-find-cameras opencv

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=jo_follow \
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=jo_lead \
    --display_data=true

--------------------
Camera #2:
  Name: OpenCV Camera @ /dev/video4
  Type: OpenCV
  Id: /dev/video4
  Backend api: V4L2
  Default stream profile:
    Format: 0.0
    Fourcc: YUYV
    Width: 640
    Height: 480
    Fps: 30.0
--------------------

#### Use Writting Token to push to Hugging Face Hub:



HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER


#### record Dataset :
```bash
lerobot-record   --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=jo_follow \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=jo_lead \
    --display_data=false \
    --dataset.repo_id=jogarulfo/dataset_MVP_store_cardboard_  \
    --dataset.num_episodes=8  \
    --dataset.single_task="Grab the cube and store it in the box" \
    --dataset.episode_time_s=90 \
    --dataset.push_to_hub=false \
    --resume=false
```

delete a dataset locally   :
rm -r ~/.cache/huggingface/lerobot/jogarulfo/dataset_MVP_store_cardboard_8


lerobot-record       --robot.type=so101_follower     --robot.port=/dev/ttyACM0     --robot.id=jo_follow     --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30} }"     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM1     --teleop.id=jo_lead     --display_data=false   --dataset.repo_id=jogarulfo/dataset_cellule_boxv2   --dataset.num_episodes=20  --dataset.single_task="Grab the cell and put it in the box" --dataset.episode_time_s=90 --dataset.push_to_hub=false --resume=false


lerobot-edit-dataset \
    --repo_id jogarulfo/dataset_MVP_store_cardboard \
    --operation.type merge \
    --operation.repo_ids "['jogarulfo/dataset_MVP_store_cardboard_3', 'jogarulfo/dataset_MVP_store_cardboard_4', 'jogarulfo/dataset_MVP_store_cardboard_5', 'jogarulfo/dataset_MVP_store_cardboard_6', 'jogarulfo/dataset_MVP_store_cardboard_7', 'jogarulfo/dataset_MVP_store_cardboard_8', 'jogarulfo/dataset_MVP_store_cardboard_9', 'jogarulfo/dataset_MVP_store_cardboard_10']" \
    --push_to_hub true



lerobot-train \
  --dataset.repo_id=jogarulfo/dataset_cellule_ \
  --policy.type=act \
  --output_dir=outputs/train/act_battery_cell \
  --job_name=act_battery_cell \
  --device=cuda \
  --batch_size=64 \
  --wandb.enable=true \
  --wandb.project=act_battery_cell


Evaluation of my model :

```bash
lerobot-record   --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=jo_follow \
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30} }" \
    --display_data=true \
    --dataset.repo_id=jogarulfo/dataset_MVP_store_cardboard_1  \
    --dataset.single_task="Grab the cube and store it in the box" \
    --dataset.reset_time=8
```

rm -r ~/.cache/huggingface/lerobot/jogarulfo/dataset_MVP_store_cardboard_1

```bash
HF_HUB_OFFLINE=1 lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=jo_follow \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30} }" \
  --display_data=true \
  --dataset.repo_id=jogarulfo/eval_dataset_cellule_ \
  --dataset.single_task="Grab the cell and put it in the hole" \
  --policy.type=act \
  --policy.pretrained_path=jogarulfo/act_battery_cell \
  --dataset.reset_time=10
```