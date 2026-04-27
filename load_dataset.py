import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "jogarulfo/dataset_MVP_store_cardboard"

# 1) Load from the Hub (cached locally)
dataset = LeRobotDataset(repo_id)