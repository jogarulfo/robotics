from lerobot.datasets.lerobot_dataset import LeRobotDataset

local_repo_id = "jogarulfo/dataset_MVP_store_cardboard_1"

def push_dataset():
    # 2. Charger le dataset depuis le cache local
    # On initialise le dataset en pointant vers le dossier local
    dataset = LeRobotDataset(local_repo_id)

    # 3. Pousser sur le Hub
    # Assurez-vous d'être connecté via 'huggingface-cli login'
    dataset.push_to_hub()
    

if __name__ == "__main__":
    push_dataset()