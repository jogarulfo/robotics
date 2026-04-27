#!/usr/bin/env python3
"""
Add continuous conditioning vectors to jogarulfo/dataset_MVP_store_cardboard.
"""
import json
import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def add_conditioning_from_mapping(repo_id: str, mapping_path: Path):
    # 1. Charger les embeddings de texte (384 dimensions)
    # Assure-toi d'avoir généré ce fichier via le script MiniLM vu précédemment !
    embeddings_path = "/home/josephrigal/workspace/robotics/task_embeddings.pt"
    if not Path(embeddings_path).exists():
        print(f"Erreur : Le fichier {embeddings_path} n'existe pas. Génère-le d'abord !")
        return False
        
    print(f"Loading text embeddings from {embeddings_path}...")
    embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True) # Shape: [8, 384]

    # 2. Charger le mapping JSON (Episode -> ID de la tâche)
    with open(mapping_path) as f:
        mapping = json.load(f)
    mapping_int = {int(k): int(v) for k, v in mapping.items()}

    # 3. Charger le dataset via LeRobot
    print(f"\nLoading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id)
    hf_dataset = dataset.hf_dataset

    # 4. Générer la liste des vecteurs pour chaque image (frame) du dataset
    print("Generating 384-d 'conditioning' vectors for all frames...")
    conditioning_data = []
    for i in range(len(hf_dataset)):
        ep_idx = hf_dataset[i]["episode_index"]
        task_id = mapping_int[ep_idx]
        
        # Extraire le tenseur 384-d et le convertir en liste Python standard
        vector = embeddings[task_id].tolist()
        conditioning_data.append(vector)

    # 5. Ajouter cette liste comme une vraie colonne dans le dataset Hugging Face
    if "conditioning" in hf_dataset.column_names:
        hf_dataset = hf_dataset.remove_columns("conditioning")
    
    hf_dataset = hf_dataset.add_column("conditioning", conditioning_data)

    # 6. SAUVEGARDER SUR LE DISQUE (La partie qui te manquait !)
    dataset_dir = dataset.root
    parquet_path = dataset_dir / "data"
    print(f"\nSaving updated dataset parquets to {parquet_path}...")
    
    # Nettoyer les anciens parquets et sauvegarder la nouvelle version
    for file in parquet_path.glob("*.parquet"):
        file.unlink()
    hf_dataset.to_parquet(parquet_path / "train-00000-of-00001.parquet")

    # 7. Mettre à jour les métadonnées pour prévenir le Dataloader de LeRobot
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
        
    # On déclare officiellement la colonne au système LeRobot
    info["features"]["conditioning"] = {
        "dtype": "float32",
        "shape": [384]
    }
    
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print("\n✓ Conditioning vectors successfully injected and saved to disk!")
    return True

if __name__ == "__main__":
    repo_id = "jogarulfo/dataset_MVP_store_cardboard"
    mapping_file = Path("/home/josephrigal/workspace/robotics/conditioning_map_8class.json")
    
    add_conditioning_from_mapping(repo_id, mapping_file)