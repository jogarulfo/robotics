from lerobot.datasets.lerobot_dataset import LeRobotDataset

print("Initializing datasets one by one to find the traitor...\n")

broken_datasets = []

for i in range(1, 11):
    repo_id = f"jogarulfo/dataset_MVP_store_cardboard_{i}"
    print(f"--- Loading {repo_id} ---")
    try:
        # We load it exactly how the merge script does
        dataset = LeRobotDataset(repo_id)
        print(f"  -> OK! Identified cameras: {dataset.meta.camera_keys}")
    except Exception as e:
        # If it crashes, we catch it and print the exact error
        print(f"  -> CRASHED: {type(e).__name__}: {e}")
        broken_datasets.append(repo_id)

print(f"\n--- SUMMARY ---")
print(f"Successfully loaded {10 - len(broken_datasets)}/10 datasets.")
if broken_datasets:
    print("These datasets are throwing errors:")
    for bd in broken_datasets:
        print(f" - {bd}")