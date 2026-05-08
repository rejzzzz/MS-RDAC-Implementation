import os
import shutil
from pathlib import Path
import random

# Update these paths to match your local dataset locations
SOURCE_DIR = r"datasets/raw_videos"
DEST_DIR = r"datasets/prepared"


def prepare_dataset():
    all_videos = list(Path(SOURCE_DIR).rglob("*.mp4"))
    if not all_videos:
        print(f"No .mp4 files found in {SOURCE_DIR}!")
        return

    print(f"Found {len(all_videos)} videos. Preparing dataset...")
    random.shuffle(all_videos)

    split_idx = int(len(all_videos) * 0.95)
    splits = {'train': all_videos[:split_idx], 'test': all_videos[split_idx:]}

    for split_name, videos in splits.items():
        dest = os.path.join(DEST_DIR, split_name)
        os.makedirs(dest, exist_ok=True)
        print(f"\n--- Copying {split_name.title()} Videos ({len(videos)}) ---")
        for idx, vid_path in enumerate(videos):
            if idx % 500 == 0:
                print(f"  {idx}/{len(videos)}...")
            shutil.copy2(vid_path, os.path.join(dest, f"video_{idx:05d}.mp4"))

    print(f"\nDone! Dataset at: {DEST_DIR}")
    print(f"Train: {len(splits['train'])}, Test: {len(splits['test'])}")


if __name__ == "__main__":
    prepare_dataset()
