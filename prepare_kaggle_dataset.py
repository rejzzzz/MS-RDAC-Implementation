import os
import shutil
from pathlib import Path
import random

# ==========================================
# CHANGE THIS TO WHERE YOU EXTRACTED KAGGLE
# ==========================================
SOURCE_DIR = r"C:\Users\jayan\OneDrive\Documents\MS\MS-RDAC-Implementation\datasets\files"

# Where the ready-to-train dataset will go
DEST_DIR = r"C:\Users\jayan\OneDrive\Documents\MS\datasets\KaggleFace_Prepared"

def prepare_dataset():
    print("Searching for .mp4 files...")
    
    # Find all .mp4 files in all subdirectories
    all_videos = list(Path(SOURCE_DIR).rglob("*.mp4"))
    
    if not all_videos:
        print(f"No .mp4 files found in {SOURCE_DIR}! Did you extract the dataset correctly?")
        return

    print(f"Found {len(all_videos)} videos. Preparing dataset...")
    
    train_dir = os.path.join(DEST_DIR, "train")
    test_dir = os.path.join(DEST_DIR, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Shuffle videos to get a random mix
    random.shuffle(all_videos)
    
    # Use 95% for training, 5% for testing
    split_idx = int(len(all_videos) * 0.95)
    train_vids = all_videos[:split_idx]
    test_vids = all_videos[split_idx:]
    
    def copy_videos(videos, dest_folder):
        for idx, vid_path in enumerate(videos):
            # Rename them to video_0001.mp4, video_0002.mp4, etc. to avoid duplicate names
            new_name = f"video_{idx:05d}.mp4"
            dest_path = os.path.join(dest_folder, new_name)
            
            # Print progress every 500 videos
            if idx % 500 == 0:
                print(f"Copying {idx}/{len(videos)} to {dest_folder}...")
                
            shutil.copy2(vid_path, dest_path)

    print("\n--- Copying Train Videos ---")
    copy_videos(train_vids, train_dir)
    
    print("\n--- Copying Test Videos ---")
    copy_videos(test_vids, test_dir)
    
    print(f"\nDone! Dataset prepared at: {DEST_DIR}")
    print(f"Train videos: {len(train_vids)}")
    print(f"Test videos: {len(test_vids)}")

if __name__ == "__main__":
    prepare_dataset()
