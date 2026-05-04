import imageio.v3 as iio
import os

path = "datasets/train/00.mp4" # Just guess a file, or let's find an actual file
print("Looking for mp4 files...")
for d in ['datasets/train', 'datasets/test']:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith('.mp4'):
                p = os.path.join(d, f)
                props = iio.improps(p, plugin='pyav')
                print(f"{p}: shape={props.shape}")
                
                # Check how fast it is to read an index
                import time
                t0 = time.time()
                frame = iio.imread(p, plugin='pyav', index=10)
                t1 = time.time()
                print(f"Read frame 10 in {t1-t0:.4f}s")
                break
        break
