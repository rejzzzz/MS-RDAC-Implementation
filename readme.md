# 1️⃣ Create a virtual environment (already done)
python -m venv venv

# 2️⃣ Activate the venv
# PowerShell:
.\venv\Scripts\Activate.ps1
# (or CMD: venv\Scripts\activate.bat)

# 3️⃣ Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

# 4️⃣ Install compressai inside the venv
pip install compressai

# 5️⃣ Run the RDAC training script using the venv’s Python
python run.py --config config/train/train_config.yaml --mode train --model_id rdac

# summarize
python scratch\summarize_metrics.py

# test
python run.py --config config\test\test_config.yaml --mode test --model_id rdac --checkpoint log\train_config_03_05_26_15_09_17\00000009-new-checkpoint.pth.tar


# train
python run.py --config config\train\train_config.yaml --mode train --model_id rdac