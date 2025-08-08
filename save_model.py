import torch
from rams_sr_net import DualImageRAMSNet

# Replace this path with the actual path to your trained checkpoint if needed
checkpoint_path = "trained_model.pth"  # You may already have this from train_and_eval.py
output_path = "models/model.pth"

# Load the trained model checkpoint (state_dict)
model = DualImageRAMSNet()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

# Save it in the expected format for the Streamlit app
torch.save(model.state_dict(), output_path)
print(f"✅ Model saved to {output_path}")