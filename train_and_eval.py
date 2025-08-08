import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from rams_sr_net import DualImageRAMSNet  # Use your RAMS model

# ---- Load input images ----
hr = ToTensor()(Image.open("data/hr.jpg")).unsqueeze(0)
lr1 = ToTensor()(Image.open("data/lr1.jpg")).unsqueeze(0)
lr2 = ToTensor()(Image.open("data/lr2.jpg")).unsqueeze(0)

# ---- Initialize model ----
model = DualImageRAMSNet()
loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# ---- Train loop ----
for epoch in range(1000):
    pred = model(lr1, lr2)
    loss = loss_fn(pred, hr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ---- Save predicted image ----
to_pil = ToPILImage()
output = model(lr1, lr2).detach().squeeze().clamp(0, 1)
predicted_img = to_pil(output)
predicted_path = "data/predicted.jpg"
predicted_img.save(predicted_path)
print(f"✅ Saved predicted image to {predicted_path}")

# ---- Save model weights ----
torch.save(model.state_dict(), "models/model.pth")
print("✅ Model weights saved to models/model.pth")

# ---- Evaluation ----
print("\n📊 Evaluating...")

hr_img = cv2.imread("data/hr.jpg")
sr_img = cv2.imread(predicted_path)

hr_img = cv2.resize(hr_img, (sr_img.shape[1], sr_img.shape[0]))
hr_rgb = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
sr_rgb = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)

psnr = peak_signal_noise_ratio(hr_rgb, sr_rgb)
ssim = structural_similarity(hr_rgb, sr_rgb, win_size=7, channel_axis=2)

print(f"🔹 PSNR: {psnr:.2f} dB")
print(f"🔹 SSIM: {ssim:.4f}")
