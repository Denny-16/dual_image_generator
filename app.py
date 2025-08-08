import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from rams_sr_net import DualImageRAMSNet
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms import ToTensor, ToPILImage

# Set page config
st.set_page_config(page_title="Dual-Image SR using RAMS", layout="centered")
st.title("🛰️ Dual-Image Super Resolution")
st.markdown("Upload two low-resolution satellite images to generate a super-resolved output.")

# Load model
@st.cache_resource
def load_model():
    model = DualImageRAMSNet()
    model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
to_tensor = ToTensor()
to_pil = ToPILImage()

# Upload images
col1, col2 = st.columns(2)
with col1:
    uploaded_file1 = st.file_uploader("Upload LR Image 1", type=["jpg", "png", "jpeg"], key="lr1")
with col2:
    uploaded_file2 = st.file_uploader("Upload LR Image 2", type=["jpg", "png", "jpeg"], key="lr2")

# Process and predict
if uploaded_file1 and uploaded_file2:
    img1 = Image.open(uploaded_file1).convert("RGB")
    img2 = Image.open(uploaded_file2).convert("RGB")

    # Resize for consistency
    img1 = img1.resize((128, 128))
    img2 = img2.resize((128, 128))

    tensor1 = to_tensor(img1).unsqueeze(0)
    tensor2 = to_tensor(img2).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor1, tensor2).squeeze().clamp(0, 1)

    pred_img = to_pil(output)

    # Show images
    st.subheader("🔍 Results")
    col1, col2, col3 = st.columns(3)
    col1.image(img1, caption="Low-Res Image 1", use_container_width=True)
    col2.image(img2, caption="Low-Res Image 2", use_container_width=True)
    col3.image(pred_img, caption="Predicted SR Output", use_container_width=True)


    # Compute metrics (against img1 as placeholder reference)
    hr = np.array(img1.resize((128, 128)))
    sr = np.array(pred_img)

    psnr = peak_signal_noise_ratio(hr, sr)
    ssim = structural_similarity(hr, sr, channel_axis=2, win_size=7)

    st.markdown(f"**📈 PSNR**: {psnr:.2f} dB")
    st.markdown(f"**📈 SSIM**: {ssim:.4f}")

else:
    st.warning("Please upload both low-resolution images to proceed.")
