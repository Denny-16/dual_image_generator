# Dual Image Super Resolution for High-Resolution Optical Satellite Imagery and its Blind Evaluation

## 📌 Problem Statement
Acquiring high-resolution data from satellite platforms is expensive due to resource constraints. Satellites often capture multiple low-resolution images, slightly shifted by half a pixel in both along and across track directions. These low-resolution images are then combined to generate a high-resolution image.

However, assessing the quality of such super-resolved images is challenging in the absence of ground-truth references. This project addresses the **Dual Image Super Resolution** task and integrates **blind (no-reference) quality assessment techniques** to evaluate perceptual realism and fidelity.

---

## 🎯 Objectives
- Generate high-resolution images from two (dual) low-resolution satellite images.
- Evaluate the generated image quality using:
  - **Full-reference metrics** (when GT available)
  - **Blind visual quality assessment techniques** (when GT not available)

---

## 🏆 Expected Outcomes
- Super-resolved images exceeding the resolution of low-resolution inputs.
- A quality evaluation pipeline for blind visual assessment.

---

## 📂 Dataset
- **Input:** Low-resolution satellite images.
- **Ground Truth (optional):** High-resolution satellite image for evaluation.

---

## 🛠️ Tools & Technologies
- **Classical Techniques:**
  - Degradation function estimation
  - Image registration
  - Non-uniform interpolation
- **Deep Learning Techniques:**
  - Data pipeline setup
  - Model selection & fine-tuning
  - Validation & testing
- **Blind Assessment Techniques:**
  - Handcrafted feature-based ML methods
  - Deep feature-based scoring
  - Metrics correlation and sensitivity analysis

---

## 📏 Evaluation Parameters
- **Full-reference metrics:** MSE, RMSE, SSIM, PSNR
- **Blind assessment metrics:** Correlation of predicted scores with full-reference results
- **Robustness testing** under varying image quality levels

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Denny-16/dual_image_generator.git
   cd dual_image_generator

2.Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
3.Place your input images in the data/ folder.

4.Run the model:

bash
Copy
Edit
python main.py
5.View the results in the output/ folder.


