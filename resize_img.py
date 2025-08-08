import cv2, numpy as np, os
from PIL import Image

os.makedirs("data", exist_ok=True)
img = cv2.imread("data/high-res/antarctica.jpg")
img = cv2.resize(img, (128, 128))

lr1 = cv2.GaussianBlur(img, (5,5), 1)
noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
lr2 = cv2.add(img, noise)

cv2.imwrite("data/hr.jpg", img)
cv2.imwrite("data/lr1.jpg", lr1)
cv2.imwrite("data/lr2.jpg", lr2)