import cv2
import numpy as np
import matplotlib.pyplot as plt

# 确保 matplotlib 支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取原图
image_path = 'view.jpg'
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Could not read view.jpg. Please ensure the image exists.")

# 2. 准备阶段：增加椒盐噪声
def add_salt_and_pepper_noise(image, amount=0.04):
    s_vs_p = 0.5
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out

noisy_img = add_salt_and_pepper_noise(img)
cv2.imwrite('view_noisy.jpg', noisy_img)
noisy_img_rgb = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)

# 3. 图像平滑 (Image Smoothing)
# 均值滤波
mean_filtered = cv2.blur(noisy_img_rgb, (5, 5))
# 中值滤波
median_filtered = cv2.medianBlur(noisy_img_rgb, 5)
# 高斯滤波
gaussian_filtered = cv2.GaussianBlur(noisy_img_rgb, (5, 5), 0)

# 4. 结果展示
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(noisy_img_rgb), plt.title('带噪图像')
plt.subplot(2, 2, 2), plt.imshow(mean_filtered), plt.title('均值滤波')
plt.subplot(2, 2, 3), plt.imshow(median_filtered), plt.title('中值滤波')
plt.subplot(2, 2, 4), plt.imshow(gaussian_filtered), plt.title('高斯滤波')
plt.tight_layout()
plt.savefig('smoothing_results.jpg')
plt.show()

print("Task 2.1: Image Smoothing completed.")
