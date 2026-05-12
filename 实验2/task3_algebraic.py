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

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. 图像代数运算 (Algebraic Operations)
# 使用原图和其水平翻转的图像进行代数运算
img1 = img_rgb
img2 = cv2.flip(img_rgb, 1)

# 加法
add_img = cv2.add(img1, img2)
# 减法
sub_img = cv2.subtract(img1, img2)

# 乘法 (利用 float32 运算后缩放以防溢出)
img1_f = img1.astype(np.float32) / 255.0
img2_f = img2.astype(np.float32) / 255.0
mul_img = cv2.multiply(img1_f, img2_f) * 255.0
mul_img = mul_img.astype(np.uint8)

# 除法 (加小量避免除以0)
div_img = cv2.divide(img1_f, img2_f + 1e-5) * 64.0 # 缩放以便于查看效果
div_img = np.clip(div_img, 0, 255).astype(np.uint8)

# 3. 结果展示
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1), plt.imshow(img1), plt.title('图像1 (原图)')
plt.subplot(2, 3, 2), plt.imshow(img2), plt.title('图像2 (水平翻转)')
plt.subplot(2, 3, 3), plt.imshow(add_img), plt.title('加法')
plt.subplot(2, 3, 4), plt.imshow(sub_img), plt.title('减法')
plt.subplot(2, 3, 5), plt.imshow(mul_img), plt.title('乘法')
plt.subplot(2, 3, 6), plt.imshow(div_img), plt.title('除法')
plt.tight_layout()
plt.savefig('algebra_results.jpg')
plt.show()

print("Task 2.3: Image Algebraic Operations completed.")
