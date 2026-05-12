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

# 2. 准备阶段：增加模糊 (均值模糊)
blurred_img = cv2.blur(img, (5, 5))
cv2.imwrite('view_blurred.jpg', blurred_img)
blurred_img_gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

# 3. 图像锐化 (Image Sharpening)
# Prewitt算子
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
img_prewittx = cv2.filter2D(blurred_img_gray, -1, kernelx)
img_prewitty = cv2.filter2D(blurred_img_gray, -1, kernely)
prewitt = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)

# Sobel算子
sobelx = cv2.Sobel(blurred_img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred_img_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(np.absolute(sobel))

# Laplacian算子
laplacian = cv2.Laplacian(blurred_img_gray, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# Canny算子
canny = cv2.Canny(blurred_img_gray, 30, 100)

# 4. 结果展示
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1), plt.imshow(blurred_img_gray, cmap='gray'), plt.title('模糊图像(灰度)')
plt.subplot(2, 3, 2), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt算子')
plt.subplot(2, 3, 3), plt.imshow(sobel, cmap='gray'), plt.title('Sobel算子')
plt.subplot(2, 3, 4), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian算子')
plt.subplot(2, 3, 5), plt.imshow(canny, cmap='gray'), plt.title('Canny边缘检测')
plt.tight_layout()
plt.savefig('sharpening_results.jpg')
plt.show()

print("Task 2.2: Image Sharpening completed.")
