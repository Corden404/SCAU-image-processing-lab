import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，防止 matplotlib 显示中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 运行前需要确保系统有黑体字体，Windows一般默认有
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def main():
    # 1. 图像的读入 (OpenCV 默认以 BGR 格式读取)
    img_bgr = cv2.imread('cat.jpg')
    if img_bgr is None:
        print("错误: 无法读取图像 'cat.jpg'。请确保图片存在于当前目录下。")
        return
    
    # 将 BGR 转换为 RGB 以便 matplotlib 能够正确显示原图颜色
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. 灰度图转换
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. 灰度反转 (s = 255 - r)
    img_invert = 255 - img_gray

    # 4. 灰度阈值化 (这里使用 Otsu 自动阈值法，也可以手动设置一个值如 127)
    # _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 直方图均衡化
    img_equalized = cv2.equalizeHist(img_gray)

    # 7. 拓展（选做）：尝试将一幅灰度图像转换为伪彩色图像
    img_pseudo_color = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    img_pseudo_color_rgb = cv2.cvtColor(img_pseudo_color, cv2.COLOR_BGR2RGB)

    # 6. 显示直方图对比 及 同时显示以上处理结果
    # 创建一个窗口同时显示多张图片
    plt.figure(figsize=(16, 10))

    # 第一行：显示各个点运算的结果
    plt.subplot(2, 4, 1)
    plt.imshow(img_rgb)
    plt.title('1. 原图 (RGB)')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title('2. 灰度图')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(img_invert, cmap='gray')
    plt.title('3. 灰度反转')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(img_thresh, cmap='gray')
    plt.title('4. 灰度阈值化')
    plt.axis('off')

    # 第二行：直方图均衡化前后的对比
    plt.subplot(2, 4, 5)
    plt.imshow(img_gray, cmap='gray')
    plt.title('均衡化前 灰度图')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.hist(img_gray.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    plt.title('均衡化前 直方图')
    plt.xlim([0, 256])

    plt.subplot(2, 4, 7)
    plt.imshow(img_equalized, cmap='gray')
    plt.title('5. 均衡化后 图像')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.hist(img_equalized.ravel(), 256, [0, 256], color='red', alpha=0.7)
    plt.title('5. 均衡化后 直方图')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.savefig('result_all.png', dpi=300) # 将总结果图保存，方便写实验报告使用
    print("总处理结果图已保存为 'result_all.png'")
    # plt.show()

    # 额外单独显示伪彩色拓展图
    plt.figure(figsize=(6, 5))
    plt.imshow(img_pseudo_color_rgb)
    plt.title('7. 伪彩色图像 (拓展)')
    plt.axis('off')
    plt.savefig('result_pseudo_color.png', dpi=300)
    print("伪彩色拓展图已保存为 'result_pseudo_color.png'")
    # plt.show()

if __name__ == "__main__":
    main()