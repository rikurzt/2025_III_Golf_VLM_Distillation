import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


class PoseVisualizer:
    def __init__(self):
        self.colors = {
            'left_arm': 'red',
            'right_arm': 'blue',
            'left_leg': 'green',
            'right_leg': 'purple'
        }

    def parse_points(self, points_str):
        """解析點位字串，例如：<points x1="54.1" y1="25.7" x2="54.1" y2="33.3" x3="53.8" y3="43.5" />"""
        pattern = r'x(\d+)="([\d.]+)" y\1="([\d.]+)"'
        matches = re.findall(pattern, points_str)
        points = [(float(x), float(y)) for _, x, y in matches]
        return points

    def get_color_from_question(self, question):
        """根據問題判斷要使用的顏色"""
        question = question.lower()
        if 'left arm' in question:
            return self.colors['left_arm']
        elif 'right arm' in question:
            return self.colors['right_arm']
        elif 'left leg' in question:
            return self.colors['left_leg']
        elif 'right leg' in question:
            return self.colors['right_leg']
        return 'black'  # 預設顏色

    def visualize_pose(self, image, question, points_str):
        """在圖片上繪製關鍵點和連接線"""
        # 將 PIL 圖片轉換為 numpy 陣列
        img_array = np.array(image)

        # 創建新的圖片用於繪製
        plt.figure(figsize=(12, 8))
        plt.imshow(img_array)

        # 解析點位
        points = self.parse_points(points_str)

        # 獲取顏色
        color = self.get_color_from_question(question)

        # 繪製點和線
        x_coords = [p[0] / 100 * image.width for p in points]
        y_coords = [p[1] / 100 * image.height for p in points]

        # 繪製點
        plt.scatter(x_coords, y_coords, c=color, s=100, marker='o')

        # 繪製連接線
        plt.plot(x_coords, y_coords, c=color, linewidth=2)

        # 添加問題文字
        plt.title(question, fontsize=12)

        plt.axis('off')
        plt.show()