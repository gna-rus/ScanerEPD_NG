import cv2
import numpy as np
from scipy import ndimage
from collections import Counter


# Установка опций для полного вывода массивов
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def count_lines(matrix):
    # Ищем горизонтальные линии

    horizontal_lines = []

    for i in range(matrix.shape[0]):
        row = matrix[i]
        print(row)
        # Получение уникальных значений и их количеств
        values, counts = np.unique(row, return_counts=True)
        # Преобразование результата в словарь
        result = dict(zip(values, counts)) # список количества в каждой строке

        if result.get(255, 0) != 0:
            if result[255] >= 50:
                horizontal_lines.append(i) # номеров пустых строк


    # Ищем вертикальные линии
    vertical_lines = []
    for j in range(matrix.shape[1]):
        col = matrix[:, j]

        # Получение уникальных значений и их количеств
        values, counts = np.unique(col, return_counts=True)
        # Преобразование результата в словарь
        result = dict(zip(values, counts)) # список количества в каждой строке

        if result.get(255, 0) != 0:
            if result[255] >= 30:
                vertical_lines.append(j) # номеров пустых строк


    horizontal_lines = horizontal_lines[1:-1]
    vertical_lines = vertical_lines[1:-1]
    return {
        'horizontal': [(num+1) for num in horizontal_lines[::2]],
        'vertical': [(num+1) for num in vertical_lines[::2]]
    }



def classify_defects(image_path):
    # Настройки чувствительности обнаружения дефектов
    DIFF_THRESHOLD = 254         # Порог различия для одиночных пикселей
    POINT_EROSION_KERNEL_SIZE = (2, 3)      # Ядро для очищения точечных дефектов
    CLUSTERS_CONNECTIVITY = 8          # Связанность компонентов для кластеров
    MIN_CLUSTER_PIXELS = 10            # Минимальное количество пикселей в кластере
    LINE_DETECTION_KERNEL_WIDTH = 3    # Ширина ядра для линий

    # Загружаем изображение
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Файл {image_path} не найден.')

    # Медианная фильтрация для удаления мелких дефектов
    filtered_img = cv2.medianBlur(img, ksize=3)

    # Абсолютная разница между исходным и фильтрованным изображением
    diff_img = np.abs(filtered_img.astype(int) - img.astype(int))

    # print(diff_img)

    print(count_lines(diff_img))





if __name__ == "__main__":
    classify_defects('./images/MY_1by1Checker.png')  # укажите путь к вашему изображению
