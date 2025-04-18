import cv2
import numpy as np


def clean_image(
        image_path,
        threshold_value=127,  # Пороговое значение для бинаризации
        blur_kernel_size=(5, 5),  # Размер фильтра Гаусса для уменьшения шума
        morph_kernel_size=(2, 2),  # Размер структурирующего элемента для морфологии
        min_area_to_keep=100,  # Минимальная площадь области, которую оставляем (количество пикселей)
        close_iterations=1,  # Количество итераций морфологического закрытия
        open_iterations=1,  # Количество итераций открытия
        invert=True  # Инвертировать изображение после обработки
):
    """
    Функция для чистки изображения путем устранения шума и улучшения четкости рисунков.
    :param image_path: путь к изображению
    :param threshold_value: порог бинаризации
    :param blur_kernel_size: размер ядра гауссова размытия
    :param morph_kernel_size: размер структурирующего элемента для морфологии
    :param min_area_to_keep: минимальный размер области (число пикселей), которую оставляем незатронутой
    :param close_iterations: количество итераций морфологического закрытия
    :param open_iterations: количество итераций открытия
    :param invert: инвертировать изображение после обработки
    :return: обработанное изображение
    """
    # Загружаем изображение
    img = cv2.imread(image_path)

    # Размытие изображения для подавления мелких шумов
    blurred_img = cv2.GaussianBlur(img, blur_kernel_size, 0)

    # Преобразуем изображение в оттенки серого
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

    # Бинаризируем изображение
    _, binary_img = cv2.threshold(gray_img, threshold_value, 1023, cv2.THRESH_BINARY_INV)

    # Создаем ядро для морфологической обработки
    kernel = np.ones(morph_kernel_size, np.uint8)

    # Морфологическое закрытие для заполнения разрывов и удаления небольших пятен
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    # Морфологическое открытие для дополнительной фильтрации оставшихся дефектов
    opened_img = cv2.morphologyEx(closed_img, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    # Ищем контуры (связанные области одного цвета)
    contours, hierarchy = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Проходим по каждому контуру и проверяем его площадь
    mask = np.zeros_like(opened_img)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area_to_keep:
            # Оставляем крупные области (узоры)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

    result = cv2.bitwise_and(opened_img, mask)

    if invert:
        # Инвертируем финальное изображение обратно
        final_img = cv2.bitwise_not(result)
    else:
        final_img = result

    return final_img


# Пример использования функции
image_path = r'D:\python\OpenCV\Scripts\ScanerNG\images\cropped_result_BP.png'
cleaned_image = clean_image(image_path,threshold_value=150, min_area_to_keep=5)

# Показываем результат
cv2.imshow('Cleaned Image', cleaned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




