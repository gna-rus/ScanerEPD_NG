import cv2
import numpy as np
import tkinter as tk
from threading import Thread
from skimage.metrics import structural_similarity as ssim


# Функция подсчета площади квадрата по координатам (принимает кортеж списков)
def calculateTheArea(tupel1):
    list1 = list(tupel1)
    return (list1[1][0] - list1[0][0]) * (list1[3][1] - list1[0][0])


def nothing(*arg):
    pass


def run_tkinter():
    global button_pressed
    root = tk.Tk()
    button = tk.Button(root, text="Сделать фото", command=lambda: set_button_pressed(True))
    button.pack()
    root.mainloop()


def set_button_pressed(value):
    global button_pressed
    button_pressed = value


def rotate_and_crop_image(image, center, angle, width, height, scale=1.0):
    """ Функция для поворота и обрезки изображения. """
    # Получаем размер изображения
    (full_height, full_width) = image.shape[:2] # размеры всей картинки

    # Создаем матрицу поворота
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Выполняем поворот изображения
    rotated_img = cv2.warpAffine(image, M, (full_width, full_height))

    # Вычисляем новые границы после поворота
    new_height, new_width = rotated_img.shape[:2]
    x1 = max(0, int(center[0] - new_width // 2))
    y1 = max(0, int(center[1] - new_height // 2))
    x2 = min(new_width, int(center[0] + new_width // 2))
    y2 = min(new_height, int(center[1] + new_height // 2))

    print(f"x1, x2, y1, y2", x1, x2, y1, y2)
    print('height ', height)
    print('width ', width)
    print('type ', type(rotated_img))

    # Обрезаем изображение по новой рамке
    # cropped_image = rotated_img[y1:y2, x1:x2]

    # Рассчитываем координаты верхнего левого и нижнего правого углов прямоугольника
    top_left_x = int(center[0] - width / 2)
    top_left_y = int(center[1] - height / 2)
    bottom_right_x = int(center[0] + width / 2)
    bottom_right_y = int(center[1] + height / 2)
    cropped_image = rotated_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]


    return cropped_image

def calculate_side_lengths(x1, y1, x2, y2):
    """ Рассчитывает длину и ширину рамки.
    :param x1: Левая граница прямоугольника
    :param y1: Верхняя граница прямоугольника
    :param x2: Правая граница прямоугольника
    :param y2: Нижняя граница прямоугольника
    :return: Длина и ширина рамки """
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width, height

def print_size(name_file, width, height):
    print(f"Размеры рамки для {name_file}: ширина = {width}, высота = {height}")


def calculate_rotation_angle(box):
    """ Рассчитывает угол поворота для выравнивания длинной стороны рамки.
    :param box: Координаты вершин рамки (np.ndarray)
    :return: Угол поворота в градусах """
    # Извлекаем координаты первой и второй точек рамки
    pt1 = box[0]
    pt2 = box[1]

    # Вычисляем вектор между двумя точками
    vector = pt2 - pt1

    # Вычисляем угол вектора относительно оси X
    angle_rad = np.arctan2(vector[1], vector[0])

    # Конвертируем радианы в градусы
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def capture_images(full_contours_frame, full_camera_frame, full_result_frame, angle):
    # Сохраняем полные изображения
    if full_contours_frame is not None and full_contours_frame.size > 0:
        cv2.imwrite('full_contours.png', full_contours_frame)
    else:
        print("Ошибка: полное изображение contours отсутствует или пустое.")

    if full_camera_frame is not None and full_camera_frame.size > 0:
        cv2.imwrite('full_camera.png', full_camera_frame)
    else:
        print("Ошибка: полное изображение camera отсутствует или пустое.")

    if full_result_frame is not None and full_result_frame.size > 0:
        cv2.imwrite('full_result.png', full_result_frame)
    else:
        print("Ошибка: полное изображение result отсутствует или пустое.")

    # Обрезаем изображения по габаритам зеленой рамки
    global x1, x2, y1, y2
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    # angle = 45  # Задаем угол поворота (можно изменять)


    if full_contours_frame is not None and full_contours_frame.size > 0:
        width, height = calculate_side_lengths(x1, y1, x2, y2)
        print_size('cropped_contours.png', width, height)
        cropped_contours_frame = rotate_and_crop_image(full_contours_frame, center, angle, width, height)
        cv2.imwrite('cropped_contours.png', cropped_contours_frame)

    else:
        print("Ошибка: обрезанное изображение contours отсутствует или пустое.")

    if full_camera_frame is not None and full_camera_frame.size > 0:
        width, height = calculate_side_lengths(x1, y1, x2, y2)
        print_size('cropped_camera.png', width, height)
        cropped_camera_frame = rotate_and_crop_image(full_camera_frame, center, angle, width, height)
        cv2.imwrite('cropped_camera.png', cropped_camera_frame)


    else:
        print("Ошибка: обрезанное изображение camera отсутствует или пустое.")

    if full_result_frame is not None and full_result_frame.size > 0:
        width, height = calculate_side_lengths(x1, y1, x2, y2)
        print_size('cropped_result.png', width, height)
        cropped_result_frame = rotate_and_crop_image(full_result_frame, center, angle, width, height)
        cv2.imwrite('cropped_result.png', cropped_result_frame)

    else:
        print("Ошибка: обрезанное изображение result отсутствует или пустое.")

    print("Шесть фотографий сделаны и сохранены.")


# Основная функция программы
def main():
    global button_pressed, x1, x2, y1, y2
    button_pressed = False
    x1, x2, y1, y2 = 0, 0, 0, 0  # Координаты для обрезки по рамке картинки

    cv2.namedWindow("result")  # Создаем главное окно
    cv2.namedWindow("settings")  # Создаем окно настроек
    cv2.namedWindow("camera")
    cap = cv2.VideoCapture(0)  # Подключаемся к видео камере, передаем в методе индекс веб-камеры
    # ######
    cap.set(cv2.CAP_PROP_FPS, 12)  # Частота кадров
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Ширина кадров в видеопотоке (в первую очередь этот параметр влияет на размер).
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Высота кадров в видеопотоке.
    # ######

    # Создаем 6 бегунков для настройки начального и конечного цвета фильтра
    # createTrackbar ('Имя', 'Имя окна', 'начальное значение','максимальное значение','вызов функции при изменении бегунка'
    cv2.createTrackbar('hue_1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('satur_1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('value_1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('hue_2', 'settings', 213, 255, nothing)
    cv2.createTrackbar('satur_2', 'settings', 240, 255, nothing)
    cv2.createTrackbar('value_2', 'settings', 73, 255, nothing)
    cv2.createTrackbar('Area', 'settings', 89000, 120000, nothing)
    ####

    while True:
        ret, img = cap.read()  # img - сама картинка с камеры, ret - флаг
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV формат изображения

        # Считывание значений бегунков
        h1 = cv2.getTrackbarPos('hue_1', 'settings')
        s1 = cv2.getTrackbarPos('satur_1', 'settings')
        v1 = cv2.getTrackbarPos('value_1', 'settings')
        h2 = cv2.getTrackbarPos('hue_2', 'settings')
        s2 = cv2.getTrackbarPos('satur_2', 'settings')
        v2 = cv2.getTrackbarPos('value_2', 'settings')
        Ar = cv2.getTrackbarPos('Area', 'settings')

        # Формируем начальный и конечный цвет фильтра
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)

        # Накладываем фильтр на кадр в модели HSV
        thresh = cv2.inRange(hsv, h_min, h_max)

        # Ищем контуры и складируем их в переменную contours
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img2 = img.copy()

        # Отображаем контуры поверх изображения
        cv2.drawContours(img, contours, -1, (255, 0, 0), 2, cv2.LINE_4, hierarchy, 2)
        cv2.imshow("camera", img)  # Показывает все контуры

        # Алгоритм поиска и рисования прямоугольника
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            tup = tuple(box.tolist())
            area = calculateTheArea(tup)

            if area >= Ar:
                cv2.drawContours(img2, [box], -1, (0, 255, 0), 2)
                # global x1, x2, y1, y2
                x1 = int(tup[0][0])
                x2 = int(tup[2][0])
                y1 = int(tup[0][1])
                y2 = int(tup[2][1])

                angle = calculate_rotation_angle(box)

                print('Поворот: ', angle)

                cv2.imshow('contours', img2) ##

        cv2.imshow('result', thresh)

        if cv2.waitKey(10) == 32:  # Клавиша Пробел
            print("________", x1, x2, y1, y2)
            img2 = img2[y1:y2, x1:x2]
            cv2.imwrite('cam.png', img2)

        if button_pressed:
            # Формирование координат зеленой рамки
            capture_images(img2, img, thresh, angle )
            button_pressed = False

        if cv2.waitKey(10) == 27:  # Клавиша Esc
            break

    cap.release()
    cv2.destroyAllWindows()

    if img_with_object is not None:
        reference_image = cv2.imread('reference_image.png')  # Загружаем эталонное изображение
        score, _ = compare_images(img_with_object, reference_image)
        if score > 0.9:  # Устанавливаем порог сходства
            print("Объекты совпадают.")
        else:
            print("Объекты отличаются.")


if __name__ == "__main__":
    thread = Thread(target=run_tkinter)
    thread.start()
    main()
