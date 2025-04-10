import cv2
import numpy as np
import tkinter as tk
from threading import Thread
from skimage.metrics import structural_similarity as ssim
import math


# Глобальная переменная для управления остановкой процесса обнаружения объектов
stop_refreshing_contours = False

# Функция подсчета площади квадрата по координатам (принимает кортеж списков)
def calculateTheArea(tupel1):
    list1 = list(tupel1)
    return (list1[1][0] - list1[0][0]) * (list1[3][1] - list1[0][0])


def nothing(*arg):
    pass


def run_tkinter():
    global button_pressed
    root = tk.Tk()
    # button = tk.Button(root, text="Сделать фото", command=lambda: set_button_pressed(True))
    # button.pack()
    root.mainloop()


def set_button_pressed(value):
    global button_pressed
    button_pressed = value


def rotate_and_crop_image(image, center, angle, width, height, scale=1.0):
    """ Функция для поворота и обрезки изображения. """

    global x1, x2, y1, y2
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
    # if width > height:
    #     height, width = width, height

    # Рассчитываем координаты верхнего левого и нижнего правого углов прямоугольника
    top_left_x = int(center[0] - width / 2)
    top_left_y = int(center[1] - height / 2)
    bottom_right_x = int(center[0] + width / 2)
    bottom_right_y = int(center[1] + height / 2)
    cropped_image = rotated_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    print(image)


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
        cv2.imwrite('./images/full_contours.png', full_contours_frame)
    else:
        print("Ошибка: полное изображение contours отсутствует или пустое.")

    if full_camera_frame is not None and full_camera_frame.size > 0:
        cv2.imwrite('./images/full_camera.png', full_camera_frame)
    else:
        print("Ошибка: полное изображение camera отсутствует или пустое.")

    if full_result_frame is not None and full_result_frame.size > 0:
        cv2.imwrite('./images/full_result.png', full_result_frame)
    else:
        print("Ошибка: полное изображение result отсутствует или пустое.")

    # Обрезаем изображения по габаритам зеленой рамки
    global x1, x2, y1, y2, name_foto
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    width, height = calculate_side_lengths(x1, y1, x2, y2)


    if full_contours_frame is not None and full_contours_frame.size > 0:
        print_size(f'./images/cropped_contours_{name_foto}.png', width, height)
        cropped_contours_frame = rotate_and_crop_image(full_contours_frame, center, angle, width, height)
        cv2.imwrite(f'./images/cropped_contours_{name_foto}.png', cropped_contours_frame)
    else:
        print("Ошибка: обрезанное изображение contours отсутствует или пустое.")

    if full_camera_frame is not None and full_camera_frame.size > 0:
        print_size(f'./images/cropped_camera_{name_foto}.png', width, height)
        cropped_camera_frame = rotate_and_crop_image(full_camera_frame, center, angle, width, height)
        cv2.imwrite(f'./images/cropped_camera_{name_foto}.png', cropped_camera_frame)
    else:
        print("Ошибка: обрезанное изображение camera отсутствует или пустое.")

    if full_result_frame is not None and full_result_frame.size > 0:
        print_size(f'./images/cropped_result_{name_foto}.png', width, height)
        cropped_result_frame = rotate_and_crop_image(full_result_frame, center, angle, width, height)
        cv2.imwrite(f'./images/cropped_result_{name_foto}.png', cropped_result_frame)
    else:
        print("Ошибка: обрезанное изображение result отсутствует или пустое.")

    print("Шесть фотографий сделаны и сохранены.")


def draw_blue_box(img, center, box_width, box_height, move_X, move_Y, angle):
    """Функция для рисования синей рамки."""
    # Новый центр синей рамки
    new_center_blue = (center[0] + move_X, center[1] + move_Y)

    # Вычисляем новые координаты углов синей рамки
    blue_rect_points = [
        (int(new_center_blue[0] - box_width // 2), int(new_center_blue[1] - box_height // 2)),
        (int(new_center_blue[0] + box_width // 2), int(new_center_blue[1] - box_height // 2)),
        (int(new_center_blue[0] + box_width // 2), int(new_center_blue[1] + box_height // 2)),
        (int(new_center_blue[0] - box_width // 2), int(new_center_blue[1] + box_height // 2))   ]


    # # Определяю координаты левого-нижнего и верхнего-правого угла СИНЕЙ рамки
    x1_loc = int(new_center_blue[0] - box_width // 2)
    y1_loc = int(new_center_blue[1] - box_height // 2)

    x2_loc = int(new_center_blue[0] + box_width // 2)
    y2_loc = int(new_center_blue[1] + box_height // 2)



    # Применяем поворот к углам синей рамки
    blue_rotated_points = []
    for point in blue_rect_points:
        rotated_point = rotate_point(point, new_center_blue, angle)
        blue_rotated_points.append(rotated_point)

    # Рисуем синюю рамку
    cv2.polylines(img, [np.array(blue_rotated_points)], True, (255, 0, 0), 2)
    return x1_loc, x2_loc, y1_loc, y2_loc

def on_button_press():
    global button_pressed
    button_pressed = True

def on_button_stop():
    """Функция для остановки поиска области"""
    global stop_refreshing_contours
    stop_refreshing_contours = not stop_refreshing_contours

    print("Поиск объектов приостановлен!")


def buttom_refresh_contours_frame():
    """Функция для обновления окна с зоной поиска"""
    global cap, img, hsv, thresh, contours, hierarchy
    global h1, s1, v1, h2, s2, v2, Ar

    print(1)
    # Проверяем, не был ли установлен флаг остановки
    if stop_refreshing_contours:
        print(2)
        return


    # Захват нового кадра с камеры
    ret, img = cap.read()

    if not ret:
        return

    # Преобразование кадра в HSV формат
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Применение фильтра HSV
    h1 = hue_1_slider.get()
    s1 = satur_1_slider.get()
    v1 = value_1_slider.get()
    h2 = hue_2_slider.get()
    s2 = satur_2_slider.get()
    v2 = value_2_slider.get()
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)
    thresh = cv2.inRange(hsv, h_min, h_max)

    # Поиск контуров
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Очистка и обновление содержимого окна 'contours'
    img2 = img.copy()
    cv2.drawContours(img2, contours, -1, (255, 0, 0), 2, cv2.LINE_4, hierarchy, 2)
    cv2.imshow('contours', img2)

def rotate_point(point, center, angle):
    """
    Функция для поворота точки вокруг указанного центра на заданный угол.

    :param point: Координаты точки (x, y)
    :param center: Координаты центра вращения (x, y)
    :param angle: Угол поворота в градусах
    :return: Новые координаты повернутой точки
    """
    # Переводим угол в радианы
    rad_angle = math.radians(angle)

    # Преобразование координат относительно центра вращения
    rel_x = point[0] - center[0]
    rel_y = point[1] - center[1]

    # Вычисляем новые координаты после поворота
    new_x = rel_x * math.cos(rad_angle) - rel_y * math.sin(rad_angle)
    new_y = rel_x * math.sin(rad_angle) + rel_y * math.cos(rad_angle)

    # Возвращаем абсолютные координаты
    return int(round(new_x + center[0])), int(round(new_y + center[1]))

# Основная функция программы
def main():
    global button_pressed, x1, x2, y1, y2
    global cap, name_foto, global_angle
    global hue_1_slider, satur_1_slider, value_1_slider, hue_2_slider, satur_2_slider, value_2_slider, area_slider, X_slider,Y_slider, X_move_slider, Y_move_slider
    button_pressed = False
    x1, x2, y1, y2 = 0, 0, 0, 0  # Координаты для обрезки по рамке картинки

    cv2.namedWindow("result")  # Создаем главное окно

    cv2.namedWindow("camera")
    cap = cv2.VideoCapture(0)  # Подключаемся к видео камере, передаем в методе индекс веб-камеры
    # ######
    cap.set(cv2.CAP_PROP_FPS, 12)  # Частота кадров
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Ширина кадров в видеопотоке (в первую очередь этот параметр влияет на размер).
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Высота кадров в видеопотоке.
    # ######

    # Создаем окно настроек с трекбарами и кнопкой
    settings_window = tk.Toplevel()
    settings_window.title("Настройки")


    # Трекбары
    hue_1_label = tk.Label(settings_window, text="Hue 1 (Оттенок):")
    hue_1_slider = tk.Scale(settings_window, from_=0, to=255, orient=tk.HORIZONTAL, length=200)
    hue_1_slider.grid(row=0, column=1)
    hue_1_label.grid(row=0, column=0)
    hue_1_slider.set(0)

    satur_1_label = tk.Label(settings_window, text="Satur 1 (Насыщенность):")
    satur_1_slider = tk.Scale(settings_window, from_=0, to=255, orient=tk.HORIZONTAL, length=200)
    satur_1_slider.grid(row=1, column=1)
    satur_1_label.grid(row=1, column=0)
    satur_1_slider.set(8)

    value_1_label = tk.Label(settings_window, text="Value 1 (Яркость):")
    value_1_slider = tk.Scale(settings_window, from_=0, to=255, orient=tk.HORIZONTAL, length=200)
    value_1_slider.grid(row=2, column=1)
    value_1_label.grid(row=2, column=0)
    value_1_slider.set(82)

    hue_2_label = tk.Label(settings_window, text="Hue 2 (Оттенок):")
    hue_2_slider = tk.Scale(settings_window, from_=0, to=255, orient=tk.HORIZONTAL, length=200)
    hue_2_slider.grid(row=3, column=1)
    hue_2_label.grid(row=3, column=0)
    hue_2_slider.set(181)

    satur_2_label = tk.Label(settings_window, text="Satur 2 (Насыщенность):")
    satur_2_slider = tk.Scale(settings_window, from_=0, to=255, orient=tk.HORIZONTAL, length=200)
    satur_2_slider.grid(row=4, column=1)
    satur_2_label.grid(row=4, column=0)
    satur_2_slider.set(202)

    value_2_label = tk.Label(settings_window, text="Value 2 (Яркость):")
    value_2_slider = tk.Scale(settings_window, from_=0, to=255, orient=tk.HORIZONTAL, length=200)
    value_2_slider.grid(row=5, column=1)
    value_2_label.grid(row=5, column=0)
    value_2_slider.set(155)

    area_label = tk.Label(settings_window, text="Area: ")
    area_slider = tk.Scale(settings_window, from_=0, to=120000, orient=tk.HORIZONTAL, length=200)
    area_slider.grid(row=6, column=1)
    area_label.grid(row=6, column=0)
    area_slider.set(100000)

    X_size_label = tk.Label(settings_window, text="X size: ")
    X_slider = tk.Scale(settings_window, from_=0, to=1000, orient=tk.HORIZONTAL, length=150)
    X_slider.grid(row=8, column=0)
    X_size_label.grid(row=7, column=0)
    X_slider.set(294)

    Y_size_label = tk.Label(settings_window, text="Y size: ")
    Y_slider = tk.Scale(settings_window, from_=0, to=1000, orient=tk.HORIZONTAL, length=150)
    Y_slider.grid(row=8, column=1)
    Y_size_label.grid(row=7, column=1)
    Y_slider.set(191)

    X_move_label = tk.Label(settings_window, text="X move: ")
    X_move_slider = tk.Scale(settings_window, from_=-200, to=200, orient=tk.HORIZONTAL, length=150)
    X_move_slider.grid(row=10, column=0)
    X_move_label.grid(row=9, column=0)
    X_move_slider.set(0)

    Y_move_label = tk.Label(settings_window, text="Y move: ")
    Y_move_slider = tk.Scale(settings_window, from_=-200, to=200, orient=tk.HORIZONTAL, length=150)
    Y_move_slider.grid(row=10, column=1)
    Y_move_label.grid(row=9, column=1)
    Y_move_slider.set(-6)

    # Поле для текста
    text_entry_label = tk.Label(settings_window, text="Наименование изделия:")
    text_entry_label.grid(row=11, column=0)

    text_entry = tk.Entry(settings_window)
    text_entry.insert(0, 'BP')

    text_entry.grid(row=11, column=1)

    name_foto = text_entry.get()


    # Кнопка "Сделать фото"
    photo_button = tk.Button(settings_window, text="Сделать фото", command=on_button_press, width=20)
    photo_button.grid(row=12, column=1)

    # Кнопка "Обновить изображение"
    photo_button = tk.Button(settings_window, text="Обновить", command=buttom_refresh_contours_frame, width=20)
    photo_button.grid(row=12, column=0)

    # Кнопка "Stop" (для прекращения поиска но при этом сохранения возможности настройки позиции для фото)
    buttom_stop = tk.Button(settings_window, text='Stop', command=on_button_stop, width=20)
    buttom_stop.grid(row=13, column=0)

    center_blue = (0, 0)
    value_rate = 50  # погрешность смещения центра синей рамки относительно центра зеленой

    while True:
        ret, img = cap.read()  # img - сама картинка с камеры, ret - флаг

        # if stop_refreshing_contours:
        #     continue

        if not ret:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV формат изображения


        # Считывание значений бегунков
        global h1, s1, v1, h2, s2,v2, Ar

        h1 = hue_1_slider.get()
        s1 = satur_1_slider.get()
        v1 = value_1_slider.get()
        h2 = hue_2_slider.get()
        s2 = satur_2_slider.get()
        v2 = value_2_slider.get()
        Ar = area_slider.get()

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
        if stop_refreshing_contours == False: # Если прожата кнопка STOP то поиск зеленой рамки не будет осуществляться

            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                tup = tuple(box.tolist())
                area = calculateTheArea(tup)

                if area >= Ar:
                    cv2.drawContours(img2, [box], -1, (0, 255, 0), 2)

                    center = (int((box[0][0] + box[2][0]) // 2), int((box[0][1] + box[2][1]) // 2))

                    # Рисуем синюю рамку в отдельной функции
                    angle = calculate_rotation_angle(box)
                    x1, x2, y1, y2 = draw_blue_box(img2, center, X_slider.get(), Y_slider.get(), X_move_slider.get(),
                                  Y_move_slider.get(), angle)

                    global_angle = angle

                    cv2.imshow('contours', img2)  ##

                cv2.imshow('result', thresh)

        else:

            x1, x2, y1, y2 = draw_blue_box(img2, center, X_slider.get(), Y_slider.get(), X_move_slider.get(),
                                           Y_move_slider.get(), global_angle)
            cv2.imshow('contours', img2)


        if button_pressed:
            # Формирование координат зеленой рамки
            capture_images(img2, img, thresh, angle)
            button_pressed = False

        if cv2.waitKey(10) == 27:  # Клавиша Esc
            break



    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    thread = Thread(target=run_tkinter)
    thread.start()
    main()
