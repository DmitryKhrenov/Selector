from PIL import Image, ImageOps
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt # графики
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# используемые типы
dt_mainclr = np.dtype([('fname_col', 'U250'), ('colors_col', object)])

# вспомогательные функции


def color_distanceHSV(arr1, arr2):
    """
    Возвращет набор расстояний для цветов из массивов. Сильно отличающиеся цвета не учитываются.
    HSV : (оттенок(0..255), насыщенность(0..255), яркость(0..255))
    дистанция 0..255 = |H1-H2|/255 * 50% + |S1-S2|/255*30% + |V1-V2|/255*20%

    :param arr1: массив основных цветов [[4]]
    :param arr2: массив основных цветов [[4]]
    :return: расстояние
    """

    min_dist = 442  # больше быть не может
    for color1 in arr1:
        hsv1 = cv2.cvtColor(np.uint8([[color1[1:]]]), cv2.COLOR_RGB2HSV_FULL)[0][0].astype('float32')
        for color2 in arr2:
            if (color1[0] > 300) and (color2[0] > 300):  # для каждого цвета > 30%
                hsv2 = cv2.cvtColor(np.uint8([[color2[1:]]]), cv2.COLOR_RGB2HSV_FULL)[0][0].astype('float32')
                if (((h := abs(hsv1[0] - hsv2[0])) < 25) and  # оттенок - 10%
                        ((s := abs(hsv1[1] - hsv2[1])) < 25) and  # насыщенность - 10%
                        ((v := abs(hsv1[2] - hsv2[2])) < 51)):  # яркость - 20%
                    dist = int((h * 60 + s * 30 + v * 10) / 100)
                    min_dist = min(min_dist, dist)
    return min_dist


def image_chage_base(img, mask_color=(0, 0, 0), cmap='BGR'):
    """
    :param img: входящее мзображение
    :param mask_color: чем закрашивать
    :param cmap: цветовая схема входящего изображения
    :return: изображение, с закрашенным фоном
    заменить изображение подложки
    """

    img = img.astype('uint8')

    # Преобразуйте изображение в черно-белое
    if cmap.upper() == 'BGR': c_conv = cv2.COLOR_BGR2GRAY
    if cmap.upper() == 'RGB': c_conv = cv2.COLOR_RGB2GRAY
    if cmap.upper() == 'HSV': c_conv = cv2.COLOR_HSV2GRAY

    # Преобразуйте изображение в черно-белое
    gray = cv2.cvtColor(img, c_conv)

    # Применение порогового фильтра
    ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    # создаем структурирующее ядро
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # применяем операцию замыкания
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # находим контуры
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # создание маски для выделения контуров на изображении 1
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    # создание маски для копирования содержимого контуров на изображение 2
    inv_mask = cv2.bitwise_not(mask)

    # копирование содержимого контуров на изображение 2
    image2 = np.zeros(img.shape, dtype=np.uint8)
    image2[:, :] = mask_color  # цвет заливки

    # заполнение пространства внутри контуров выбранным цветом
    cv2.fillPoly(image2, contours, (0, 0, 0))

    image1_masked = cv2.bitwise_and(img, img, mask=mask)
    result = cv2.add(image1_masked, image2)

    return result


def get_main_colors(img, num_clusters=None, in_cmap='BGR', out_cmap='RGB'):
    # Преобразование изображения в массив
    img_array = img.reshape((-1, 3))

    # Создание объекта KMeans
    if num_clusters:
        kmeans = KMeans(n_init='auto', n_clusters=num_clusters)
    else:
        kmeans = KMeans(n_init='auto')

    # Обучение модели KMeans на массиве цветов
    kmeans.fit(img_array)

    # Получение центров кластеров
    colors = kmeans.cluster_centers_[:10]

    if ((in_cmap == 'RGB') and (out_cmap == 'BGR')) or ((in_cmap == 'BGR') and (out_cmap == 'RGB')):
        colors = colors[:,
                 ::-1]  # перейдем к световой модели cv2 #colors = cv2.cvtColor(colors.astype(int), cv2.COLOR_RGB2BGR)

    # получить метки кластеров
    labels = kmeans.labels_

    # подсчитать размеры кластеров
    sizes = np.bincount(labels)

    # Объединение массивов по первому столбцу
    result = np.concatenate((sizes.reshape(-1, 1), colors), axis=1).astype(int)
    result = result[result[:, 0].argsort()[::-1]]

    # Создание булевого массива для удаления строк
    mask = np.sum(result[:, -3:],
                  axis=1) <= 3  # здесь завязано на маску по черному цвету - отсекаем группы цветов, у которых ясркость меньше 3 - х

    # Удаление строк из массива
    return result[~mask]

def color_grouping(colors_freq, dist=50, min_fraction_slice=None):
    """
    Объединить кластеры с близкими цветами
    на вход 4N отсортированная матрица, расстояние объединения, доля для игнорирования
    :param colors_freq:
    :param dist:
    :param min_fraction_slice:
    :return: основные цвета [[4]]
    """
    colors_len = colors_freq.shape[0]  # сколько кластеров передано на группировку
    new_colors = []
    summator = np.zeros(colors_freq.shape[1], dtype=np.uint8)
    total_weight = np.sum(colors_freq[:, 0])  # сумма долей/точек поступивших кластеров
    if min_fraction_slice:  # с каким количеством  точек отбрасывать итоговые кластеры
        slice_level = int(min_fraction_slice * 1000)

    i = 0
    c = 0  # количество группируемых значений. но зачем ограничивать
    while (i < colors_len) and (c < 100):
        c += 1
        v = colors_freq[i][0]
        summator = np.array([v, colors_freq[i][1] * v, colors_freq[i][2] * v, colors_freq[i][3] * v])

        j = i + 1
        while (j < colors_len):
            d = distance.euclidean(colors_freq[i][1:], colors_freq[j][1:])

            if d > dist:
                break
            else:
                v1 = colors_freq[j][0]
                v += v1
                summator += [v1, colors_freq[j][1] * v1, colors_freq[j][2] * v1, colors_freq[j][3] * v1]
                colors_freq = np.delete(colors_freq, j, axis=0)
                colors_len -= 1
                # j увеличивать не надо, т.к. удалили строку

        summator[1:] = summator[1:] / v
        summator[0] = summator[0] * 1000 / total_weight

        if (i == 0) or not (min_fraction_slice) or ((min_fraction_slice) and (summator[0] > slice_level)):
            new_colors.append(summator)
        i = j

    return np.array(new_colors)


def image_main_colors(img_file_name, image=None, num_clusters=7, dist=50, min_fraction_slice=0.2, in_cmap='BGR',
                      out_cmap='RGB'):
    """
    Получение основных цветов картинки
    Возвращает строку, содержащую полное имя картинки (или текстовую метку), и массив 4N цветов с их долями.
    Тип возвращаемой строки dt = np.dtype([('fname_col', 'U250'), ('colors_col', object)])
    Если img=None, то пытается загрузить картинку по имени.
    Если DATA_DIR опрtделена, то добавляется к имени при загрузке, но не сохраняется в выходном массиве
    :param img_file_name:
    :param image:
    :param num_clusters:
    :param dist:
    :param min_fraction_slice:
    :param in_cmap: цветовая схемя входящей картинки
    :param out_cmap: цветовая схема найденных цветов
    :return:
    """
    if image is None:  # значит надо вычитать из файла
        image = cv2.imread(img_file_name)  # по умолчанию загружает в формате 'bgr'
        if in_cmap == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_under_mask = image_chage_base(image, (0, 0, 0), cmap=in_cmap)  # заливка черным цветом по контуру

    main_colors = get_main_colors(img_under_mask, num_clusters=num_clusters, in_cmap=in_cmap, out_cmap=out_cmap)

    main_colors = color_grouping(main_colors, dist=dist, min_fraction_slice=min_fraction_slice)

    return np.array([(img_file_name, main_colors)], dtype=dt_mainclr)[0]


def show_images(images, labels=None, file_names=None, per_line=None, min_size=2, cmap='RGB'):
    """
    Показать картинки, по N в одну строку
    :param images:
    :param labels:
    :param file_names:
    :param per_line:
    :param min_size:
    :param cmap:
    :return:
    """
    img_cnt = len(images)

    if labels:
        l_cnt = len(labels)
    else:
        l_cnt = 0

    if per_line:
        h_cnt = per_line
    else:
        h_cnt = round(img_cnt ** (0.5))  # корень квадратный

    v_cnt = img_cnt // h_cnt
    if img_cnt % h_cnt: v_cnt += 1

    h_len = 20  # ширина/высота полотна в дюймах
    a = 18 // h_cnt
    v_len = v_cnt * a

    # Рисуем изображения в виде квадратной сетки h_cnt x v_cnt
    fig, axes = plt.subplots(v_cnt, h_cnt, figsize=(h_len, v_len))  # count, 1, figsize=(10,10))
    axes = axes.ravel()
    for i in np.arange(0, len(axes)):
        if i < img_cnt:
            # Преобразование в RGB
            if cmap == 'RGB':
                axes[i].imshow(images[i].astype(
                    "uint8"))  # вариант для загрузки через cv2 : cv2.cvtColor(images[i].numpy().astype("uint8"), cv2.COLOR_BGR2RGB) )
            if cmap == 'BGR':
                axes[i].imshow(cv2.cvtColor(images[i].numpy().astype("uint8"), cv2.COLOR_BGR2RGB))
            if cmap == 'HSV':
                axes[i].imshow(cv2.cvtColor(images[i].numpy().astype("uint8"), cv2.COLOR_HSV2RGB))

        if i < l_cnt:
            axes[i].set_title(labels[i],
                              fontsize=8)  # f"Label: {labels[i]}, File Name: {file_names[i]}" if i < l_cnt else str(i))
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0.5)
    plt.show()


def image_color_samples(img, color_samples, samples_wide=0.2, max_samples=10, max_sample_size=0.08, img_cmap='BRG',
                        clr_cmap='RGB'):
    """
    нарисовать N-цетовых сэмплов на изображении
    :param img:
    :param color_samples:
    :param samples_wide:
    :param max_samples:
    :param max_sample_size:
    :param img_cmap:
    :param clr_cmap:
    :return:
    """
    # на всякий случай переведм  значения цветов в целое
    color_weight = []

    if (color_samples.shape[1]) == 4:
        color_weight = color_samples[:, 0]
        color_samples = np.round(color_samples[:, 1:]).clip(0, 255).astype(np.uint8)
    else:
        color_samples = np.round(color_samples).clip(0, 255).astype(np.uint8)

    # получение размеров изображения
    h, w = img.shape[:2]
    # вычисление ширины добавляемой области
    border_width = int(w * 0.2)
    # добавление пустой области справа
    img_border = cv2.copyMakeBorder(img, 0, 0, 0, border_width, cv2.BORDER_REPLICATE, value=(255, 255, 255))

    # вычисление размера фигуры
    part_step = min(h / (min(color_samples.shape[0], max_samples) + 1), border_width)
    part_width = int(part_step * 0.8)
    part_offs = int(part_step / 2)

    # создание изображения для отображения цветовых квадратов
    color_img = np.zeros((part_width, part_width, 3), dtype=np.uint8)

    x = int(w + border_width * 0.1)
    for i, color in enumerate(color_samples):
        # заполнение квадрата цветом
        if ((img_cmap == 'RGB') and (clr_cmap == 'BGR')) or ((img_cmap == 'BGR') and (clr_cmap == 'RGB')):
            color_img[:, :, :] = color[::-1]
        else:
            color_img[:, :, :] = color[::]

        # вывод квадрата на изображение
        y = part_offs + int(i * part_step)
        if (i >= max_samples) or (y + part_width >= h): break

        img_border[y: y + part_width, x: x + part_width, :] = color_img
        if len(color_weight) > 0:
            cv2.putText(img_border, f'{(color_weight[i] / 10):2.1f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 0, 0],
                        1)

    return img_border

