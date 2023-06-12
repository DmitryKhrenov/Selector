import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt  # графики
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.image import resize

from help_funcs import *


class SelectorException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message


class SelectorClass:
    DATA_DIR = 'C:\\Selector\\Images'  # каталог с картинками по пакам - классам
    COLOR_DIR = 'C:\\Selector'  # каталог с сохраненными таблицами и обученной сеткой
    encoder_shape = (224, 224, 3)
    catalog_dict = {
        'dress business'    : 'dress  business  986',
        'dress casual'      : 'dress casual   1005',
        'dress homemade'    : 'dress homemade 1005',
        'dress solemn'      : 'dress solemn 1057',
        'shirt men'         : 'shirt men - 4 112',
        'shirt women'       : 'shirt women 1208',
        'sportswear men'    : 'sportswear men 1003',
        'sportswear women'  : 'sportswear women 1065',
        't-shirt men'       : 't-shirt men  1025',
        't-shirt women'     : 't-shirt women 1248',
        'trousers men'      : 'trousers men - 2656',
        'trousers women'    : 'trousers women 1037'
    }

    def __init__(self, encoder_file='encoder224.h5', color_file='_colors.npy', impress_file='impress224.npy',
                 shape=(224, 224, 3)):
        print(f'Папка с картинками : {self.DATA_DIR}')
        print(self.encoder_shape)
        self.encoder = load_model(os.path.join(self.COLOR_DIR, encoder_file))


        self._colors = np.load(os.path.join(self.COLOR_DIR, color_file), allow_pickle=True)
        # удалю название папки данных, и подпапок. останутся только категории и названия файлов
        OLD_DATA_DIR = 'res_Selector'  # из этой папки считал цвета. она осталась в названии файлов
        files = list(s.replace(OLD_DATA_DIR + "/", "").replace('training/', "").replace('validation/', "") for s in
                     self._colors[:]['fname_col'])
        # заменю категории на старые имена папок
        files = list(os.path.join(self.catalog_dict[d], f) for d, f in (os.path.split(f) for f in files))
        self._colors[:]['fname_col'] = files
        print(f'Загужено цветовых меток {self._colors.shape[0]}')


        OLD_IMPRESS_DIR = 'res224_Selector'  # из этой папки считал цвета. она осталась в названии файлов
        self._impress = np.load(os.path.join(self.COLOR_DIR, impress_file), allow_pickle=True)
        files = list(s.replace(OLD_IMPRESS_DIR + "/", "").replace('training/', "").replace('validation/', "") for s in
                     self._impress[:]['fname_col'])
        files = list(os.path.join(self.catalog_dict[d], f) for d, f in (os.path.split(f) for f in files))
        self._impress[:]['fname_col'] = files
        print(f'Загружено образов картинок {self._impress.shape[0]}')


        self.encoder_shape = shape

        # восстановим словарь категорий
        self.class_names = list(dict(np.unique(self._impress[['category_name', 'category_col']])))
        self.class_indices = {class_name: i for i, class_name in enumerate(self.class_names)}
        print('Имена категорий:\n',self.class_indices,'\n')

    def find_impresses(self, img=None, image_file=None, img_category=None, img_category_name=None, imp_count=9,
                       in_cmap='RGB'):
        """
        можно указать или код категории img_category,или название категории img_category_name
        для поиска без учета категории не указывать ни чего
        автоматически определять категорию img_category = -1
        :param image_file:
        :param img:
        :param img_category:
        :param img_category_name:
        :param imp_count:
        :param in_cmap:
        :return:
        """
        dt_imp = np.dtype(
            [('fname_col', 'U250'), ('colors_dist', float), ('hiden_dist', float), ('colors_col', object),
             ('idx', int)])

        imp_result = np.empty((0,), dtype=dt_imp)  # сразу создаем массив максимального размера

        # загружу, если требуется картинку
        if (image_file is not None) and os.path.exists(image_file):
            img = np.array(load_img(image_file, target_size=self.encoder_shape[:2], color_mode=in_cmap.lower()))
        elif img is not None:
            img_shape = img.shape
            if (len(img_shape) != 3) or not (img_shape[0] == img_shape[1]) or (img_shape[2] not in [1, 3, 4]):
                raise ValueError("find_impresses: img должен иметь размерность изображения")
            img = np.array(resize(img, size=self.encoder_shape[:2]))
        else:
            raise ValueError("find_impresses: должны быть указаны или 'img', или 'image_file'")

        # получу основные цвета искомой картинки
        img_colors = \
            image_main_colors('', image=img, num_clusters=5, dist=75, min_fraction_slice=0.30, in_cmap=in_cmap,
                              out_cmap='RGB')[1]
        # image_main_colors возвращает tuple, второй элемент - набор основных цветов. надо убрать первй столбец  - доля цвета

        # Кодирование входного изображения
        code = self.encoder.predict(np.expand_dims(img, axis=0), verbose=0)

        # восстановим код категории
        if (img_category is None) and (img_category_name is not None):
            img_category = self.class_names.index(img_category_name)

        if img_category is not None:
            idxs = np.where(self._impress['category_col'] == img_category)[0]
        else:
            idxs = range(self._impress['category_col'].shape[0])

        for idx in idxs:
            impress = self._impress[idx]
            colors_dist = color_distanceHSV(img_colors, impress['colors_col'][0])
            if colors_dist < 40:  # предлагаем только близкие цвета
                dist = np.linalg.norm(code - impress['hiden_col'])
                #            dist =  cosine(code[0], impress['hiden_col'][0])
                new_row = np.array([(impress['fname_col'], colors_dist, dist, impress['colors_col'], idx)],
                                   dtype=dt_imp)
                imp_result = np.append(imp_result, new_row)

        imp_result = imp_result[imp_result['hiden_dist'].argsort()]

        return imp_result[:min(imp_count, imp_result.shape[0])]


    def find_show_impresses(self, img=None, image_file=None, img_category=None, img_category_name=None, imp_count=9, in_cmap='RGB', show_sample=False):
        # загружу, если требуется картинку
        if (image_file is not None) and os.path.exists(image_file):
            img = np.array(load_img(image_file, target_size=self.encoder_shape[:2], color_mode=in_cmap.lower()))
        elif img is not None:
            img_shape = img.shape
            if (len(img_shape) != 3) or not (img_shape[0] == img_shape[1]) or (img_shape[2] not in [1, 3, 4]):
                raise ValueError("find_impresses: img должен иметь размерность изображения")
            img = resize(img, size=(self.encoder_shape[:2]))
        else:
            raise ValueError("find_impresses: должны быть указаны или 'img', или 'image_file'")


        if show_sample:
            img_colors = image_main_colors('', image=img, num_clusters=5, dist=75, min_fraction_slice=0.30, in_cmap='RGB')[
                1]
            img_for_show = image_color_samples(img, img_colors, samples_wide=0.2, max_samples=10, max_sample_size=0.08,
                                               img_cmap='RGB')
        else:
            img_for_show = img

        if (img_category is None) and (img_category_name is not None):
            img_category = self.class_names.index(img_category_name)
        elif (img_category is not None) and (img_category_name is None):
            img_category_name = self.class_names[img_category]

        # print('Поиск изображения из класса - ',img_category )
        plt.axis("off")
        plt.imshow(img_for_show.astype("uint8"))
        plt.title(img_category_name)

        impress_imgs = self.find_impresses(img, img_category=img_category, img_category_name=img_category_name, in_cmap=in_cmap,
                                      imp_count=imp_count)

        images = []
        labels = []
        for imp in impress_imgs:
            img = np.uint8(load_img(os.path.join(self.DATA_DIR, imp[0]), target_size=self.encoder_shape[:2], color_mode="rgb"))
            if show_sample:
                # color_samples = image_main_colors('', image=img, in_cmap='RGB')[1]
                img_for_show = image_color_samples(img, imp[3][0], samples_wide=0.2, max_samples=10, max_sample_size=0.08,
                                                   img_cmap='RGB')
            else:
                img_for_show = img

            images.append(img_for_show)
            labels.append(f'H-{imp[2]:5.1f} C-{imp[1]:3.1f}')

        show_images(images, labels=labels, per_line=5)

        return