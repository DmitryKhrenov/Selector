from Selector import *


Selector = SelectorClass()
# print(Selector.find_impresses(image_file=os.path.join(Selector.DATA_DIR, 'dress casual   1005\\14 (2).jpg'), img_category=1))
# Selector.find_show_impresses(image_file=os.path.join(Selector.DATA_DIR, 'dress casual   1005\\14 (2).jpg'), img_category=1)
# Selector.find_show_impresses(image_file=os.path.join(Selector.DATA_DIR, 'dress homemade 1005\\139.jpg'), img_category=2) --
print(Selector.find_impresses(image_file=os.path.join(Selector.DATA_DIR, 'shirt men - 4 112\\3 (4).jpg'), img_category=4))
Selector.find_show_impresses(image_file=os.path.join(Selector.DATA_DIR, 'shirt men - 4 112\\3 (4).jpg'), img_category=4)

