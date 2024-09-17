
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


file_name = 'Step size0.0030_Dwell time4 OVCAR5_799_30mW_1040_50mW_P15450_F15060_MFT_51_60Xobj_xoffset_-0.9_yoffset_0.8_Xchan_TC_2us_1.tif'
folder_name = '20240306_OVCAR5_50mW/'
x = imread('data/' + folder_name + file_name)

axes = 'ZYX'
print('image size =', x.shape)
print('image axes =', axes)

model = CARE(config=None, name='my_model_20240908_test_4unet', basedir='models')



# restored = model.predict(x, axes)
restored = model.predict(x, axes, n_tiles=(1,4,4))


Path('results').mkdir(exist_ok=True)
save_tiff_path = 'results/%s_'+file_name
print(save_tiff_path)
save_tiff_imagej_compatible(save_tiff_path % model.name, restored, axes)
