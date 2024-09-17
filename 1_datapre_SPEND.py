

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches


file_name = 'Step size0.0030_Dwell time4 OVCAR5_799_30mW_1040_50mW_P15450_F15060_MFT_51_60Xobj_xoffset_-0.9_yoffset_0.8_Xchan_TC_2us_4.tif_4_2.tif'
y = imread('data/Input/' + file_name)
x = imread('data/Target/' + file_name)
print('image size =', x.shape)

plt.figure(figsize=(16,10))
plot_some(np.stack([x,y]),
          title_list=[['low (maximum projection)','GT (maximum projection)']], 
          pmin=2,pmax=99.8);



raw_data = RawData.from_folder (
    basepath    = 'data/',
    source_dirs = ['Input'],
    target_dir  = 'Target',
    axes        = 'ZYX',
)



X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = (16,64,64),
    n_patches_per_image = 128,
    save_file           = 'data/my_training_data.npz',
)


assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)




for i in range(2):
    plt.figure(figsize=(16,4))
    sl = slice(8*i, 8*(i+1)), 0
    plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
    plt.show()
None;




