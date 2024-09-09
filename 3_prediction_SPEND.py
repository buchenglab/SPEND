#!/usr/bin/env python
# coding: utf-8

# <hr style="height:2px;">
# 
# # Demo: Apply trained CARE model for denoising of *Tribolium castaneum*
# 
# This notebook demonstrates applying a CARE model for a 3D denoising task, assuming that training was already completed via [2_training.ipynb](2_training.ipynb).  
# The trained model is assumed to be located in the folder `models` with the name `my_model`.
# 
# More documentation is available at http://csbdeep.bioimagecomputing.com/doc/.

# In[1]:


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

# <hr style="height:2px;">
# 
# # Download example data
# 
# The example data (also for testing) should have been downloaded in [1_datagen.ipynb](1_datagen.ipynb).  
# Just in case, we will download it here again if it's not already present.

# <hr style="height:2px;">
# 
# # Raw low-SNR image and associated high-SNR ground truth
# 
# Plot the test stack pair and define its image axes, which will be needed later for CARE prediction.

# In[7]:

file_name = 'Step size0.0030_Dwell time4 OVCAR5_799_30mW_1040_50mW_P15450_F15060_MFT_51_60Xobj_xoffset_-0.9_yoffset_0.8_Xchan_TC_2us_1.tif'
folder_name = '20240306_OVCAR5_50mW/'
# y = imread('data/tribolium/test/GT/' + file_name)
x = imread('data/' + folder_name + file_name)
# y = x

axes = 'ZYX'
print('image size =', x.shape)
print('image axes =', axes)

# plt.figure(figsize=(16,10))
# plot_some(np.stack([x,y]),
#           title_list=[['low (maximum projection)','GT (maximum projection)']],
#           pmin=2,pmax=99.8);


# <hr style="height:2px;">
# 
# # CARE model
# 
# Load trained model (located in base directory `models` with name `my_model`) from disk.  
# The configuration was saved during training and is automatically loaded when `CARE` is initialized with `config=None`.

# In[8]:


model = CARE(config=None, name='my_model_20240908_test_4unet', basedir='models')


# ## Apply CARE network to raw image
# 
# Predict the restored image (image will be successively split into smaller tiles if there are memory issues).

# In[9]:


# get_ipython().run_cell_magic('time', '', 'restored = model.predict(x, axes)\n')
# restored = model.predict(x, axes)
restored = model.predict(x, axes, n_tiles=(1,4,4))
# Alternatively, one can directly set `n_tiles` to avoid the time overhead from multiple retries in case of memory issues.
# 
# **Note**: *Out of memory* problems during `model.predict` can also indicate that the GPU is used by another process. In particular, shut down the training notebook before running the prediction (you may need to restart this notebook).

# In[5]:


# get_ipython().run_cell_magic('time', '', 'restored = model.predict(x, axes, n_tiles=(1,4,4))\n')


# ## Save restored image
# 
# Save the restored image stack as a ImageJ-compatible TIFF image, i.e. the image can be opened in ImageJ/Fiji with correct axes semantics.

# In[10]:


Path('results').mkdir(exist_ok=True)
save_tiff_path = 'results/%s_'+file_name
print(save_tiff_path)
save_tiff_imagej_compatible(save_tiff_path % model.name, restored, axes)
# save_tiff_imagej_compatible('results/%s_Ecoli_200ns_g1kmW_600k_60x__FOV 0_200X200X100nm_dwell20us_14h27m37s_Hyper_AutoWN0_LIA.tif' % model.name, restored, axes)


# <hr style="height:2px;">
# 
# # Raw low/high-SNR image and denoised image via CARE network
#
# Plot the test stack pair and the predicted restored stack (middle).

# In[11]:


# plt.figure(figsize=(16,10))
# plot_some(np.stack([x,restored,y]),
#           title_list=[['low (maximum projection)','CARE (maximum projection)','GT (maximum projection)']],
#           pmin=2,pmax=99.8);
# plt.show()


# In[ ]:




