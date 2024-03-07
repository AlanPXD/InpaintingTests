from os import environ
environ["CUDA_VISIBLE_DEVICES"]="1"

from tensorflow import keras
from TensorflowUtils.misc import get_model
from ImageMetrics.metrics import SSIM3, PSNRB, SSIM
from ImageMetrics.losses import L3SSIM, LPSNRB, LSSIM
from tensorflow._api.v2.image import ssim
from KerasGAN.CustomModels import GAN
from TensorflowUtils.Callbacks import GANTrainingLogger, ImageToImageLogger
from KerasGAN.losses import w_dist
from TensorflowUtils.DataSet import LocalInpaintingDataSet
from TensorflowUtils.Inpainting.masks import generate_square_mask

import numpy as np

data_set = LocalInpaintingDataSet(generate_square_mask)
data_set.normalize()

generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.6, beta_2=0.9
)

generator = get_model(model_json_name = "G-AutoEncoder-0.0-64x64.json")

# getting test inputs
rng = np.random.default_rng(12345)
samples = rng.integers(size = [10], low = 0, high = 10_000)
training_samples = data_set.x_test[samples] 

callbacks = [ImageToImageLogger(samples, monitor_metric_name = "SSIM", data_set = data_set)]
metrics = [SSIM(), PSNRB()]

generator.compile(loss = LSSIM(), metrics = metrics, optimizer = generator_optimizer)

generator.fit(x = data_set.x_train, y = data_set.y_train, 
              epochs = 30, batch_size = 20,
              callbacks = callbacks,
              validation_data = (data_set.x_test, data_set.y_test))
