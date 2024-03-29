from os import environ
environ["CUDA_VISIBLE_DEVICES"]="1"

from tensorflow import keras
from TensorflowUtils.misc import get_model
from ImageMetrics.metrics import SSIM3, PSNRB, SSIM
from ImageMetrics.losses import L3SSIM, LPSNRB, LSSIM
from tensorflow._api.v2.image import ssim
from KerasGAN.CustomModels import GAN
from TensorflowUtils.Callbacks import GANTrainingLogger, MultipleTrainingLogger
from KerasGAN.losses import w_dist
from TensorflowUtils.DataSet import LocalInpaintingDataSet
from TensorflowUtils.Inpainting.masks import generate_square_mask

import numpy as np

data_set = LocalInpaintingDataSet(generate_square_mask)
data_set.normalize()

generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.6, beta_2=0.9
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.6, beta_2=0.9
)

discriminator = get_model(model_json_name = "D-Teste1.json")
generator = get_model(model_json_name = "G-AutoEncoder-0.0-64x64.json")

wgan = GAN(discriminator = discriminator, generator=generator, gp_weight=15)

wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    min_max_loss=w_dist,
    g_losses = [LSSIM()],
    g_losses_weights = [1],
    g_metrics = [SSIM(), PSNRB()]
)

# getting test inputs
rng = np.random.default_rng(12345)
samples = rng.integers(size = [10], low = 0, high = 10_000)
training_samples = data_set.x_test[samples] 

callbacks = [GANTrainingLogger(samples, monitor_metric_name = "SSIM", data_set = data_set)]

wgan.fit(x = data_set.x_train, y = data_set.y_train,
         batch_size=20, epochs=30,
         validation_data = (data_set.x_test, data_set.y_test),
         callbacks = callbacks)
