

from os import environ
environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
from tensorflow import keras
from TensorflowUtils.misc import get_model
from TensorflowUtils.CustomModels import WGAN
from TensorflowUtils.DataUtils import generate_square_mask
from ImageMetrics.metrics import SSIM3, PSNRB, SSIM
from ImageMetrics.losses import L3SSIM, LPSNRB, LSSIM
from tensorflow._api.v2.image import ssim
import sys
sys.path.insert(1, '/home/apeterson056/AutoEncoder/codigoGitHub/IC-AutoEncoder/modules') 
from DataMod import DataSet

data_set = DataSet().load_rafael_cifar_10_noise_data()

mask1 = np.array([generate_square_mask((64, 64, 1), 15, 25) for _ in range(50_000)])
data = np.array(np.array([data_set.x_train*mask1, data_set.y_train]))/255

def LW_distance(real_disc_output, fake_disc_output):
    real_loss = tf.reduce_mean(real_disc_output)
    fake_loss = tf.reduce_mean(fake_disc_output)
    return fake_disc_output - real_disc_output

generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

discriminator = get_model(model_json_name = "D-Teste1.json")
generator = get_model(model_json_name = "G-AutoEncoder-0.0-64x64.json")
wgan = WGAN(discriminator = discriminator, generator=generator)

wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    min_max_loss=LW_distance,
    g_losses = [LSSIM(), LPSNRB()],
    g_losses_weights = [5, 0.1],
    g_metrics = [SSIM(), PSNRB()]
)

wgan.fit(x = data[0], y = data[1], batch_size=25, epochs=8)