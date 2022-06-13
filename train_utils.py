from tensorflow.keras.utils import Sequence, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.losses import Loss
import tensorflow as tf
import numpy as np
import os
from utils import *
from mcnn import *
class CrowdHumanDataGenerator(Sequence):
    def __init__(self, processed_image_folder, gt_density_maps, batch_size):
        self.gt_density_maps = gt_density_maps
        self.image_ids = [id for id in gt_density_maps.keys()]
        self.image_paths = [os.path.join(processed_image_folder, id +".jpg") for id in gt_density_maps.keys()]
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.image_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        
        imgs = np.array([img_to_array(load_img(img_path)) for img_path in batch_x])
        labels = np.array([self.gt_density_maps[img_id] for img_id in batch_y])
        #print(imgs.shape, labels.shape)
        return imgs / 255.0, labels.astype(np.float32)

import math
from keras.callbacks import Callback
from keras import backend as K


class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

from datetime import datetime
from pytz import timezone

class CrowdHumanTrainer:
    def __init__(self, processed_train_image_folder, processed_valid_image_folder, train_gt_density_pkl, valid_gt_density_pkl):
        self.train_folder = processed_train_image_folder
        self.valid_folder = processed_valid_image_folder
        
        self.train_gt_density_maps = load_gt_density_maps(train_gt_density_pkl)
        self.valid_gt_density_maps = load_gt_density_maps(valid_gt_density_pkl)
        
        self.model = self.build_mcnn_model()
        
    def build_mcnn_model(self):
        def mape(y_true, y_pred):
            batch_true = tf.reduce_sum(y_true, axis=(1,2))
            batch_pred = tf.reduce_sum(y_pred, axis=(1,2))
            mape = tf.reduce_mean(tf.math.abs((batch_true - batch_pred)/batch_true))
            return mape

        model = MCNN()
        optimizer = tf.keras.optimizers.Adam(0.001)

        model.compile(loss="mse",
                    optimizer=optimizer,
                    metrics=['mae', 'mse', mape])
        return model
    
    def get_model_size(self):
        size = 0
        for layer in self.model.layers:
            if layer:
                for weight in layer.weights:
                    size += tf.math.reduce_prod(weight.shape).numpy()
        return size * (32 / 8) * (10 ** (-6)) 

    
    def train(self, epochs, batch_size):
        #print("model size %.2f MB" % (self.get_model_size()))
        def lr_scheduler(epoch, lr):
            if epoch < 5:
                return lr
            else:
                return lr * np.exp(-0.1)
        train_generator = CrowdHumanDataGenerator(self.train_folder, self.train_gt_density_maps, batch_size)
        valid_generator = CrowdHumanDataGenerator(self.valid_folder, self.valid_gt_density_maps, batch_size)
        now= datetime.now(timezone('Asia/Seoul')).strftime("%Y%m%d-%H%M%S")
        logdir = "./logs/" + now
        ckpt_path = "./alpaca/" + now
        print("Tensorboard path", logdir)
        callbacks = [TensorBoard(log_dir=logdir, histogram_freq = 1, write_graph=True, update_freq = "epoch"), \
                     #CosineAnnealingScheduler(T_max = epochs, eta_max = 0.0001, eta_min=1e-7,), \
                     #EarlyStopping(monitor="val_loss", patience=10),\
                     #ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=5, verbose=1), \
                     ModelCheckpoint(filepath=ckpt_path, monitor="val_mape", mode="min", save_freq="epoch", save_best_only=True)]
        
        history = self.model.fit(train_generator, validation_data=valid_generator, epochs=epochs, callbacks=callbacks)
    
    
    def evaluate(self):
        data_generator = CrowdHumanDataGenerator(self.valid_folder, self.valid_gt_density_maps, batch_size = 1)

        mae = 0.0
        mape = 0.0
        adj_mae = 0.0
        adj_mape = 0.0
        for i in range(len(data_generator)):
            image, gt_density = data_generator[i]
            gt_num = np.sum(gt_density)
            pred_density = self.model(image)
            pred_num = np.sum(pred_density)    
            mae += abs(pred_num - gt_num)
            mape += abs((pred_num - gt_num)/gt_num)
            adj_pred_num = tf.reduce_sum(pred_density[pred_density > np.mean(pred_density)]).numpy()
            adj_mae += abs(adj_pred_num - gt_num)
            adj_mape += abs((adj_pred_num - gt_num)/gt_num)
            
        mae /= len(data_generator)
        mape /= len(data_generator)
        adj_mae /= len(data_generator)
        adj_mape /= len(data_generator)
        model_size = self.get_model_size()
        img_size = tf.reduce_prod(image.shape[1:]).numpy() * (32 / 8) * (10  ** (-6))
        mem = model_size + img_size
        print("mae %.2f mape %.2f adj_mae %.2f adj_mape %.2f mem %.2f MB" % (mae, mape, adj_mae, adj_mape, mem))
