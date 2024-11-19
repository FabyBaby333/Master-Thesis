import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import Functions.dataFrameTools as dataFrameTools

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from matplotlib.pyplot import figure

def plot_history(history):
        font = {'family' : 'normal',
                'weight' : 'normal',
                'size'   : 14}

        plt.rc('font', **font)
        plt.rc('axes', labelsize=22)     # fontsize of the axes title
        plt.rc('axes', titlesize=22) 
        figure(figsize=(8, 5))
        plt.plot(history.history['loss'],color='#173000')
        plt.plot(history.history['val_loss'], color='#b73000')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

def loadData2():
    df=dataFrameTools.normalizeDF(0).astype('float32')
    split = train_test_split(df, test_size=0.3, random_state=42)
    dtrain, dtv = split
    dtest, dval = train_test_split(dtv, test_size=0.5, random_state=42)
    ddims=len(dtrain.columns)
    return dtrain,dval,dtest,ddims


def my_loadData2(meaningfull_bikes_df):
    df=dataFrameTools.normalizeDF(0).astype('float32')

    df = df.loc[meaningfull_bikes_df.index]

    split = train_test_split(df, test_size=0.3, random_state=42)
    dtrain, dtv, = split
    dtest, dval = train_test_split(dtv, test_size=0.5, random_state=42)
    ddims=len(dtrain.columns)
    return dtrain,dval,dtest,ddims


def loadData2():
    df=dataFrameTools.normalizeDF(0).astype('float32')
    split = train_test_split(df, test_size=0.3, random_state=42)
    dtrain, dtv = split
    dtest, dval = train_test_split(dtv, test_size=0.5, random_state=42)
    ddims=len(dtrain.columns)
    return dtrain,dval,dtest,ddims


def custom_sigmoid_cross_entropy_loss_with_logits(x_true, x_recons_logits):
    raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_true, logits=x_recons_logits)
    if len(np.shape(x_true))==4:
        neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy, axis=[1, 2, 3])
    else:
        neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy, axis=[1])
    return tf.math.reduce_mean(neg_log_likelihood)


class dVAE:
    def __init__(self, datadims, latent_dim, kl_weight, learning_rate):
        self.dim_x = datadims
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
    def encoder(self, return_mean = 0, trainable=True):
        # define prior distribution for the code, which is an isotropic Gaussian
        prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(self.latent_dim), scale=1.), 
                                reinterpreted_batch_ndims=1)
        
        model=[layers.InputLayer(input_shape=self.dim_x)]
        
        model.append(layers.Dense(200, name="e0"))
        model.append(layers.BatchNormalization())
#         model.append(layers.LeakyReLU())
        model.append(layers.ReLU())
        model.append(layers.Dropout(0.05))

        model.append(layers.Dense(200, name="e1"))
        model.append(layers.BatchNormalization())
#         model.append(layers.LeakyReLU())
        model.append(layers.ReLU())
        model.append(layers.Dropout(0.05))
        
        model.append(layers.Dense(200, name="e2"))
        model.append(layers.BatchNormalization())
#         model.append(layers.LeakyReLU())
        model.append(layers.ReLU())
        model.append(layers.Dropout(0.05))
        
        model.append(layers.Dense(200, name="e3"))
        model.append(layers.BatchNormalization())
        model.append(layers.LeakyReLU())
        model.append(layers.Dropout(0.05))
        
        model.append(layers.Dense(200, name="e4"))
        model.append(layers.BatchNormalization())
#         model.append(layers.LeakyReLU())
        model.append(layers.ReLU())
        model.append(layers.Dropout(0.05))
        
        model.append(layers.Dense(tfp.layers.IndependentNormal.params_size(self.latent_dim), name="e5"))
        if return_mean:
                model.append(tfp.layers.IndependentNormal(self.latent_dim, convert_to_tensor_fn=tfp.distributions.Distribution.mean, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=self.kl_weight), name="e6"))
        else:
                model.append(tfp.layers.IndependentNormal(self.latent_dim, convert_to_tensor_fn=tfp.distributions.Distribution.sample, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=self.kl_weight), name="e6"))
        return keras.Sequential(model, name='encoder')
    
    
    def decoder(self):
        decoder=keras.Sequential()
        model=[layers.InputLayer(input_shape=(self.latent_dim,))]

        model.append(layers.Dense(200, name="d0"))
        model.append(layers.BatchNormalization())
#         model.append(layers.LeakyReLU())
        model.append(layers.ReLU())
        model.append(layers.Dropout(0.05))

        model.append(layers.Dense(200, name="d1"))
        model.append(layers.BatchNormalization())
#         model.append(layers.LeakyReLU())
        model.append(layers.ReLU())
        model.append(layers.Dropout(0.05))
        
        model.append(layers.Dense(200, name="d2"))
        model.append(layers.BatchNormalization())
#         model.append(layers.LeakyReLU())
        model.append(layers.ReLU())
        model.append(layers.Dropout(0.05))
        
        model.append(layers.Dense(200, name="d3"))
        model.append(layers.BatchNormalization())
#         model.append(layers.LeakyReLU())
        model.append(layers.ReLU())
        model.append(layers.Dropout(0.05))
        
        model.append(layers.Dense(200, name="d4"))
        model.append(layers.BatchNormalization())
#         model.append(layers.LeakyReLU())
        model.append(layers.ReLU())
        model.append(layers.Dropout(0.05))
        
        
        model.append(layers.Dense(self.dim_x, name="d5"))

        return keras.Sequential(model, name='decoder')
    
    def build_vae_keras_model(self):
        x_input = keras.Input(shape=self.dim_x)
        encoder = self.encoder()
        decoder = self.decoder()
#         encoder.summary()
#         decoder.summary()
        z = encoder(x_input)
        output=decoder(z)
        negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
        
        # compile VAE model
        model = keras.Model(inputs=x_input, outputs=output)
        model.compile(loss=custom_sigmoid_cross_entropy_loss_with_logits, optimizer=keras.optimizers.Adam(self.learning_rate))
        return model
    
    def build_vae_keras_model_mean(self):
        x_input = keras.Input(shape=self.dim_x)
        encoder = self.encoder(return_mean=1)
        decoder = self.decoder()
        z = encoder(x_input)
        output=decoder(z)
        negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
        
        # compile VAE model
        model = keras.Model(inputs=x_input, outputs=output)
        model.compile(loss=custom_sigmoid_cross_entropy_loss_with_logits, optimizer=keras.optimizers.Adam(self.learning_rate))
        return model
    

def freeze_layers_before(model:dVAE, target_layer_name:str):
    found_target = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            found_target = freeze_layers_before(layer, target_layer_name)
        if layer.name == target_layer_name:
            found_target = True
        if not found_target:
            layer.trainable = False
    return found_target

def unfreeze_all_layers(model :dVAE):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # Handle nested models
            unfreeze_all_layers(layer)
        layer.trainable = True