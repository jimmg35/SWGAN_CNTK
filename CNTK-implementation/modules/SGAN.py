from __future__ import print_function, division


# Import essential modules
import os
import cv2
import random
import numpy as np
from os import listdir
import matplotlib as mpl
from numpy.random import randn
from numpy.random import randint
import matplotlib.pyplot as plt



import cntk as C
from cntk.device import try_set_default_device, gpu
from cntk.learners import learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk.layers import Dense, ConvolutionTranspose2D
from cntk.layers import BatchNormalization, Convolution2D


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class SGAN():
    def __init__(self):
        self.img_width = 449
        self.img_height = 449
        self.channels = 3
        self.img_shape = (self.channels, self.img_width, self.img_height)
        self.num_classes = 4
        self.latent_shape = np.ones(100)
        
        self.generator_hype = {'lr':0.0002, 'momentum':0.5}
        self.discriminator_hype = {'lr':0.0002, 'momentum':0.5}
        self.classifier_hype = {'lr':0.0002, 'momentum':0.5}
        
        self.dataset_hype = {'val_size':0.3}
        self.dataset_schema = {'Rice':0, 'SugarCane':1, 'BananaTree':2, 'GreenOnion':3}
        self.x = []
        self.y_hot = []
        self.x_unlabeled = []
        
    def load_supervised_data(self, Data_Path):
        print("\n\n  Loading supervised dataset...")
        
        for i in listdir(Data_Path):
            class_name = i[:i.index('_')]
            self.y_hot.append(self.dataset_schema[class_name])
            
            # load images
            img = cv2.imread(os.path.join(Data_Path, i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
            self.x.append(np.array(img))
        
        self.x = np.array(self.x)
        self.x = np.moveaxis(self.x, 3, 1)
        
        self.y_hot = np.array(self.y_hot)
        self.y_hot = self.y_hot.reshape(-1, 1)
        onehotencoder = OneHotEncoder()
        self.y_hot = onehotencoder.fit_transform(self.y_hot).toarray()
        
        print("  X : {}".format(self.x.shape))
        print("  Y : {}".format(self.y_hot.shape))

    def load_unsupervised_data(self, Data_Path):
        print("\n\n  Loading unsupervised dataset...")
        
        for i in listdir(Data_Path):
            # load images
            img = cv2.imread(os.path.join(Data_Path, i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
            self.x_unlabeled.append(img)
        
        self.x_unlabeled = np.array(self.x_unlabeled)
        self.x_unlabeled = np.moveaxis(self.x_unlabeled, 3, 1)
        
        print("  X_unlabeled : {}\n\n".format(self.x_unlabeled.shape))

    def custom_activation(self, input_tensor):
        logexpsum = C.reduce_sum(C.exp(input_tensor), 0)
        result = logexpsum / (logexpsum + 1.0)
        return result
 
    def leaky_relu(self, input_tensor, leak=0.2):
        output_tensor = C.param_relu(C.constant((np.ones(input_tensor.shape)*leak).astype(np.float32)), input_tensor)
        return output_tensor
    
    def build_generator(self, input_tensor):
        with C.layers.default_options(init=C.normal(scale=0.02)):
            print("\n\n  ************************")
            print("  *  Building Generator  *")
            print("  ************************")
            print("  Input shape : {}".format(input_tensor.shape))
            
            # 15*15*256
            L1_1 = Dense(15*15*256)(input_tensor)
            L1_2 = BatchNormalization(map_rank=1)(L1_1)
            L1_3 = self.leaky_relu(L1_2)
            L1_4 = C.reshape(L1_3, (256, 15, 15))
            
            # 29*29*128
            L2_1 = ConvolutionTranspose2D((3,3), 128, strides=2, pad=True)(L1_4)
            L2_2 = BatchNormalization(map_rank=1)(L2_1)
            L2_3 = self.leaky_relu(L2_2)
            
            # 57*57*64
            L3_1 = ConvolutionTranspose2D((3,3), 64, strides=2, pad=True)(L2_3)
            L3_2 = BatchNormalization(map_rank=1)(L3_1)
            L3_3 = self.leaky_relu(L3_2)
            
            # 113*113*32
            L4_1 = ConvolutionTranspose2D((3,3), 32, strides=2, pad=True)(L3_3)
            L4_2 = BatchNormalization(map_rank=1)(L4_1)
            L4_3 = self.leaky_relu(L4_2)   
            
            # 225*225*16
            L5_1 = ConvolutionTranspose2D((3,3), 16, strides=2, pad=True)(L4_3)
            L5_2 = BatchNormalization(map_rank=1)(L5_1)
            L5_3 = self.leaky_relu(L5_2)
            
            # 449*449*8
            L6_1 = ConvolutionTranspose2D((3,3), 8, strides=2, pad=True)(L5_3)
            L6_2 = BatchNormalization(map_rank=1)(L6_1)
            L6_3 = self.leaky_relu(L6_2)
            
            # 449*449*3
            z = Convolution2D((3,3), 3, pad=True, strides=(1,1), activation=C.tanh)(L6_3)
            print("  Output shape : {}".format(z.shape))
            
            return z
        
    def build_discriminator(self, input_tensor):
        with C.layers.default_options(init=C.normal(scale=0.02)):
            print("\n\n  *****************************************")
            print("  *  Building Discriminator / Classifier  *")
            print("  *****************************************")
            print("  Input shape : {}".format(input_tensor.shape))

            # 225*225*8
            L1_1 = Convolution2D((3,3), 8, strides=2, pad=True)(input_tensor)
            L1_2 = BatchNormalization(map_rank=1)(L1_1)
            L1_3 = self.leaky_relu(L1_2)
            
            # 113*113*16
            L2_1 = Convolution2D((3,3), 16 ,strides=2, pad=True)(L1_3)
            L2_2 = BatchNormalization(map_rank=1)(L2_1)
            L2_3 = self.leaky_relu(L2_2)
            
            # 57*57*32
            L3_1 = Convolution2D((3,3), 32 ,strides=2, pad=True)(L2_3)
            L3_2 = BatchNormalization(map_rank=1)(L3_1)
            L3_3 = self.leaky_relu(L3_2)
            
            # 29*29*64
            L4_1 = Convolution2D((3,3), 64 ,strides=2, pad=True)(L3_3)
            L4_2 = BatchNormalization(map_rank=1)(L4_1)
            L4_3 = self.leaky_relu(L4_2)
            
            # 15*15*128
            L5_1 = Convolution2D((3,3), 128 ,strides=2, pad=True)(L4_3)
            L5_2 = BatchNormalization(map_rank=1)(L5_1)
            L5_3 = self.leaky_relu(L5_2)
            
            shared_node = Dense(self.num_classes)(L5_3)
            C_out_layer = C.softmax(shared_node)
            D_out_layer = self.custom_activation(shared_node)
            
            print("  Classifier output shape : {}".format(C_out_layer.shape))
            print("  Discriminator output shape : {}\n\n".format(D_out_layer.shape))
            
            return C_out_layer, D_out_layer
            
    def compile(self, visualize = True):
        
        '''Construct four models, 
            d_real, c_model, g_fake for seperate model
            d_fake for combined gan.
        '''
        
        input_dynamic_axes = [C.Axis.default_batch_axis()]
        
        # Generator input tensor
        latent = C.input_variable(self.latent_shape.shape, dynamic_axes=input_dynamic_axes)
        
        # Discriminator real input tensor
        x_d_real = C.input_variable(self.img_shape, dynamic_axes=input_dynamic_axes)
        x_d_real_scaled = 2 * (x_d_real / 255.0) - 1.0
        
        # Classifier output
        c_out_tensor = C.input_variable((self.num_classes), dynamic_axes=input_dynamic_axes)
        
        # Build generator
        g_fake = self.build_generator(latent)
        
        # Build discriminator / classifier
        c_model, d_real = self.build_discriminator(x_d_real_scaled)
        
        # Combine generator and discriminator
        d_fake = d_real.clone(
            method = 'share',
            substitutions = {x_d_real_scaled.output: g_fake.output}
        )
        
        # Compile the loss function(Wasserstein GAN loss function)
        G_loss = -d_fake
        D_loss = -d_real + d_fake
        C_loss = C.cross_entropy_with_softmax(c_model, c_out_tensor)
        
        # Learner
        G_learner = C.adam(
            parameters = g_fake.parameters,
            lr = learning_rate_schedule(self.generator_hype['lr'], UnitType.sample),
            momentum = momentum_as_time_constant_schedule(self.generator_hype['momentum'])
        )
        
        D_learner = C.adam(
            parameters = d_real.parameters,
            lr = learning_rate_schedule(self.discriminator_hype['lr'], UnitType.sample),
            momentum = momentum_as_time_constant_schedule(self.discriminator_hype['momentum'])
        )
        
        C_learner = C.adam(
            parameters = c_model.parameters,
            lr = learning_rate_schedule(self.classifier_hype['lr'], UnitType.sample),
            momentum = momentum_as_time_constant_schedule(self.classifier_hype['momentum'])
        )
        
        # Trainer
        G_Trainer = C.Trainer(
            g_fake,
            (G_loss, None),
            G_learner
        )
        
        D_Trainer = C.Trainer(
            d_real,
            (D_loss, None),
            D_learner
        )
        
        C_Trainer = C.Trainer(
           c_model,
           (C_loss, None),
           C_learner
        )
        
        #d_real, c_model, g_fake
        if visualize:
            C.logging.graph.plot(d_real, 'discriminator.png')
            C.logging.graph.plot(c_model, 'classifier.png')
            C.logging.graph.plot(g_fake, 'generator.png')
            C.logging.graph.plot(d_fake, 'combined.png')
        
        return x_d_real, g_fake, latent, G_Trainer, D_Trainer, C_Trainer
        
    def train(self, epoch, batch_size = 32):
        
        
        x_d_real, g_fake, latent, G_Trainer, D_Trainer, C_Trainer = self.compile()
        
            
            
        