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
import cntk.io.transforms as xforms


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
np.random.seed(123)


class SGAN():
    def __init__(self):
        self.img_width = 27
        self.img_height = 27
        self.channels = 3
        self.img_shape = (self.channels, self.img_width, self.img_height)
        self.num_classes = 10
        self.latent_shape = np.ones(100)
        
        self.generator_hype = {'lr':0.00001, 'momentum':0}
        self.discriminator_hype = {'lr':0.00001, 'momentum':0}
        self.classifier_hype = {'lr':0.00001, 'momentum':0}
        
        self.dataset_hype = {'val_size':0.3}
        self.dataset_schema = {'Rice':0, 'SugarCane':1, 'BananaTree':2, 'GreenOnion':3}
        self.x = []  # Labeled Data
        self.y_hot = [] # Label onehot-encoded
        self.x_unlabeled = [] # Unlabeled Data
        
        self.D_loss_list = []
        self.G_loss_list = []
        self.C_loss_list = []

    
    def noise_sample(self, num_samples):
        return np.random.uniform(
            low = -1.0,
            high = 1.0,
            size = [num_samples, 100]        
        ).astype(np.float32)

    def create_supervised_reader(self, map_file):
        transforms = []
        transforms += [xforms.scale(width=self.img_width, height=self.img_height, channels=self.channels, interpolations='nearest')]
        
        self.reader_supervised_train = C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        features = C.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = C.io.StreamDef(field='label', shape=self.num_classes))))
        
        with open(r'C:\Users\Jim\Desktop\dataset\map_su.txt', mode='r') as f:
            data_path = [i for i in f]
            self.superivsed_data_size = len(data_path)
        
    def create_unsupervised_reader(self, map_file):
        transforms = []
        transforms += [xforms.scale(width=self.img_width, height=self.img_height, channels=self.channels, interpolations='nearest')]
        
        self.reader_unsupervised_train = C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        features = C.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = C.io.StreamDef(field='label', shape=self.num_classes))))
        
        with open(r'C:\Users\Jim\Desktop\dataset\map_un.txt', mode='r') as f:
            data_path = [i for i in f]
            self.unsuperivsed_data_size = len(data_path)

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
    
    def measure_loss(self, data_x, data_y, x, y, trainer, minibatch_size):
        errors = []
        for i in range(0, int(len(data_x) / minibatch_size)):
            data_sx, data_sy = slice_minibatch(data_x, data_y, i, minibatch_size)
            errors.append(trainer.test_minibatch({x: data_sx, y: data_sy}))
        return np.mean(errors)
    
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
    
    def build_generator_mnist(self, input_tensor):
        with C.layers.default_options(init=C.normal(scale=0.02)):
            print("\n\n  ************************")
            print("  *  Building Generator For Mnist!  *")
            print("  ************************")
            print("  Input shape : {}".format(input_tensor.shape))
            
            # 14*14*256
            L1_1 = Dense(14*14*8)(input_tensor)
            L1_2 = BatchNormalization(map_rank=1)(L1_1)
            L1_3 = self.leaky_relu(L1_2)
            L1_4 = C.reshape(L1_3, (8, 14, 14))
            
            # 27*27*128
            L2_1 = ConvolutionTranspose2D((3,3), 64, strides=2, pad=True)(L1_4)
            L2_2 = BatchNormalization(map_rank=1)(L2_1)
            L2_3 = self.leaky_relu(L2_2)
            
            # 3*27*27
            z = Convolution2D((3,3), 3, pad=True, strides=(1,1), activation=C.tanh)(L2_3)
            print("  Output shape : {}".format(z.shape))
            
            return z
    
    def build_discriminator_mnist(self, input_tensor):
        with C.layers.default_options(init=C.normal(scale=0.02)):
            print("\n\n  *****************************************")
            print("  *  Building Discriminator / Classifier  *")
            print("  *****************************************")
            print("  Input shape : {}".format(input_tensor.shape))

            L1_1 = Convolution2D((3,3), 64, strides=1, pad=True)(input_tensor)
            L1_2 = BatchNormalization(map_rank=1)(L1_1)
            L1_3 = self.leaky_relu(L1_2)
            
            shared_node = Dense(self.num_classes)(L1_3)
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
        g_fake = self.build_generator_mnist(latent)
        
        # Build discriminator / classifier
        c_model, d_real = self.build_discriminator_mnist(x_d_real_scaled)
        
        # Combine generator and discriminator
        d_fake = d_real.clone(
            method = 'share',
            substitutions = {x_d_real_scaled.output: g_fake.output}
        )
        
        # Compile the loss function(Wasserstein GAN loss function)
        G_loss = 1.0 - C.log(d_fake)
        D_loss = -(C.log(d_real) + C.log(1.0 - d_fake))
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
        
        return x_d_real, g_fake, latent, G_Trainer, D_Trainer, C_Trainer, c_out_tensor
        
    def train(self, epoch, batch_size = 2, k = 1):

        '''shape = self.x[0].shape
        data_size = self.x.shape[0]
        
        # Split data
        test_portion = int(data_size * 0.3)
        indices = np.random.permutation(data_size)
        test_indices = indices[:test_portion]
        training_indices = indices[test_portion:]

        validation_data = (self.x[test_indices], self.y_hot[test_indices])
        training_data = (self.x[training_indices], self.y_hot[training_indices])

        print("  Splited Training Data : {}".format(training_data[0].shape))
        print("  Splited Validation Data : {}".format(validation_data[0].shape))
        print("  Splited Training GT : {}".format(training_data[1].shape))
        print("  Splited Validation GT : {}".format(validation_data[1].shape))
        print("  Unlabeled data shape : {}\n\n".format(self.x_unlabeled.shape))
        
        # Fake label 
        valid = np.ones((batch_size//2, 1))
        fake = np.zeros((batch_size//2, 1))'''
        
        # Calculate step_per_epoch
        step_per_epoch_su = int(self.superivsed_data_size / batch_size)
        step_per_epoch_un = int(self.unsuperivsed_data_size / batch_size)
        
        # Loading models
        x_d_real, g_fake, latent, G_Trainer, D_Trainer, C_Trainer, c_out_tensor = self.compile(visualize=False)

        
        input_map_C = {x_d_real:self.reader_supervised_train.streams.features, c_out_tensor:self.reader_supervised_train.streams.labels}
        input_map_D = {x_d_real:self.reader_unsupervised_train.streams.features}
        
        # Start training
        for e in range(0, epoch):
            c_total_loss = 0
            d_total_loss = 0
            g_total_loss = 0
            for s in range(0, step_per_epoch_su):
                # Update classifier
                X_data = self.reader_supervised_train.next_minibatch(batch_size, input_map_C)
                batch_inputs = {x_d_real: X_data[x_d_real].data, c_out_tensor: X_data[c_out_tensor].data}
                C_Trainer.train_minibatch(batch_inputs)
                c_total_loss += C_Trainer.previous_minibatch_loss_average
    
            for s in range(0, step_per_epoch_un):
                # Update discriminator
                for dk in range(k):
                    Z_data = self.noise_sample(batch_size)
                    X_data = self.reader_unsupervised_train.next_minibatch(batch_size, input_map_D)
                    if X_data[x_d_real].num_samples == Z_data.shape[0]:
                        batch_inputs = {x_d_real: X_data[x_d_real].data, latent: Z_data}
                        D_Trainer.train_minibatch(batch_inputs)
                        d_total_loss += D_Trainer.previous_minibatch_loss_average
                # Update generator
                Z_data = self.noise_sample(2 * batch_size)
                batch_inputs = {latent: Z_data}
                G_Trainer.train_minibatch(batch_inputs)
                g_total_loss += G_Trainer.previous_minibatch_loss_average
                
            self.D_loss_list.append(d_total_loss/step_per_epoch_un)
            self.G_loss_list.append(g_total_loss/step_per_epoch_un)
            self.C_loss_list.append(c_total_loss/step_per_epoch_su)
            print(" Epoch {} | D_loss:{} G_loss:{} C_loss:{}".format(e, d_total_loss/step_per_epoch_un, g_total_loss/step_per_epoch_un, c_total_loss/step_per_epoch_su))
            print(d_total_loss)
            
    def plot_classifier_results(self):
        plt.figure("Classifier Loss")
        mpl.style.use('seaborn')
        epo_list = [i for i in range(len(self.C_loss_list))]
        plt.plot(epo_list, self.C_loss_list, color = 'red', label='C_model Loss')
        plt.legend()
        plt.savefig('ClassifierLoss_{}'.format(len(epo_list)))
        plt.show()
    
    def plot_gan_results(self):
        plt.figure("GAN Loss")
        mpl.style.use('seaborn')
        epo_list = [i for i in range(len(self.D_loss_list))]
        plt.plot(epo_list, self.D_loss_list, color = 'green', label='D_model Loss')
        plt.plot(epo_list, self.G_loss_list, color = 'blue', label='G_model Loss')
        plt.legend()
        plt.savefig('GANLoss_{}'.format(len(epo_list)))
        plt.show()
                    
                

                 
                
            
        