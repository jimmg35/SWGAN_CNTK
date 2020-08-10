
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


# Import Keras related modules
from keras import losses
from keras import backend
import keras.backend as K
from keras import optimizers
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, multiply, GaussianNoise
from keras.preprocessing.image import ImageDataGenerator

# Import sklearn tools
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split



class SGAN:
    def __init__(self):
        self.img_rows = 240
        self.img_cols = 240
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 4
        self.latent_dim = 100
        self.random_state_test = 42
        self.random_state_val = 43
        self.data_aug = True
        

        self.d_loss_real_list = []
        self.d_loss_fake_list = []
        self.g_loss_list = []
        self.c_loss_list = []
        self.c_acc_list = []
        self.c_loss_val_list = []
        self.c_acc_val_list = []

        self.c_aug_loss_list = []
        self.c_aug_acc_list = []
        self.c_aug_valloss_list = []
        self.c_aug_valacc_list = []
        self.a_aug_loss_list = []
        self.a_aug_acc_list = []
        self.a_aug_valloss_list = []
        self.a_aug_valacc_list = []
 
        optimizer = Adam(0.0002, 0.5)
 
        # Build and compile the discriminator
        self.discriminator, self.classifier = self.build_discriminator()

        # Build and compile the StandAloneClassifier
        self.AloneClassifier = self.build_aloneClassifier()

        self.discriminator.summary()
        self.classifier.summary()
 
        # Build the generator
        self.generator = self.build_generator()
 
        # The generator takes noise as input and generates imgs
        noise = Input(shape=(100,))
        img = self.generator(noise)
 
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
 
        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)
 
        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)
    
    def load_supervied_data(self):

        
        test_size = 0.2
        DATA_PATH = r'C:\Users\iadc\GAN_test\Dataset\Train'
        dict_hot = {'Rice':0,
            'SugarCane':1,
            'BananaTree':2,
            'GreenOnion':3}
        label = []
        label_hot = []
        images = []

        for i in listdir(DATA_PATH):
            # load the target
            class_name = i[:i.index('_')]
            label.append(class_name)
            label_hot.append(dict_hot[class_name])
            
            # load images
            img = cv2.imread(os.path.join(DATA_PATH, i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_rows,self.img_cols), interpolation=cv2.INTER_LINEAR)
            images.append(np.array(img))
        #Convert list to numpy array
        images = np.array(images)
        label_hot = np.array(label_hot)
        print('-----Supervised data loading complete-----')

        (supData, testData, supLabels, testLabels) = train_test_split(images, label_hot, test_size = test_size, random_state = self.random_state_test)
        (supData, valData, supLabels, valLabels) = train_test_split(supData, supLabels, test_size = 0.2, random_state = self.random_state_val)
        return supData, valData, supLabels, valLabels

    def load_unsupervised_data(self):

        images = []
        TYPE = ['fruit_dirty','sugarcane_dirty','rice_dirty','veg_dirty']
        DATA_PATH = r'C:\Users\iadc\Jim_CropImage\Dataset'
        
        for i in TYPE:
            TYPE_PATH = os.path.join(DATA_PATH, i)
            for j in listdir(TYPE_PATH)[:1600]:

                img = cv2.imread(os.path.join(TYPE_PATH, j))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_rows,self.img_cols), interpolation=cv2.INTER_LINEAR)
                images.append(np.array(img))

        images = np.array(images)
        np.random.shuffle(images)
        print('-----Unsupervised data loading complete-----')

        return images

    def select_supervised_samples(self, dataset, batch_s, n_classes):

        X, y = dataset
        X_list, y_list = list(), list()
        n_per_class = int(batch_s / n_classes)
        for i in range(n_classes):
            # get all images for this class
            X_with_class = X[y == i]
            # choose random instances
            ix = randint(0, len(X_with_class), n_per_class)
            # add to list
            [X_list.append(X_with_class[j]) for j in ix]
            [y_list.append(i) for j in ix]
        X_list_a = np.array(X_list)
        Y_list_a = np.array(y_list)
        
        
        np.random.shuffle(X_list_a)
        np.random.shuffle(Y_list_a)

        return X_list_a, Y_list_a

    def custom_activation(self, output):
        logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
        result = logexpsum / (logexpsum + 1.0)
        return result

    def build_generator(self):

        # image generator input
        in_lat = Input(shape=(self.latent_dim,))
        # foundation for 10*10 image
        
        gen = Dense(15*15*128)(in_lat)
        gen = LeakyReLU(alpha = 0.2)(gen)
        gen = Reshape((15,15,128))(gen)

        #30
        gen = UpSampling2D()(gen)
        gen = Conv2D(256, (3,3), strides = (1,1), padding = 'same')(gen)
        gen = LeakyReLU(alpha = 0.2)(gen)    
        gen = BatchNormalization(momentum = 0.8)(gen)   

        #60
        gen = UpSampling2D()(gen)
        gen = Conv2D(128, (3,3), strides = (1,1), padding = 'same')(gen)
        gen = LeakyReLU(alpha = 0.2)(gen)
        gen = BatchNormalization(momentum = 0.8)(gen)  

        #120
        gen = UpSampling2D()(gen)
        gen = Conv2D(64, (3,3), strides = (1,1), padding = 'same')(gen)
        gen = LeakyReLU(alpha = 0.2)(gen) 
        gen = BatchNormalization(momentum = 0.8)(gen)

        #240
        gen = UpSampling2D()(gen)
        gen = Conv2D(32, (3,3), strides = (1,1), padding = 'same')(gen)
        gen = LeakyReLU(alpha = 0.2)(gen)
        gen = BatchNormalization(momentum = 0.8)(gen)


        gen = Conv2D(self.channels, (3,3), strides = (1,1), padding = 'same')(gen)
        out_layer = Activation('tanh')(gen)


        # define model
        model = Model(in_lat, out_layer)
        return model

    def build_discriminator(self):
                     
        # image input
        in_image = Input(shape=self.img_shape)

        fe = Conv2D(32, (3,3), strides=(2,2), padding='same')(in_image)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = BatchNormalization(momentum=0.8)(fe)
        
        fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = BatchNormalization(momentum=0.8)(fe)
        
        fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = BatchNormalization(momentum=0.8)(fe)
        
        fe = Conv2D(256, (3,3), strides=(2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = BatchNormalization(momentum=0.8)(fe)
        

        fe = Flatten()(fe)
            
        # output layer nodes
        fe = Dense(self.num_classes)(fe)
        # supervised output
        c_out_layer = Activation('softmax')(fe)
        # define and compile supervised discriminator model
        c_model = Model(in_image, c_out_layer)
        c_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
        # unsupervised output
        d_out_layer = Lambda(self.custom_activation)(fe)
        # define and compile unsupervised discriminator model
        d_model = Model(in_image, d_out_layer)
        d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return d_model, c_model

    def build_aloneClassifier(self):

        # image input
        in_image = Input(shape=self.img_shape)

        fe = Conv2D(32, (3,3), strides=(2,2), padding='same')(in_image)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = BatchNormalization(momentum=0.8)(fe)
        
        fe = Conv2D(64, (3,3), strides=(2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = BatchNormalization(momentum=0.8)(fe)
        
        fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = BatchNormalization(momentum=0.8)(fe)
        
        fe = Conv2D(256, (3,3), strides=(2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = BatchNormalization(momentum=0.8)(fe)
        

        fe = Flatten()(fe)
            
        # output layer nodes
        fe = Dense(self.num_classes)(fe)
        # supervised output
        c_out_layer = Activation('softmax')(fe)
        # define and compile supervised discriminator model
        c_model = Model(in_image, c_out_layer)
        c_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
        return c_model

    def train(self, epoch, BatchSize=32, sample_interval=50):

        # Load supervised dataset
        supData, valData, supLabels, valLabels = self.load_supervied_data()
 
        # Load unsupervised dataset
        unsData = self.load_unsupervised_data()
 
        # Rescale -1 to 1
        supData = (supData.astype(np.float32) - 127.5) / 127.5
        valData = (valData.astype(np.float32) - 127.5) / 127.5
        unsData = (unsData.astype(np.float32) - 127.5) / 127.5

        # Calculate the stpes
        len_dataset = supData.shape[0] + unsData.shape[0]
        step_per_epoch = int(len_dataset / BatchSize)
        n_steps = step_per_epoch * epoch

        # One-Hot encoding
        supLabels_gen = supLabels.reshape(-1,1)
        supLabels_gen = to_categorical(supLabels_gen)
        valLabels = valLabels.reshape(-1, 1)
        valLabels = to_categorical(valLabels)

        # Define ImageDataGenerator(No need to recale)
        datagen = ImageDataGenerator()
        datagen.fit(supData)
        train_generator = datagen.flow(supData, supLabels_gen, shuffle = True, batch_size = BatchSize)
        val_generator = datagen.flow(valData, valLabels, shuffle = True, batch_size = BatchSize)
 


        # Class weights:
        # To balance the difference in occurences of digit class labels.
        # 50% of labels that the discriminator trains on are 'fake'.
        # Weight = 1 / frequency
        half_batch = BatchSize // 2
        # cw1 = {0: 1, 1: 1}
        # cw2 = {}
        # class_w = class_weight.compute_class_weight('balanced', np.unique(supLabels), supLabels)
        # for i,j in zip([0,1,2,3],class_w):
        #     cw2[i] = j

        

        # Adversarial ground truths
        valid_gan = np.ones((BatchSize, 1))
        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))
        
        e_count = 0
        for step in range(n_steps):
 
            # Select a batch of supervised image(with labels)
            supData_subset, labels_sup_subset = self.select_supervised_samples([supData, supLabels], BatchSize, self.num_classes)
            labels_sup_subset = labels_sup_subset.reshape(-1, 1)
            labels_sup_subset = to_categorical(labels_sup_subset)

            # Select a batch of unsupervised image(with no labels)
            idx2 = np.random.randint(0, unsData.shape[0], half_batch)
            unsData_subset = unsData[idx2]

            # Sample noise and generate a batch of new images
            noise_gan = np.random.normal(0, 1, (BatchSize, self.latent_dim))
            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
            gen_imgs = self.generator.predict(noise)


            # ---------------------
            #  Update Classifier (Supervised mode)
            # ---------------------
            #c_loss, c_acc = self.classifier.train_on_batch(supData_subset, labels_sup_subset) #, class_weight = cw2
            
            # ---------------------
            #  Update Discriminator (Unsupervised mode)
            # ---------------------
            d_loss_real = self.discriminator.train_on_batch(unsData_subset, valid) #, class_weight = cw1
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake) #, class_weight = cw1

            # ---------------------
            #  Update Generator
            # ---------------------
            g_loss = self.combined.train_on_batch(noise_gan, valid_gan) #, class_weight = cw1

            # ---------------------
            #  Update Classifier using ImageDataGenerator
            # ---------------------
            if (step+1) % step_per_epoch == 0:
                print("=================== Augmentation training start ===================")
                train_history_classifier = self.classifier.fit_generator(train_generator,
                                            steps_per_epoch = step_per_epoch*3, epochs = 1,
                                            validation_data = val_generator, validation_steps = step_per_epoch*1)

                train_history_AloneClassifier = self.AloneClassifier.fit_generator(train_generator,
                                            steps_per_epoch = step_per_epoch*3, epochs = 1,
                                            validation_data = val_generator, validation_steps = step_per_epoch*1)

                self.c_aug_loss_list.append(train_history_classifier.history['loss'][0])
                self.c_aug_acc_list.append(train_history_classifier.history['acc'][0])
                self.c_aug_valloss_list.append(train_history_classifier.history['val_loss'][0])
                self.c_aug_valacc_list.append(train_history_classifier.history['val_acc'][0])

                self.a_aug_loss_list.append(train_history_AloneClassifier.history['loss'][0])
                self.a_aug_acc_list.append(train_history_AloneClassifier.history['acc'][0])
                self.a_aug_valloss_list.append(train_history_AloneClassifier.history['val_loss'][0])
                self.a_aug_valacc_list.append(train_history_AloneClassifier.history['val_acc'][0])
                e_count += 1
                

            # ---------------------
            #  Validate Classifier
            # ---------------------
            #val_loss, val_acc = self.classifier.evaluate(valData, valLabels, verbose = 0, batch_size = 32)
 
            # Plot the progress
            # , c_loss, c_acc*100, val_loss, val_acc*100
            # C[Loss:{:.4f} Acc:{:.2f}% Val_Loss:{:.4f} Val_Acc:{:.2f}%]   
            print('>{} D[R_Loss:{:.4f} F_Loss:{:.4f}]   G[Loss:{:.4f}]'.format(step, d_loss_real, d_loss_fake, g_loss))
            

            # Record the results
            self.d_loss_real_list.append(d_loss_real)
            self.d_loss_fake_list.append(d_loss_fake)
            self.g_loss_list.append(g_loss)
            #self.c_loss_list.append(c_loss)
            #self.c_acc_list.append(c_acc)
            #self.c_loss_val_list.append(val_loss)
            #self.c_acc_val_list.append(val_acc)
            
            
            # save the result of every epoch
            if (step+1) % step_per_epoch == 0:
                print('Saving result of {} epoch'.format((step+1) / step_per_epoch))
                self.sample_images((step+1) / step_per_epoch)
                self.save_model((step+1) / step_per_epoch)
                self.TrainHistory_GAN(self.d_loss_real_list, self.d_loss_fake_list, self.g_loss_list, (step+1) / step_per_epoch)
                #self.TrainHistory_classifier(self.c_loss_list, self.c_acc_list, self.c_loss_val_list, self.c_acc_val_list, step)
                self.TrainHistory_aug_val(self.c_aug_valacc_list, self.a_aug_valacc_list, (step+1) / step_per_epoch)
                self.TrainHistory_aug_train(self.c_aug_acc_list, self.a_aug_acc_list, (step+1) / step_per_epoch)    

    def sample_images(self, step):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
 
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
 
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(r"C:\Users\iadc\GAN_test\results\Fig\mnist_%d.png" % step)
        plt.close()
    
    def save_model(self,step):

        self.generator.save(r"C:\Users\iadc\GAN_test\results\Generator\{}_{}.h5".format('G_model',step))
        self.discriminator.save(r"C:\Users\iadc\GAN_test\results\Discriminator\{}_{}.h5".format('D_model',step))
        self.classifier.save(r"C:\Users\iadc\GAN_test\results\Classifier\{}_{}.h5".format('C_model',step))
        self.AloneClassifier.save(r"C:\Users\iadc\GAN_test\results\StandAlone\{}_{}.h5".format('StandAlone',step))

    def TrainHistory_GAN(self, d_loss_real_list, d_loss_fake_list, g_loss_list, step):
        
        plt.figure('GAN' + str(step))
        mpl.style.use('seaborn')
        folder_path = r'C:\Users\iadc\GAN_test\results\trainHistory\GAN'
        file_name = 'trainHistory_GAN_{}step.png'.format(step)
        step_list = [i for i in range(len(d_loss_real_list))]
        plt.plot(step_list, d_loss_real_list, color = 'blue', label = 'd_loss_real')
        plt.plot(step_list, d_loss_fake_list, color = 'green', label = 'd_loss_fake')
        plt.plot(step_list, g_loss_list, color = 'red', label = 'g_loss')
        plt.legend()
        plt.savefig(os.path.join(folder_path,file_name)) 
        plt.cla()
        plt.close("all")

    def TrainHistory_classifier(self, c_loss_list, c_acc_list, c_loss_val_list, c_acc_val_list, step):

        plt.figure('Classifier' + str(step))
        mpl.style.use('seaborn')
        folder_path = r'C:\Users\iadc\GAN_test\results\trainHistory\Classifier'
        file_name = 'trainHistory_C_{}step.png'.format(step)
        step_list = [i for i in range(len(c_loss_list))]
        plt.plot(step_list, c_loss_list, color = 'red', label = 'loss')
        plt.plot(step_list, c_acc_list, color = 'blue', label = 'acc')
        plt.plot(step_list, c_loss_val_list, color = 'orange', label = 'val_loss')
        plt.plot(step_list, c_acc_val_list, color = 'green', label = 'val_acc')
        plt.legend()
        plt.savefig(os.path.join(folder_path,file_name))
        plt.cla()
        plt.close("all")

    def TrainHistory_aug_val(self, c_aug_valacc_list, a_aug_valacc_list, e_count):
        plt.figure('Classifier_val' + str(e_count))
        mpl.style.use('seaborn')
        folder_path = r'C:\Users\iadc\GAN_test\results\trainHistory\Classifier'
        file_name = 'trainHistory_C_val_{}step.png'.format(e_count)
        step_list = [i for i in range(len(c_aug_valacc_list))]
        plt.plot(step_list, c_aug_valacc_list, color = 'red', label = 'SGAN With Augmentation')
        plt.plot(step_list, a_aug_valacc_list, color = 'green', label = 'Stand Alone Classifier With Augmentation')
        plt.legend()
        plt.savefig(os.path.join(folder_path,file_name))
        plt.cla()
        plt.close("all")

    def TrainHistory_aug_train(self, c_aug_acc_list, a_aug_acc_list, e_count):
        plt.figure('Classifier_train' + str(e_count))
        mpl.style.use('seaborn')
        folder_path = r'C:\Users\iadc\GAN_test\results\trainHistory\Classifier'
        file_name = 'trainHistory_C_train_{}step.png'.format(e_count)
        step_list = [i for i in range(len(c_aug_acc_list))]
        plt.plot(step_list, c_aug_acc_list, color = 'red', label = 'SGAN With Augmentation')
        plt.plot(step_list, a_aug_acc_list, color = 'green', label = 'Stand Alone Classifier With Augmentation')
        plt.legend()
        plt.savefig(os.path.join(folder_path,file_name))
        plt.cla()
        plt.close("all")
    