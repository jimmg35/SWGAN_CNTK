from modules.SGAN import SGAN
import numpy as np




if __name__ == '__main__':
    sgan = SGAN()
    
    #sgan.load_supervised_data(r'C:\Users\Jim\Desktop\dataset\labeled')
    #sgan.load_unsupervised_data(r'C:\Users\Jim\Desktop\dataset\unlabeled')
    sgan.create_supervised_reader(r'C:\Users\Jim\Desktop\dataset\map_su.txt')
    sgan.create_unsupervised_reader(r'C:\Users\Jim\Desktop\dataset\map_un.txt')
    sgan.train(100, batch_size=32)
    sgan.plot_classifier_results()
    sgan.plot_gan_results()
    