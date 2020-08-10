from modules.SGAN import SGAN
import numpy as np




if __name__ == '__main__':
    sgan = SGAN()
    sgan.load_supervised_data(r'C:\Users\Jim\Desktop\dataset\labeled')
    sgan.load_unsupervised_data(r'C:\Users\Jim\Desktop\dataset\unlabeled')
    sgan.compile(visualize = False)