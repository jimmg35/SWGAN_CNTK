import os 
from os import listdir

data_path = r'C:\Users\Jim\Desktop\dataset\unlabeled'
c = 0
for i in listdir(data_path):
    os.rename(os.path.join(data_path,i), os.path.join(data_path,"unlabeled_{}.png".format(c)))
    c += 1