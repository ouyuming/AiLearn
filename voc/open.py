import pickle

open_path = ('C:/Users/ouyuming/Desktop/pyProjects/voc/file/exercise/test.imdb')
f = open(open_path,'rb')
info = pickle.load(f)
print(info)