import pickle

open_path = ('C:/Users/ouyuming/Desktop/pyProjects/allFile/exercise/test.imdb')
f = open(open_path,'rb')
info = pickle.load(f)
print(info)