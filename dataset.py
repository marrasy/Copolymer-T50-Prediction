import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
#
from torch_geometric.data import Data

def load_6_structure_data(train_ratio=0.8, nead_norm=True):
    DF_MRI = pd.read_excel(open('./data/Our Dataset.xlsx', 'rb'),
              sheet_name='Data organized fluoro-monomer')
    
    Flag = [i != 'X' for i in DF_MRI['T50']]
    X=pd.DataFrame([
        [0, 0, 1, 3, 1, 0, 0, 2, 2, 2, 0, 1],
        [0, 1, 0, 1, 1, 0, 0, 1, 2, 3, 0, 1],
        [0, 1, 0, 4, 1, 0, 0, 3, 2, 7, 0, 1],
        [1, 1, 0, 4, 1, 0, 0, 5, 4, 7, 0, 1],
        [0, 0, 1, 5, 1, 1, 0, 6, 3, 11, 1, 1],
        [0, 0, 0, 2, 1, 1, 1, 3, 2, 8, 0, 1]
    ])
    Mix_X_6Block = []
    for i in range(len(DF_MRI[Flag])):
            Sequency_X = np.zeros((6, 12))
            for j in range(6):
                Sequency_X[j]=X.iloc[j]*DF_MRI.iloc[i,j]
            Mix_X_6Block.append(Sequency_X)

    Mix_X_6Block = np.array(Mix_X_6Block)
    Mix_Y = np.array(DF_MRI[Flag]['T50'])
    if nead_norm:
        np.savez('./data/6Block_mean_std.npz', x_mean=Mix_X_6Block.mean(), x_std=Mix_X_6Block.std(), y_mean=Mix_Y.mean(), y_std=Mix_Y.std())
        Mix_X_6Block = (Mix_X_6Block - Mix_X_6Block.mean()) / Mix_X_6Block.std()
        Mix_Y = (Mix_Y - Mix_Y.mean()) / Mix_Y.std()

    x_train, x_test, y_train, y_test = train_test_split(Mix_X_6Block, Mix_Y, train_size=train_ratio, random_state=42)
    return x_train, x_test, y_train, y_test

def load_100_structure_data(train_ratio=0.8, need_norm=True):
    DF_MRI = pd.read_excel(open('./data/Our Dataset.xlsx', 'rb'),
              sheet_name='Data organized fluoro-monomer') 
    X = pd.DataFrame([
        [0, 0, 1, 3, 1, 0, 0, 2, 2, 2, 0, 1],
        [0, 1, 0, 1, 1, 0, 0, 1, 2, 3, 0, 1],
        [0, 1, 0, 4, 1, 0, 0, 3, 2, 7, 0, 1],
        [1, 1, 0, 4, 1, 0, 0, 5, 4, 7, 0, 1],
        [0, 0, 1, 5, 1, 1, 0, 6, 3, 11, 1, 1],
        [0, 0, 0, 2, 1, 1, 1, 3, 2, 8, 0, 1]
    ])
    Flag = [i != 'X' for i in DF_MRI['T50']]
    Mix_X_100Block = []
    for i in range(len(DF_MRI[Flag])):
            random.seed(10)

            Random_position = []
            Random_position_all = []

            Rest = range(0, 100)
            for col in ['VPA', 'QJ2', 'QJ3', 'FDH3', 'LX', 'ZDH1']:
                
                X_random_position = random.sample(Rest, int(DF_MRI[Flag][col].iloc[i] * 100))
                Random_position.append(X_random_position)
                for p in X_random_position:
                    Random_position_all.append(p)
                Rest = []
                for x in range(0, 100):
                    if x not in Random_position_all:
                        Rest.append(x)
            
            Sequency_X = np.zeros((100, 12))
            for j in range(100):
                if j in Random_position[0]:
                    Sequency_X[j] = X.iloc[0].values
                elif j in Random_position[1]:
                    Sequency_X[j] = X.iloc[1].values
                elif j in Random_position[2]:
                    Sequency_X[j] = X.iloc[2].values
                elif j in Random_position[3]:
                    Sequency_X[j] = X.iloc[3].values
                elif j in Random_position[4]:
                    Sequency_X[j] = X.iloc[4].values
                elif j in Random_position[5]:
                    Sequency_X[j] = X.iloc[5].values
                    
            Mix_X_100Block.append(Sequency_X)   

    Mix_X_100Block = np.array(Mix_X_100Block)
    Mix_Y = np.array(DF_MRI[Flag]['T50'])
    if need_norm:
        np.savez('./data/100Block_mean_std.npz', x_mean=Mix_X_100Block.mean(), x_std=Mix_X_100Block.std(), y_mean=Mix_Y.mean(), y_std=Mix_Y.std())
        Mix_X_100Block = (Mix_X_100Block - Mix_X_100Block.mean()) / Mix_X_100Block.std()
        Mix_Y = (Mix_Y - Mix_Y.mean()) / Mix_Y.std()

    x_train, x_test, y_train, y_test = train_test_split(Mix_X_100Block, Mix_Y, train_size=train_ratio, random_state=42)
    return x_train, x_test, y_train, y_test

class CopolymerData(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]