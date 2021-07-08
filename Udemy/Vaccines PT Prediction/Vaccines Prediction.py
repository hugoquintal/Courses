#importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir

directory = "dataset/"


def find_csv_filenames( directory, suffix=".csv" ):
    script_dir = os.path.dirname(__file__)
    filenames = listdir(os.path.join(script_dir,directory))
    for filename in filenames:
        if filename.endswith( suffix ):
            return os.path.join(script_dir,directory,filename)

data=pd.read_csv(find_csv_filenames(directory, ".csv"))
data.describe(include=None,exclude=None)

pessoas_vacinadas= data['pessoas_vacinadas_completamente'] #coluna com os registos das vacinas
pessoas_vacinadas.head()

pessoas_vacinadas.info()
