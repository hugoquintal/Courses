#importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

directory ="../dataset/"
for root,dirs,files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            data = pd.read_csv(file)

data.info()
