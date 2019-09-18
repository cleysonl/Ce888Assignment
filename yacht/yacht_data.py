import numpy as np
import pandas as pd


file_data = np.loadtxt(r'Ce888Assignment\yacht\yacht_hydrodynamics.data', usecols=(0,1,2,3,4,5,6))

data = pd.DataFrame()
data['x1']= file_data[:,0]
data['x2']= file_data[:,1]
data['x3']= file_data[:,2]
data['x4']= file_data[:,3]
data['x5']= file_data[:,4]
data['x6']= file_data[:,5]
data['y']= file_data[:,6]
print(data.columns)
data.to_csv('data_yacht.csv')