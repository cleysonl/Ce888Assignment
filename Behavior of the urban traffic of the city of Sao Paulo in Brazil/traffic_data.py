import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

# Extract the data and change the 4,6 to 4.6 in Slowness in traffic (%)
# data = pd.read_csv('Ce888Assignment\Behavior of the urban traffic of the city of Sao Paulo in Brazil\SP_traffic.csv',sep=';')

# for i in range(len(data)):
#     data['Slowness in traffic (%)'].iloc[i]=data['Slowness in traffic (%)'].iloc[i].replace(',','.')
    
# data.to_csv('traffic_data1.csv')

#Read the data
data=pd.read_csv('traffic_data1.csv')

#Preprocessing - hour (coded) from 1 to 27
lb = LabelBinarizer()
# Label Binarizer for Hour (Coded) (Action, Product and Inspiration)
lb_hour=lb.fit_transform(data['Hour (Coded)'])

lb_hour = pd.DataFrame(lb_hour)

data =pd.concat([data,lb_hour],axis=1)
print(data.keys)

