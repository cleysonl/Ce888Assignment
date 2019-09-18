import numpy as np
import pandas as pd

# data=pd.read_csv('Ce888As
listr=[]
# Open Parkinson file
with open(r'Ce888Assignment\Concrete Slump\slump_test.data', 'r', encoding="utf8") as file:
  # Go through each line of the file
    for line in file:
      # Create a list separated by spaces
      items = line.split("\n")
      for numbers in items:
          listr.append(numbers)
           
# print(listr[0].split(','))
# print(listr[1].split(','))
# print(listr[2].split(','))
# print(listr[3].split(','))
# print(len(listr))

# Create the dataframe
data= pd.DataFrame(columns=listr[0].split(','))
print(data.columns)
listr.pop(0)
print(len(listr))

# To avoid the '' in the list 
for i in listr:
  if i!='':
    data.loc[len(data)]=i.split(',')

#Print into a csv files
data.to_csv('data_slump.csv')       