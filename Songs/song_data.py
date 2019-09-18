import numpy as np
import pandas as pd

listr=[]

# Open Parkinson file
with open(r'C:\Users\CLL\Documents\Github\Ce888 Assignment\Ce888Assignment\Songs\YearPredictionMSD.txt', 'r', encoding="utf8") as file:
  # Go through each line of the file
    for line in file:
      # Create a list separated by spaces
      items = line.split("\n")
      for numbers in items:
          listr.append(numbers)
           
# print(len(listr[0].split(',')))
# print(listr[1].split(','))
# print(listr[2].split(','))
# print(listr[3].split(','))
# print(len(listr))
col = list(range(91))
data=pd.DataFrame(columns = col)
# Create the dataframe
# To avoid the '' in the list 
n=0

for i in listr:
  if i!='':
    data.loc[len(data)]=i.split(',')
    n+=1
  print(n)

print(data.keys)
data.to_csv('data_songs.csv')

###### SUBSET DATA########
