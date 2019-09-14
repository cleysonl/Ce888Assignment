
import numpy as np
import pandas as pd

lines_s=[]
#  file_data = open(r'C:\Users\CLL\Documents\Github\Ce888 Assignment\Parkinsons\parkinsons_updrs.data'): # Open file on read mode
#  file_data.readline()


# for i in range(len(data)):
#     data_final.loc[len(data_final)]=data.loc[i]

# data.to_csv('data.csv')
i=0
listr=[]
listc=[]
listt=[]
# Open wikiner file
with open(r'C:\Users\CLL\Documents\Github\Ce888 Assignment\Parkinsons\parkinsons_updrs.data', 'r', encoding="utf8") as file:
  # Go through each line of the file
    for line in file:
      # Create a list separated by spaces of the word and its tags (POS, NER)
      items = line.split("\n")
      for numbers in items:
          listr.append(numbers)
          
# print(listr[0])
for number in listr:
    for i in number.split(' '):
        listc.append(i)
    listt.append(listc)

print(listt[0])
# data=pd.DataFrame(listc)
# data.to_csv('data4.csv')    
