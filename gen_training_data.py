"""
 Student : Ibrahim Hashmi
      ID : 6352926
Language : Python
     IDE : VSCode
"""
import csv

def expectedOutput(num):
     total=0
     for digit in str(num):
          if digit == "1":
               total+=1
     if total%2==0:
          return 1
     else:
          return 0

def numOf1s(input,output):
     total = str(input).count("1") + output
     return total


num = 0

for i in range(16):
    b_num = format(num, 'b').zfill(4)
    with open('trainingdata.csv','a',newline='') as file:
     writer = csv.writer(file)
     writer.writerow([b_num,b_num.count("1"),expectedOutput(b_num),numOf1s(b_num,expectedOutput(b_num))])
    num+=1

