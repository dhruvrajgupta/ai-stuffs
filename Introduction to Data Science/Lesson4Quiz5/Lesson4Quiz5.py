# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 20:53:35 2021

@author: guptad
"""
import csv

f_in = open('data.txt','r')
f_out = open('updated_data.csv','w',newline='')
reader_in = csv.reader(f_in, delimiter = ',')
writer_out = csv.writer(f_out, delimiter =',')

for line in reader_in:
    print(line)
    rep = ",".join(line[0:3])
    print(len(line))
    for i in range(3,len(line)):
        print(i)
        if (i-3)%5==0 and i>3:
            rep = rep.split(",")
            print(rep)
            #writer_out.writerow(rep)
            rep = ",".join(line[0:3])+","+line[i].strip()
        else:
            rep += ","+line[i].strip()
        
        if i==(len(line)-1):
            print(rep.split(","))
    break

        
f_out.close()
f_in.close()

