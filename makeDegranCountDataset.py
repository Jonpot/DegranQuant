from PIL import Image
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

#18c
#1c

os.chdir(r'D:\degranulationData\cropped_images\1c\processedPNG')
degranList = []
file_index=1000
for file in glob.glob(r'D:\degranulationData\cropped_images\1c\processedPNG\*'):
    im = Image.open(file)
    #im = im.convert('RGB')
    im.show()
    events = input("How many degranulation events did you see? [EXIT to exit]\n")
    if events == "EXIT":
        break
    degranList.append(int(events)//3)
    #im.save(str(file_index)+".png")
    file_index += 1

print(degranList)
m = open("degranulationCountData.txt","w")
m.write(str(degranList))
m.close