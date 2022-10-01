from pathlib import Path
import glob
import numpy as np
from PIL import Image
import os

images = list()
os.chdir(r'D:\degranulationData\SC699IgG\C05IgG\output2')
i=1000

for file in glob.glob(r'D:\degranulationData\SC699IgG\C05IgG\*'):
    im = Image.open(file)
    im = im.crop((555,413,655,513))
    imarray = np.array(im)
    max = np.amax(imarray)
    print(max)
    imarray = np.divide(imarray,5000)
    #imarray = np.multiply(imarray,65535)
    #imarray = np.power(imarray, 0.25)
    imarray = np.clip(imarray,0,65535)
    im = Image.fromarray(imarray)
    outstr = str(i)+"_out.tif"
    im = im.save(outstr)
    i +=1
    