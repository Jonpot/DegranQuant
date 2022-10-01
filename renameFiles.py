import os, glob
from PIL import Image

inDir = r'D:\degranulationData\SC699IgG\C05IgG\output\pngified\100999\*'
outDir = r'D:\degranulationData\training_images'
batchNum = 1

os.chdir(outDir)

i = 1000 * batchNum
for file in glob.glob(inDir):
    im = Image.open(file)
    outstr = str(i)+"_out.png"
    im = im.save(outstr)
    i +=1