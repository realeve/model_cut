
import numpy as np
import glob
import random

from shutil import copy

"""
从指定目录中选出一定数量的文件
"""
def listDir(dirName,fileNum,dstDir):

    fileList = glob.glob(dirName+'*.bmp')
    random.shuffle(fileList)
    
    imgList = fileList[:fileNum]
    xmlList = list(map(lambda x:x.replace('.bmp','.xml'),imgList))

    for i,pathname in enumerate(imgList):
        copy(pathname,dstDir)
        if(i%1000==0):
            print("完成{:.2f}%",i*100/fileNum)
    
    for i,pathname in enumerate(xmlList):
        copy(pathname,dstDir)
        if(i%1000==0):
            print("完成{:.2f}%",i*100/fileNum)

listDir('g:/data_train_202001-202106/fake_image/',200000,'g:/data_train/fake_image/')
listDir('g:/data_train_202001-202106/normal_image/',200000,'g:/data_train/normal_image/')

