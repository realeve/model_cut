
import numpy as np
import glob
import random
import os
import json
from shutil import copy

choose_dir = 'g:/data_train_2018/'
choose_to = 'g:/data_train_tiny/'
file_num = 1250

"""
从指定目录中选出一定数量的文件
"""
def listDir(dirName,fileNum,dstDir):
    cache_file_name = choose_dir+'/annotation.txt'
    # 从txt中读取文件,缓存加速
    HAS_CACHE_FILE = os.path.exists(cache_file_name)
    label = 0 if '/fake_image' in dstDir else 1

    fileList = []

    if HAS_CACHE_FILE:
        print("Find cache file:"+cache_file_name)
        cacheFile = open(cache_file_name, "r", encoding='UTF-8')
        out = cacheFile.read()
        instances = json.loads(out)
        print("Read file from cache:{} files".format(len(instances)))
        for item in instances:   
            if item[1] == label:
                fileList.append(item[0].replace('\\','/'))
    else:
        fileList = glob.glob(dirName+'*.bmp')
        
    random.shuffle(fileList)
    
    imgList = fileList[:fileNum]
    xmlList = list(map(lambda x:x.replace('.bmp','.xml'),imgList))

    for i,pathname in enumerate(imgList):
        copy(pathname,dstDir)
        if(i%1000==0):
            print("完成{:.2f}%".format(i*100/fileNum))
    
    for i,pathname in enumerate(xmlList):
        copy(pathname,dstDir)
        if(i%1000==0):
            print("完成{:.2f}%".format(i*100/fileNum))


listDir(choose_dir+'fake_image/',file_num,choose_to+'fake_image/')
listDir(choose_dir+'normal_image/',file_num,choose_to+'normal_image/')

