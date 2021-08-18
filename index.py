import label_util as util
import glob
import threading
import numpy as np
import os 

# 从模板图像中匹配出原始图像

last_end = 0

# 当前主机共24线程，取23个作处理，留1个处理余数部分
threadingNum = 23

FAKE_IMG_DIR = 'g:\\data_202107\\'
OUTPUT_DIR = 'g:\\data_202107_test\\'


class mThread(threading.Thread):
    def __init__(self, threadId,filelist,input_dir,output_dir,filetype):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.filelist = filelist
        self.filetype = filetype
        self.output_dir = output_dir
        self.input_dir = input_dir
    def run(self):
        print('开始线程'+str(self.threadId))
        handleFilelist(self.filelist,self.threadId,self.input_dir,self.output_dir,self.filetype)
        print('退出线程:'+str(self.threadId))

def multiThread(filelist,input_dir,output_dir,filetype):
    length = len(filelist)
    last = length % threadingNum
    lastArr = filelist[-last:]
    pre = length - last

    arr = np.array(filelist[:pre])
    arr = np.split(arr,threadingNum)

    threads = [mThread(threadingNum+1,lastArr,input_dir,output_dir,filetype)] if last>0 else []

    for threadId in np.arange(threadingNum):
        item = mThread(threadId+1,arr[threadId],input_dir,output_dir,filetype)
        threads.append(item)
    
    for item in threads:
        item.start()
    
    for item in threads:
        item.join()
    print('所有线程处理完成')

def handleDir(input_dir,output_dir=OUTPUT_DIR,filetype = 'fake_image'):        

    filelist = glob.glob(input_dir+'\\'+filetype+'\\*.bmp')

    filelist = filelist[last_end:]

    multiThread(filelist,input_dir,output_dir,filetype)
 
    # handleFilelist(filelist,filetype)



def handleFilelist(filelist,threadId,input_dir,OUTPUT_DIR,filetype='fake_image'):    
    fileLen = len(filelist)

    print('线程{}:{}个文件需要处理'.format(threadId,fileLen))

    for i, filename in enumerate(filelist):
        strfile = filename.split(filetype+'\\')[1]

        bmpfileName = strfile.replace('.bmp', '')
        
        util.matchModel(bmpfileName,input_dir,OUTPUT_DIR,filetype)

        # try:
        #     util.matchModel(bmpfileName)
        # except:
        #     print(bmpfileName+'处理出错')
        

        if(i%1000==0):        
            print("{},线程{},{}/{} 处理完成({:2f}%),已处理至{}".format(filetype,threadId,i+1,fileLen,(i+1)*100/fileLen,i+last_end))
    
def handleTxtList(url):
    data = []
    with open(url,'r') as f:
        data = f.readlines()
        data = list(map(lambda x:x.replace('\n',''),data))

    # data  = list(filter(lambda x:x[:2] == '15',data))

    data = list(map(lambda x:util.FAKE_IMG_DIR+x,data))

    # handleFilelist(['fake_image\\131_1_2075H359_1225'])
    handleFilelist(data)
      
# handleTxtList('g:\\data_202009-202104\\append.txt')

  
# handleDir(FAKE_IMG_DIR,OUTPUT_DIR,'fake_image')
# handleDir(FAKE_IMG_DIR,OUTPUT_DIR,'normal_image')


def handleDirByCart():       

    dir_list = glob.glob(FAKE_IMG_DIR+'\\*')
    for dir_name in dir_list:
        input_name = dir_name
        cart = dir_name.replace(FAKE_IMG_DIR,'')
        print(dir_name)
        output_dir = OUTPUT_DIR+cart+'\\'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(output_dir+'fake_image')
            os.makedirs(output_dir+'normal_image')
                
        handleDir(input_name+'\\',output_dir,'fake_image')
        handleDir(input_name+'\\',output_dir,'normal_image')
         
handleDirByCart()