import label_util as util
import glob
import threading
import numpy as np

# 从模板图像中匹配出原始图像

last_end = 0

# 当前主机共24线程，取23个作处理，留1个处理余数部分
threadingNum = 23

class mThread(threading.Thread):
    def __init__(self, threadId,filelist,filetype):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.filelist = filelist
        self.filetype = filetype
    def run(self):
        print('开始线程'+str(self.threadId))
        handleFilelist(self.filelist,self.threadId,self.filetype)
        print('退出线程:'+str(self.threadId))

def multiThread(filelist,filetype):
    length = len(filelist)
    last = length % threadingNum
    lastArr = filelist[-last:]
    pre = length - last

    arr = np.array(filelist[:pre])
    arr = np.split(arr,threadingNum)

    threads = [mThread(threadingNum+1,lastArr,filetype)] if last>0 else []

    for threadId in np.arange(threadingNum):
        item = mThread(threadId+1,arr[threadId],filetype)
        threads.append(item)
    
    for item in threads:
        item.start()
    
    for item in threads:
        item.join()
    print('exit main thread')

def handleDir(filetype = 'fake_image'):        

    filelist = glob.glob(util.FAKE_IMG_DIR+'\\'+filetype+'\\*.bmp')

    filelist = filelist[last_end:]

    multiThread(filelist,filetype)
 
    # handleFilelist(filelist,filetype)


def handleFilelist(filelist,threadId,filetype='fake_image'):    
    fileLen = len(filelist)

    print('线程{}:{}个文件需要处理'.format(threadId,fileLen))

    for i, filename in enumerate(filelist):
        strfile = filename.split(filetype+'\\')[1]

        bmpfileName = strfile.replace('.bmp', '')
        
        util.matchModel(bmpfileName,filetype)

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
        
# handleDir('fake_image')

handleDir('normal_image')

# handleTxtList('g:\\data_202009-202104\\append.txt')


