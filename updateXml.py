import glob
import threading
import numpy as np

dirname='G:/data_train_202001-202106/fake_image/'
# dirname='G:/data_train_202001-202106/normal_image/'
# dirname = 'g:/t/'
# 1273369 实废  1281760 误废
# 当前主机共24线程，取23个作处理，留1个处理余数部分
threadingNum = 23

# 文件手工拷贝后需要更新 xml文件内容  

def readxml(dirname,url):   
    # print('read',dirname+url) 
    with open (dirname+url,'r+') as f:
        data=[]
        filename=''
        for i,line in enumerate(f.readlines()):
            # line = line.strip()
            if('<?xml' in line or 'annotation>ject>' in line):
                continue
            elif('<filename>' in line):
                filename = line.replace('<filename>','').replace('</filename>','').strip()
            elif('<path>' in line):
                pathIdx=line.index('</path>')
                line = line.replace(line[:pathIdx],'\t<path>'+dirname+filename)
            data.append(line.replace('\/','/')) 
        f.truncate(0)
        f.seek(0)
        f.writelines(data)
        f.close

def fixxml(dirname,url):   
    # print('read',dirname+url) 
    with open (dirname+url,'rt+') as f:
        data='<annotation>\n'
        for i,line in enumerate(f.readlines()):
            if i>0:
                data+=line
        f.truncate(0)
        f.seek(0)
        f.write(data)
        f.close

# readxml(dirname,'101_0_1980H159_1003.xml')

def getFiles():
    bmpPaths = glob.glob(dirname+'/*.bmp')
    return list(map(lambda path:path.replace('\\','/').replace(dirname,''),list(bmpPaths)))

def handleFilelist(filelist,threadId):    
    fileLen = len(filelist)
    
    errlist = []
    print('线程{}:{}个文件需要处理'.format(threadId,fileLen))

    for i, filename in enumerate(filelist):
        try:
            fixxml(dirname,filename.replace('.bmp','.xml'))
        except:
            # print(filename)
            errlist.append(filename+'\\n')
        if(i%1000==0):        
            print("线程{},{}/{} 处理完成({:2f}%),已处理至{}".format(threadId,i+1,fileLen,(i+1)*100/fileLen,i))
    
    # with open('err{}.txt'.format(threadId),'at+',encoding='utf-8') as file:
    #     file.writelines(errlist)
       
    

class mThread(threading.Thread):
    def __init__(self, threadId,filelist):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.filelist = filelist
    def run(self):
        print('开始线程'+str(self.threadId))
        handleFilelist(self.filelist,self.threadId)
        print('退出线程:'+str(self.threadId))

def multiThread(filelist):
    length = len(filelist)
    last = length % threadingNum
    lastArr = filelist[-last:]
    pre = length - last

    arr = np.array(filelist[:pre])
    arr = np.split(arr,threadingNum)

    threads = [mThread(threadingNum+1,lastArr)] if last>0 else []

    for threadId in np.arange(threadingNum):
        item = mThread(threadId+1,arr[threadId])
        threads.append(item)
    
    for item in threads:
        item.start()
    
    for item in threads:
        item.join()
    print('所有线程处理完成') 

def handleDir():   
    filelist = getFiles()
    print('共需处理图片 {} ',len(filelist))
    multiThread(filelist)

handleDir()