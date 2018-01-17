import os
import glob
import shutil
import random

basePath = "/Users/alicelmx/Documents/实习/文本分类/基于深度学习/SogouData/ClassFile/"
newPath = "/Users/alicelmx/Documents/实习/文本分类/基于深度学习/text/"

listPath = list(map(lambda  x:basePath+str(x)+"/",list(filter(lambda  x:not str(x).startswith("."),os.listdir(basePath)))))
"""
训练集: 1100*7
验证集: 100*7
测试集: 200*7
"""
def copy(listPath,MAXCOUNT=1400):
    for  path in listPath:
        newdir = newPath+ str(path).split("/")[-2]
        print("====================")
        print(newdir)
        print("====================")

        if not os.path.exists(newdir):
            os.mkdir(newdir)
        files=glob.glob(path+"*.txt")
        
        if len(files) < MAXCOUNT:
            resultlist = []
            for i in range(MAXCOUNT):
                resultlist.append(random.choice(files))
        else:
            resultlist = random.sample(files,MAXCOUNT)
        for file in resultlist:
            shutil.copy(file,newdir)

if  __name__=='__main__':
    copy(listPath)
    print("抽取成功!")