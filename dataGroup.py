#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
将文本整合到 train、test、val 三个文件中
"""
import  os

basePath = "/Users/alicelmx/Documents/实习/文本分类/基于深度学习/text/"
trainPath = "/Users/alicelmx/Documents/实习/文本分类/基于深度学习/data/"

def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')

def  save_file(dirname):
    """
        将多个文件整合并存到3个文件中
        dirname: 原数据目录
        文件内容格式:  类别\t内容
    """
    f_train = open(trainPath+"sougou.train.txt",'w',encoding='utf-8')
    f_test = open(trainPath + "sougou.test.txt", 'w', encoding='utf-8')
    f_val = open(trainPath + "sougou.val.txt", 'w', encoding='utf-8')
    
    for category in os.listdir(dirname):
        catdir = os.path.join(dirname,category)
        if not os.path.isdir(catdir):
            continue
        files = os.listdir(catdir)
        print(len(files))
        
        count = 0
        for cur_file in files:
            filename = os.path.join(catdir,cur_file)
            content = _read_file(filename)

            if count < 1100:
                f_train.write(category+"\t"+content+"\n")
            elif count < 1300:
                f_test.write(category+"\t"+content+"\n")
            else:
                f_val.write(category + '\t' + content + '\n')
            count += 1
        
        print("===============")
        print("finish:",category)
        print("===============")
   
    f_train.close()
    f_test.close()
    f_val.close()

if  __name__=='__main__':
    save_file(basePath)
    print(len(open(trainPath+"sougou.train.txt", 'r', encoding='utf-8').readlines()))
    print(len(open(trainPath + "sougou.test.txt", 'r', encoding='utf-8').readlines()))
    print(len(open(trainPath + "sougou.val.txt", 'r', encoding='utf-8').readlines()))

