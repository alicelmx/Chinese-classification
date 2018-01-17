# Chinese-classification
利用tensorflow框架实现CNN中文文本分类

中间经过了漫长的期末考试周，第二步拖了半个多月，终于把第二步做好了，使用了两种方法，现在我先主要介绍基于深度学习的方法
### ***数据集选择***
一开始数据集大概每类300条，准确率只有86%左右，文本分类要求数据量足够，才能训练处合适的模型，我选择数据集的过程中经历了很多波折，最后使用的清华的THUCNews，我觉得是我能找到的最优的数据集了，关于数据集我专门写了一个博文，[请点这里](http://blog.csdn.net/alicelmx/article/details/79083903)。
最后使用的数据格式如下，因为原始数据量太大了，只抽取了一部分。

![这里写图片描述](http://img.blog.csdn.net/20180117154252097?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWxpY2VsbXg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### ***抽取、整理数据*** 
存放在text文件夹中，其中涉及两个模块：

 - copyData.py: 用于从每个分类拷贝1400个文件。
 - cnews_group.py：用于将多个文件整合到一个文件中。
#### ***从每个分类拷贝1400个文件***
##### 数据使用：
训练集: 1100*7
验证集: 100*7
测试集: 200*7
##### 代码如下

```
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
```
#### ***将多个文件整合到一个文件中***
##### 目标
创建sougou.train.txt（训练集1100*7）、sougou.test.txt（验证集100*7）、sougou.val.txt（测试集200*7），其中每一个文件包含每个类下的部分文件，存放于data文件夹中。
 ![这里写图片描述](http://img.blog.csdn.net/20180117160257014?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWxpY2VsbXg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
##### 代码实现

```
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
```
### ***数据预处理***
#### ***data/cnews_loader.py为数据的预处理文件。***

 1. read_file(): 读取文件数据; 
 2. build_vocab():构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理; 
 3. read_vocab():读取上一步存储的词汇表，转换为{词：id}表示; read_category(): 将分类目录固定，转换为{类别: id}表示;
 4. to_words(): 将一条由id表示的数据重新转换为文字; 

#### ***完整代码***

```
#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import os

def open_file(filename, mode='r'):
    """
    Commonly used file reader, change this to switch between python2 and python3.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content))
                labels.append(label)
            except:
                pass
    return contents, labels

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def read_category():
    """读取分类目录，固定"""
    categories =  [ '财经','房产','股票','家居','科技','时政','娱乐' ]
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示

    return x_pad, y_pad

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
```
### ***配置CNN卷积神经网络模型***
见cnnModel.py文件
```
#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 7        # 类别数
    num_filters = 256        # 卷积核数目
    kernel_size = 5         # 卷积核尺寸
    vocab_size = 5000       # 词汇表达小

    hidden_dim = 128        # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 10         # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

```
### ***训练和验证***
#### ***代码实现***
见runCNN.py
**若之前进行过训练，请把tensorboard/textcnn删除，避免TensorBoard多次训练结果重叠。**
```
#!/usr/bin/python
# -*- coding: utf-8 -*-

from cnnModel import *
from data.sougouLoader1 import *
from sklearn import metrics
import sys

import time
from datetime import timedelta


base_dir = '/Users/alicelmx/Documents/实习/文本分类/基于深度学习/data/'
train_dir = os.path.join(base_dir, 'sougou.train.txt')
test_dir = os.path.join(base_dir, 'sougou.test.txt')
val_dir = os.path.join(base_dir, 'sougou.val.txt')
vocab_dir = os.path.join(base_dir, 'sougou.vocab.txt')

save_dir = '/Users/alicelmx/Documents/实习/文本分类/基于深度学习/测试结果/'
save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0              # 总批次
    best_acc_val = 0.0           # 最佳验证集准确率
    last_improved = 0            # 记录上一次提升批次
    require_improvement = 1000   # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)   # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                    + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break

def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
    for i in range(num_batch):   # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)

    if sys.argv[1] == 'train':
        train()
    else:
        test()
```
#### ***训练结果***
python runCNN.py train

![这里写图片描述](http://img.blog.csdn.net/20180117163908834?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWxpY2VsbXg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
#### ***验证结果***
python runCNN.py test
结果还行**96.43%**，至少达到我老板的需求了，可以交差了

![这里写代码片](http://img.blog.csdn.net/20180117163943790?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWxpY2VsbXg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
