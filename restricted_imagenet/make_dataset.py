import os
import shutil
train_root="/newsd4/zgh/ImageNet"
val_root="/newsd4/zgh/ImageNet_Val"
val_label='restricted_imagenet/orig_val.txt'
target_train="/newsd4/zgh/data/RestrictedImageNet/train"
target_val="/newsd4/zgh/data/RestrictedImageNet/val"

def make_trainset(source,target):
    if not os.path.exists(source):
        print('no source dir!')
    if not os.path.exists(target):
        print('no target dir, new one!')
        os.mkdir(target)
    index=0
    for i in range(9):
        target_path=os.path.join(target,str(i))
        if not os.path.exists(target_path):
            os.mkdir(target_path)
    for sdir in os.listdir(source):
        newclass=transfer_class(index)
        print(index,sdir,newclass)
        index+=1
        if newclass>=0:
            target_path = os.path.join(target, str(newclass))
            for file in os.listdir(os.path.join(source,sdir)):
                sf=os.path.join(source,sdir,file)
                tf=os.path.join(target_path,file)
                os.system('cp %s %s' % (sf, tf))

def make_valset(source,target,label):
    if not os.path.exists(source):
        print('no source dir!')
    if not os.path.exists(label):
        print('no label file!')
    if not os.path.exists(target):
        print('no target dir, new one!')
        os.mkdir(target)

    for i in range(9):
        target_path=os.path.join(target,str(i))
        if not os.path.exists(target_path):
            os.mkdir(target_path)
    with open(label, mode='r') as labelfile:
        while True:
            lines = labelfile.readline()  # 整行读取数据
            if not lines:
                break
                pass
            p_tmp, E_tmp = [i for i in lines.split()]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            index=float(E_tmp)
            newclass=transfer_class(index)
            if newclass>=0:
                sf=os.path.join(source,p_tmp)
                target_path = os.path.join(target, str(newclass))
                #tf=os.path.join(target_path,p_tmp)
                tf=target_path
                print(lines,sf,tf,newclass)
                os.system('cp %s %s' % (sf, tf))
                #shutil.copy(sf, tf)

def transfer_class(origin_class):
    if origin_class>=151 and origin_class<=268:
        return 0
    if origin_class>=281 and origin_class<=285:
        return 1
    if origin_class>=30 and origin_class<=32:
        return 2
    if origin_class>=33 and origin_class<=37:
        return 3
    if origin_class >= 80 and origin_class <= 100:
        return 4
    if origin_class >= 365 and origin_class <= 382:
        return 5
    if origin_class >= 389 and origin_class <= 397:
        return  6
    if origin_class >= 118 and origin_class <= 121:
        return 7
    if origin_class >= 300 and origin_class <= 319:
        return 8
    return  -1

make_trainset(train_root,target_train)
#make_valset(
#    val_root,target_val,val_label
#)