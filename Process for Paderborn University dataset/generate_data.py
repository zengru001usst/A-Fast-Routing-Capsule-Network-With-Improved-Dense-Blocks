import scipy.io as scio
import os
from shutil import copy2
from STFT_v2 import calc_stft
import matplotlib.pyplot as plt

########################
# 这个文件用来生成轴承时频图数据集
# 分为一下几个步骤
# 从mat 文件中读取相应位置的震动数据
# 切分成小的片段，然后转化为时频图
# 保存热力图到目标文件夹
# ######################

def read_data(path,name):
    ## return vibration data
    data = scio.loadmat(path)
    data = data[name][0][0][2][0][6][2]
    return data

def G_data(source_dir,target_dir):
    Class_list=os.listdir(source_dir)
    for class_item in Class_list:
        if class_item=='1':
            S_dir1=os.path.join(source_dir,class_item)
            T_dir1=os.path.join(target_dir,class_item)
            if os.path.isdir(T_dir1):
                pass
            else:
                os.mkdir(T_dir1)
            Data_list=os.listdir(S_dir1)
            for data_item in Data_list:
                S_dir2=os.path.join(S_dir1,data_item)
                print(data_item[0:-4])
                if data_item[0:16] >= 'N15_M01_F10_KA08':
                    ##  至此，已经得到mat文件的路劲
                    try:
                        signal=read_data(S_dir2,data_item[0:-4])[0]
                    except:
                        continue
                    #signal_length=len(signal)
                    for i in range(0,2):
                        img_path=os.path.join(T_dir1,'{}_{}.png'.format(data_item[0:-4],i))
                        if os.path.isdir(img_path):
                            pass
                        else:
                            signal_segment=signal[i*6800:(i+1)*6800]
                            img=calc_stft(signal_segment)
                            plt.axis('off')
                            plt.imshow(img, aspect='equal')
                            plt.tight_layout()
                            plt.gca().xaxis.set_major_locator(plt.NullLocator())
                            plt.gca().yaxis.set_major_locator(plt.NullLocator())
                            print(os.path.join(T_dir1,'{}_{}.png'.format(data_item[0:-4],i)))
                            plt.savefig(img_path,bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    source_path='Paderborn Unibersity dataset/processed_data_v2/N15_M01_F10'
    target_path='Paderborn Unibersity dataset/image_v3/N15_M01_F10'
    G_data(source_path,target_path)

