import scipy.io as scio

dataFile='Paderborn Unibersity dataset/processed_data_v2/N15_M01_F10/0/N15_M01_F10_KA08_20'
data=scio.loadmat(dataFile)
data_a=data['N15_M01_F10_KA08_20'][0,0]
for i,item in enumerate(data_a):
    print(i)
    print(item)

print('----------------------------------------------------------')
data_b=data_a[2][0]
for i,item in enumerate(data_b):
    print(i)
    print(item)

print('-----------------------------------------------------------')
data_c=data_b[6]
for i,item in enumerate(data_c):
    print(i)
    print(item)
# print(data_a[1][0,1][2].shape)
print(data_c[2].shape)

###
# 一级数据 data_a 包含4个部分 1）三个数字，感觉像标记 2）x坐标 包含Mech_4kHz(采样率), 和HostService
# 3) y坐标 对应x坐标的采样率， 分别有三种数据，Hostservice中有两个电流current的和一个vibration震动的 4）额外信息
# 二级数据 data_b， 在data_b=data_a[1]时为x坐标， 其中data_b[1]对应HostService
# data_b=data_a[2]时为y坐标， 其中data_b[6]对应vibration
####
