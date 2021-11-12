import os
from shutil import copy2

OR_list=['KA01','KA03','KA05','KA06','KA07','KA08','KA09','KA04','KA15','KA16','KA22','KA30']
IR_list=['KI01','KI03','KI05','KI07','KI08','KI04','KI14','KI16','KI17','KI18','KI21']
HL_list=['K001','K002','K003','K004','K005','K006']
ALL_list=[OR_list, IR_list, HL_list]
Condition_list=['N15_M07_F10','N15_M01_F10','N15_M07_F04']


dir1='D:/NSL/python3/charmpjs/data/Paderborn Unibersity dataset/data'
target_path='D:/NSL/python3/charmpjs/data/Paderborn Unibersity dataset/processed_data_v2'
bearing_list=os.listdir(dir1)


# for i in bearing_list:
#     dir2=os.path.join(dir1,i)
#     data_list=os.listdir(dir2)
#     for item in data_list:
#         if item[0:3]=='N15':
#             print('Yes')
#         else:
#             print('No')
#
#     break

for bearing in bearing_list:
    for i, clas in enumerate(ALL_list):
        #print(bearing)
        if bearing in clas:
            dir2=os.path.join(dir1,bearing)
            # tar_clas=os.path.join(target_path,str(i))
            # if os.path.isdir(tar_clas):
            #     pass
            # else:
            #     os.mkdir(tar_clas)
            data_list=os.listdir(dir2)
            for item in data_list:
                if item[0:11] in Condition_list:
                    tar_condi_path=os.path.join(target_path,item[0:11])
                    if os.path.isdir(tar_condi_path):
                        pass
                    else:
                        os.mkdir(tar_condi_path)
                    tar_condi_clas_path=os.path.join(tar_condi_path,str(i))
                    if os.path.isdir(tar_condi_clas_path):
                        pass
                    else:
                        os.mkdir(tar_condi_clas_path)
                    item_path=os.path.join(dir2,item)
                    copy2(item_path,tar_condi_clas_path)
