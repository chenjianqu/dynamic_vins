import csv
import sys
import pandas as pd

if len(sys.argv)!=3:
    raise RuntimeError("must input src_file and dst_file parameters")

src_file = sys.argv[1]
dst_file = sys.argv[2]

'''
要读取的文件如下：
room_static_2_VO_raw_LinePoint_Odometry.txt
APE w.r.t. translation part (m)
(with SE(3) Umeyama alignment)

       max	0.706135
      mean	0.397440
    median	0.396046
       min	0.135221
      rmse	0.421358
       sse	80.426744
       std	0.139942
'''

if __name__ == '__main__':
    src=open(src_file)
    first_line = src.readline() #第一行
    head_list = first_line.split('.')
    key = head_list[0]

    max_v=''
    mean_v=''
    median_v=''
    rmse_v=''

    for line in src.readlines():
        cur_line = line.strip()
        cur_list = cur_line.split('\t')
        #print(cur_list)
        if(cur_line.startswith('max')):
            max_v=cur_list[1]
        elif(cur_line.startswith('mean')):
            mean_v=cur_list[1]
        elif (cur_line.startswith('median')):
            median_v = cur_list[1]
        elif (cur_line.startswith('rmse')):
            rmse_v = cur_list[1]

    data_row=[key,max_v,mean_v,median_v,rmse_v]
    #追加写入
    with open(dst_file,'a+') as f:
        csv_file = csv.writer(f)
        csv_file.writerow(data_row)

