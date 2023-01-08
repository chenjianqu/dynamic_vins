import csv
import sys
import pandas as pd

if len(sys.argv)!=3:
    raise RuntimeError("must input src_file and dst_file parameters")

src_file = sys.argv[1]
dst_file = sys.argv[2]

'''
要读取的文件如下：
### 0000
#### 0000_VO_raw_PointOnly_Odometry
![2023-01-03 13-04-04屏幕截图](imgs/kitti_mot_exp_imgs/2023-01-03 13-04-04屏幕截图.png)

```shell
APE w.r.t. translation part (m)
(with SE(3) Umeyama alignment)

       max	4.243920
      mean	1.312137
    median	1.169781
       min	0.181570
      rmse	1.494236
       sse	341.609621
       std	0.714870

RPE w.r.t. translation part (m)
for delta = 1 (frames) using consecutive pairs
(with SE(3) Umeyama alignment)

       max	3.287083
      mean	0.694021
    median	0.649339
       min	0.262533
      rmse	0.763477
       sse	88.600480
       std	0.318170

RPE w.r.t. rotation part (unit-less)
for delta = 1 (frames) using consecutive pairs
(with SE(3) Umeyama alignment)

       max	0.251412
      mean	0.034645
    median	0.032418
       min	0.000844
      rmse	0.041001
       sse	0.255520
       std	0.021926
```

'''

if __name__ == '__main__':
    src=open(src_file)

    key=''

    ate_max_v=''
    ate_mean_v=''
    ate_median_v=''
    ate_rmse_v=''
    rpe_t_max_v=''
    rpe_t_mean_v=''
    rpe_t_median_v=''
    rpe_t_rmse_v=''
    rpe_r_max_v=''
    rpe_r_mean_v=''
    rpe_r_median_v=''
    rpe_r_rmse_v=''

    mode='ATE'

    for line in src.readlines():
        cur_line = line.strip()

        #写入csv文件
        if(cur_line=='```'):
            data_row=[key,ate_max_v,ate_mean_v,ate_median_v,ate_rmse_v,rpe_t_max_v,rpe_t_mean_v,rpe_t_median_v,rpe_t_rmse_v,
                      rpe_r_max_v,rpe_r_mean_v,rpe_r_median_v,rpe_r_rmse_v]
            with open(dst_file,'a+') as f:#追加写入
                csv_file = csv.writer(f)
                csv_file.writerow(data_row)

        if(cur_line=='APE w.r.t. translation part (m)'):
            mode='ATE'
        elif(cur_line=='RPE w.r.t. translation part (m)'):
            mode='RPE_t'
        elif(cur_line=='RPE w.r.t. rotation part (unit-less)'):
            mode='RPE_r'

        cur_list = cur_line.split('\t')
        #print(cur_list)
        if(cur_line.startswith('####')):
            split_list = cur_line.split(' ')
            print(cur_line)
            key=split_list[1]

        if(cur_line.startswith('max') or cur_line.startswith('mean') or cur_line.startswith('median') or cur_line.startswith('rmse')):
            if(mode=='ATE'):
                if(cur_line.startswith('max')):
                    ate_max_v=cur_list[1]
                elif(cur_line.startswith('mean')):
                    ate_mean_v=cur_list[1]
                elif (cur_line.startswith('median')):
                    ate_median_v = cur_list[1]
                elif (cur_line.startswith('rmse')):
                    ate_rmse_v = cur_list[1]
            elif(mode=='RPE_t'):
                if(cur_line.startswith('max')):
                    rpe_t_max_v=cur_list[1]
                elif(cur_line.startswith('mean')):
                    rpe_t_mean_v=cur_list[1]
                elif (cur_line.startswith('median')):
                    rpe_t_median_v = cur_list[1]
                elif (cur_line.startswith('rmse')):
                    rpe_t_rmse_v = cur_list[1]
            elif(mode=='RPE_r'):
                if(cur_line.startswith('max')):
                    rpe_r_max_v=cur_list[1]
                elif(cur_line.startswith('mean')):
                    rpe_r_mean_v=cur_list[1]
                elif (cur_line.startswith('median')):
                    rpe_r_median_v = cur_list[1]
                elif (cur_line.startswith('rmse')):
                    rpe_r_rmse_v = cur_list[1]

