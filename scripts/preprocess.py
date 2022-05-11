from PIL import Image
import csv
import numpy as np

radar_name = "../data/AZ9010_256/Z_RADR_I_Z9010_%s_P_DOR_SA_R_10_230_15.010_clean.png"

rain_list = []
with open('../data/train.csv') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        rain_list.append(row)


slist = rain_list[::3]
for pic in slist:
    img1 = Image.open(radar_name % pic[0])
    img2 = Image.open(radar_name % pic[1])
    img3 = Image.open(radar_name % pic[2])
   
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)
    img3 = np.array(img3).astype(np.uint8)

    img = np.concatenate((img1, img2, img3), axis=0)
    img = Image.fromarray(img,mode='L')

    name = '../data/3dset/%s.png' % pic[0]

    img.save(name)
    print(name)

