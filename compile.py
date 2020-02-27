import os
import h5py

import numpy as np

import cv2

from train import init
from test import get_img
import data.MPII.ref as ds
import utils.img

from angle import angle


func, config = init()
num_train = 100

input_res = config['train']['input_res']
output_res = config['train']['output_res']

train_f = h5py.File(os.path.join(ds.annot_dir, 'train.h5') ,'r') 

pairs = [
  (0, 1), (1, 2),
  (2, 6), (2, 12),
  (3, 4), (3, 13),
  (3, 6), (4, 5),
  (7, 8), (8, 9),
  (7, 12), (7, 13),
  (10, 11), (11, 12),
  (13, 14), (14, 15)
]

err = 0


for i in range(0, len(train_f['imgname'])):
  path_t = '%s/%s' % (ds.img_dir, train_f['imgname'][i].decode('UTF-8'))        
  
  ## img
  orig_img = cv2.imread(path_t)[:,:,::-1]
  c = train_f['center'][i]
  s = train_f['scale'][i]
  im = utils.img.crop(orig_img, c, s, (input_res, input_res))
  
  ## kp
  kp = train_f['part'][i]
  vis = train_f['visible'][i]
  kp2 = np.insert(kp, 2, vis, axis=1)
  kps = np.zeros((1, 16, 3))
  kps[0] = kp2
  
  ## normalize (to make errors more fair on high pixel imgs)
  n = train_f['normalize'][i]



  ## check if there is cross between body points
  # points must exists
  v1 = kp2[2, :2] - kp2[12, :2]
  v2 = kp2[3, :2] - kp2[12, :2]
  v3 = kp2[13, :2] - kp2[12, :2]

  if np.linalg.norm(v1) == 0 or \
    np.linalg.norm(v2) == 0 or \
    np.linalg.norm(v3) == 0:
    continue

  ag1, ag2 = angle(v1, v2), angle(v2, v3)

  if ag1*ag2 > 0:
    continue
  else:
    err += 1
    print(train_f['imgname'][i].decode('UTF-8'))

  ## visualization
  # vis_img = orig_img.copy()
  # vis_kp = kp2.astype(np.int)

  # for i in range(kp2.shape[0]):
  #   if vis_kp[i, 0] != 0 and vis_kp[i, 1] != 0:
  #     if i in [2, 3, 12, 13]:
  #       cv2.putText(
  #         vis_img, str(i), (vis_kp[i, 0], vis_kp[i, 1]), 0, 0.75,
  #         (255, 0, 0), 3
  #       )

  # for pair in pairs:
  #   if vis_kp[pair[0], 0] != 0 and vis_kp[pair[0], 1] != 0 and \
  #     vis_kp[pair[1], 0] != 0 and vis_kp[pair[1], 1] != 0:
  #     cv2.line(
  #       vis_img, (vis_kp[pair[0], 0], vis_kp[pair[0], 1]),
  #       (vis_kp[pair[1], 0], vis_kp[pair[1], 1]), (255, 0, 0), 3
  #     )

  # cv2.imshow('_', vis_img[:, :, ::-1])
  # if cv2.waitKey(0) == 27:
  #   break

print('error images: ', err, ' of ', len(train_f['imgname']), ' images!')













