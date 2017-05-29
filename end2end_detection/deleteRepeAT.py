import os
import random

base_path = '/data/stars/share/people_depth/people-depth/fulldata'
for i in range(150000, 1750000, 4):
	del_id = i + random.randint(0,3)
	pos_del_path = os.path.join(base_path, 'positives', '{0:08d}.jpg'.format(del_id))
	del_id_neg = int(i*1.9) + 1
	del_id_neg_2 = int(i*1.9) + 2	
	neg_del_path_1 = os.path.join(base_path, 'negatives', '{0:08d}.jpg'.format(del_id_neg))
	neg_del_path_2 = os.path.join(base_path, 'negatives', '{0:08d}.jpg'.format(del_id_neg_2))
	print("Now deleting ... ", pos_del_path)
	os.remove(pos_del_path)
	print("Now deleting ... ", neg_del_path_1)
	os.remove(neg_del_path_1)
	print("Now deleting ... ", neg_del_path_2)
	os.remove(neg_del_path_2)