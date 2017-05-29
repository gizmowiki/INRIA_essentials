'''
 __init__(imgname, bboxes, num_neg, max_width, max_overlap=0, maxattempts=1000, ar_min=0.3, ar_max=0.6, min_width=30):
'''
from plot_detection import PlotDetection

plt = PlotDetection(base_path='/user/rpandey/home/inria/dataset/chalearn/input',
                    detection_path='/user/rpandey/home/inria/dataset/chalearn/detection_csv',
                    extension='.jpg',
                    show_image=True)
