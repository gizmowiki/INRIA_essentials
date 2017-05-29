import skimage.io as skio
import skimage.feature as skfeat
import skimage.transform as sktrans
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
import random
# Constant Params

fit_generator = True

filelist = []
base_path = "/local/people_depth"
base_path_neg = "/local/people_depth/nyud"
rf_trees = 90
out_path = "/home/rpandey/people_depth/hog/"
if not os.path.exists(out_path):
    os.makedirs(out_path)

### Functions for parameters
def load():
    global filelist
    # mean_img = cv2.imread(os.path.join(base_path, 'mean_img.jpg'))
    for lines in open(os.path.join(base_path, 'files_hog_positives.txt')):
        lines = lines.strip()
        filelist.append(lines)
    for lines in open(os.path.join(base_path_neg, 'files_hog_negatives.txt')):
        lines = lines.strip()
        filelist.append(lines)

    random.shuffle(filelist)
    random.shuffle(filelist)

load()

num_train_samples = len(filelist[:int(0.8 * len(filelist))]) - (len(filelist[:int(0.8 * len(filelist))]) % 64)
remaining = len(filelist) - num_train_samples
num_test_samples = remaining - (remaining%64)

img_shape = (128, 64)

def load_features(validate=False):
    sr_pt = 0
    st_pt = num_train_samples
    if validate:
        print ("\t\t\t\t\t\t\tNow loading validation data\n----------------------------------------------------------------------------")
        sr_pt = st_pt
        st_pt += num_test_samples
    else:
        print ("\t\t\t\t\t\t\tNow loading training data\n------------------------------------------------------------------------------")
    feat = []
    label = []
    for i in range(sr_pt, st_pt):
        img = skio.imread(filelist[i])
        img = sktrans.resize(img_shape)
        hog_feat = skfeat.hog(img)
        feat.append(hog_feat)
        if "positive" in filelist[i]:
            label.append(1)
        else:
            label.append(0)

    return feat, label


X_train, Y_train = load_features()

print ("\t\t\t\t\t\t\t\t\t\tNow training Random Forest with no of trees %d with HOG features of feature size %d and total no of training samples %d\n----------------------------------------------------------------------------------------------------------------"
         % (rf_trees, X_train[0].shape[0], len(X_train)))

clf = RandomForestClassifier(n_estimators=rf_trees)

clf.fit(X_train, Y_train)

X_test, Y_test = load_features(validate=True)
print("\n\t\t\t\t\t\t\t\t\t\t\tNow calculating cross val score\n--------------------------------------------------------------------------------------")
scores = cross_val_score(clf, X_test, Y_test, cv=10)
print (scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

joblib.dump(clf, os.path.join(out_path,"rf_90.pkl"))
print ("Saved the model")