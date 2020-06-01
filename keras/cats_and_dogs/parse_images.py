import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm

DATADIR = 'C:\\Datasets\\kagglecatsanddogs_3367a\\PetImages'
CATEGORIES = ['Dog', 'Cat']

IMG_SIZE = 200


training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                resize_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([resize_img_array, class_num])
            except Exception as e:
                pass
            # except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            # except Exception as e:
            #    print("general exception", e, os.path.join(path,img))


create_training_data()
print(len(training_data))
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# pickle_out = open('X.pickle', 'wb')
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open('y.pickle', 'wb')
# pickle.dump(y, pickle_out)
# pickle_out.close()

# to load
# pickle_in = open('X.pickle', 'rb')
# X = pickle.load(pickle_in)

np.save('X.npy', X)  # saving featureset
np.save('y.npy', X)  # saving classes
