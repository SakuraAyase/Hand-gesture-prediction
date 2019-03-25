from keras.models import *
from keras.applications import *
from keras.layers import *
from keras.utils import *
import numpy as np
from skimage.io import imread, imsave
from skimage.util import pad
import csv

file_name = os.path.expanduser('~/data/hand-rheumatism/')


def read_csv(file):
    csv_data = []
    with open(file, "r") as f:
        for line in f:
            le = line.split("\n")[0].split(",")
            csv_data.append([int(le[1]), int(le[2]), int(le[3]), int(le[4])])
    return csv_data

def get_img(R):
    imgs = []
    ERO =[]
    JSN = []

    PAD_WIDTH = 50

    for i in range(1,R+1):
        sub_index = "%03d" % i
        file = file_name+sub_index
        csv_file = file+'\\xray.joint.csv'
        img_file = file +'\\xray.bmp'

        img = imread(img_file, as_grey=True)
        img = pad(img, PAD_WIDTH, mode='constant', constant_values=0)

        csv_data = read_csv(csv_file)
        csv_data = np.array(csv_data)
        for j in range(len(csv_data)):
            row = int(csv_data[j][0]) + PAD_WIDTH
            col = int(csv_data[j][1]) + PAD_WIDTH
            E = int(csv_data[j][2])
            j = int(csv_data[j][3])
            # img_temp = np.array(img[row-50:row+50,col-50:col+50])
            img_temp = np.array(img[col-50:col+50,row-50:row+50])
            try:
                # img_temp.shape = (100,100,1)
                img_temp.shape = (100,100)
            except ValueError as e:
                print(img_temp.shape)
                print(img.shape, "@", row, col)
                raise e
            imgs.append(np.stack((img_temp, img_temp, img_temp), axis=2))
            ERO.append(E)
            JSN.append(j)
    return np.array(imgs),np.array(ERO),np.array(JSN)


def model():
    col = 100
    row = 100
    channel = 3
    m = Sequential()

    model = VGG16(input_shape=(row, col, channel), weights='imagenet', include_top=False)
    print('Model loaded.')
    for layer in model.layers[:25]:
        layer.trainable = False
    model.summary()

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))

    top_model.add(Dense(64, activation='relu'))
    top_model.add(BatchNormalization())
    top_model.add(Dense(5))
    m.add(model)
    print(len(model.layers))
    # add the model on top of the convolutional base
    m.add(top_model)

    m.summary()
    return m


def train(x,y):
    print(y)
    y = to_categorical(y, num_classes=5)
    print(y)
    m=model()
    m.compile(optimizer='rmsprop',loss = 'mse',metrics=['accuracy'])
    m.fit(x,y,epochs=100)

    return m

def get_acc(y,y_):
    print(type(y), y.shape, y[0], " => ", np.argmax(y[0]))
    print(type(y_), y_.shape, y_[0])

    a = 0
    for i in range(len(y)):
        if(np.argmax(y[i])==y_[i]):
            a = a+1
    return a/len(y)

def div_byType(img,type):
    img0 = []
    type0 = []
    print(type)
    for i in range(5):
        temp_img = []
        temp_type = []
        for j in range(len(img)):
            if(type[j] == i):
                temp_img.append(img[j])
                temp_type.append(type[j])
        img0.append(np.array(temp_img))
        type0.append(np.array(temp_type))

    return img0,type0

def div_Ten(img_orig,type_orig):
    img, type0 = div_byType(img_orig,type_orig)


    for i in range(len(type0)):
        length = int(np.ceil(len(type0[i])*0.9))
        img[i][:length].shape

        if(i == 0):
            train_x = copy.deepcopy(img[i][:length])
            train_y = copy.deepcopy(type0[i][:length])
            test_x = copy.deepcopy(img[i][length:])
            test_y =copy.deepcopy(type0[i][length:])
        else:
            train_x = np.concatenate((train_x,copy.deepcopy(img[i][:length])))
            train_y = np.concatenate((train_y,copy.deepcopy(type0[i][:length])))
            test_x = np.concatenate((test_x,copy.deepcopy(img[i][length:])))
            test_y = np.concatenate((test_y,copy.deepcopy(type0[i][length:])))

    return train_x,np.array([train_y]).T,test_x,np.array([test_y]).T

img,ero,jsn = get_img(200)

img_train,ero_train,img_test,ero_test = div_Ten(img,ero)
print(img_train.shape)
print(ero_train.shape)

m_ero = train(img_train,ero_train)

img_train,jsn_train,img_test,jsn_test = div_Ten(img,jsn)

m_jsn = train(img_train,jsn_train)

print(get_acc(m_ero.predict(img_test),ero_test))
print(get_acc(m_jsn.predict(img_test),jsn_test))