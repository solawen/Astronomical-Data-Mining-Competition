import pandas as pd
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D,MaxPooling1D
def creat_conv_model():
    model = Sequential()
    model.add(Conv1D(48,11,strides=4,padding='valid',input_shape =(2600,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3,strides=2))

    model.add(Conv1D(128,5,strides=1,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3,strides=2))

    model.add(Conv1D(192,3,strides=1,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3,strides=2))

    model.add(Flatten())
    model.add(Dense(200,kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(200,kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(3,kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

def read_data_byID(id, filepath):
    with open(filepath +str(id)+'.txt','r') as f:
        for line in f:
            df_star = list(map(float, line.split(',')))
        return df_star
def data_pre_process(a):
    #out = np.log10(a - a.min()+1)
    out = ((a - a.min())*100)/(a.max()-a.min())
    #out = medfilt(out)
    return out
def read_data(df, filename):
    X_train = np.zeros((df.shape[0],2600),dtype = float)
    for i,id in enumerate(df['id']):
        a = np.array(read_data_byID(id,filename))
        X_train[i,:] =data_pre_process(a)
        if i%1000 == 0:
            print("reading %d files"%i)
            print(X_train[i,:])
    print('Finished reading files')
    return X_train

if __name__ == '__main__':
    filename = '../data/first_train_data_20180131/'
    df = pd.read_csv('../data/train.csv')
    qso = df[df['type']=='qso']
    galaxy = df[df['type']=='galaxy']
    star = df[df['type']=='star']
    frames = [star,galaxy,qso]
    cw = {0:1,1:int(star.shape[0]/galaxy.shape[0]),2:int(star.shape[0]/qso.shape[0])}
    train = pd.concat(frames)
    train = train.sample(frac=1).reset_index(drop=True)
    X_train = read_data(train, filename)
    label = np.array(train['type'])
    y_train = np.zeros_like(label, dtype = int)
    y_train = (label == 'galaxy').astype(int) * 1 + (label == 'qso').astype(int) * 2
    #star =0 galaxy = 1 qso =2
    mean_X = X_train.mean(axis = 0,keepdims = True)
    std_X = X_train.std(axis = 0, keepdims = True)
    X_train = (X_train - mean_X)/(std_X)
    Y_train = (np.arange(3)==y_train[:,None]).astype(int)
    #---------------------training-------------------------------
    lr_rate = 1e-3
    reg = 0.02
    best_accu = 0
    best_model = None
    results = {}
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    model = creat_conv_model()
    optimizer = SGD(lr=lr_rate,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    model.fit(X_train,Y_train,batch_size=256,epochs=10,shuffle=True,verbose=1,validation_split=0.2,class_weight = cw)
    #------------------------------------------------------------

    #------------------Saving---------------------
    Version = '0.0.1'
    model.save('3_classfier_unknown_model_ver'+Version+'.h5')
    np.save('mean_X_3_classfier_unknown_model_ver'+Version+'.npy',mean_X)
    np.save('std_X_3_classfier_unknown_model_ver'+Version+'.npy',std_X)
    #---------------------------------------------
