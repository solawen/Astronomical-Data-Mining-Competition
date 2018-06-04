import pandas as pd
import numpy as np
from keras.models import load_model
def read_data_byID(id, filepath):
    with open(filepath +str(id)+'.txt','r') as f:
        for line in f:
            df_star = list(map(float, line.split(',')))
        return df_star
def data_pre_process(a):
    out = ((a - a.min())*100)/(a.max()-a.min())
    return out
def read_data(df, filename):
    X_train = np.zeros((df.shape[0],2600),dtype = float)
    for i,id in enumerate(df['id']):
        a = np.array(read_data_byID(id,filename))
        X_train[i,:] =data_pre_process(a)
        if i%1000 == 0:
            print("reading %d files"%i)
    print('Finished reading files')
    return X_train

if __name__ == '__main__':
    df = pd.read_csv('../data/first_rank_index_20180307.csv')
    X_predict = read_data(df, '../data/first_rank_data/')
    unknown_mean_X = np.load('mean_X_two_classfier_unknown_model_ver0.0.5.npy')
    unknown_std_X = np.load('std_X_two_classfier_unknown_model_ver0.0.5.npy')
    unknown_model =  load_model('two_classfier_unknown_model_ver0.0.5.h5')
    X_unknown = ((X_predict - unknown_mean_X)/(unknown_std_X)).reshape((X_predict.shape[0],X_predict.shape[1],1))
    Y_unknown = unknown_model.predict(X_unknown, batch_size = 100)#.argmax(axis = 1)
    Y_predict = np.zeros(Y_unknown.shape[0],dtype = int)
    #Y_predict = Y_unknown * 3
    star_mean_X = np.load('mean_X_3classfier_model_ver0.0.4.npy')
    star_std_X = np.load('std_X_3classfier_model_ver0.0.4.npy')
    star_model =  load_model('3classfier_model_ver0.0.4.h5')
    X_3 = ((X_predict - star_mean_X)/(star_std_X)).reshape((X_predict.shape[0],X_predict.shape[1],1))
    Y_3 = star_model.predict(X_3, batch_size = 100)#.argmax(axis = 1)
    Y_predict = np.zeros(Y_unknown.shape[0],dtype = int)
    for i,pre in enumerate(Y_3):
        if ((Y_unknown[i,1]>Y_unknown[i,0])):#and(pre.max()<2322)):
            Y_predict[i] = 3
        elif (pre.max()<0.5):
            Y_predict[i] = 3
        else:
            Y_predict[i] = pre.argmax()
    Label_names = ['star','galaxy','qso','unknown']
    y_predict = [Label_names[i] for i in Y_predict]
    df_out = pd.DataFrame(df['id'])
    df_out = df_out.join(pd.DataFrame({"type":y_predict}))
    df_out.to_csv('rank005_004_0500.csv', header=False, index=False, sep=',')