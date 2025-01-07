
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, LSTM, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

class models():
    def __init__(self, path):
        self.path = path
       
    
    def read_h5(self):
        f = h5py.File(self.path, 'r')
        X = f.get('inputs')
        y = f.get('labels') 
        X = np.array(X)
        y = np.array(y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.4, random_state = 1)
        print(self.x_train.shape)
    
    def cnn_model(self):
        K = len(set(self.y_train))
        self.x_train = np.expand_dims(self.x_train, -1)
        self.x_test = np.expand_dims(self.x_test,-1)
        i = Input(shape=self.x_train[0].shape)
        x = Conv2D(32, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(i)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2D(64, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Conv2D(64, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)    
        x = Dropout(0.2)(x)
        x = Dense(512,activation = 'relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(K, activation = 'softmax')(x)       
        self.model = Model(i,x)
        self.model.compile(optimizer = Adam(learning_rate=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

        self.r = self.model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), epochs = 50, batch_size = 32 )
        print(self.model.summary())
        return self.r
    
    def dnn_model(self):
        K = len(set(self.y_train))
        i = Input(shape=self.x_train[0].shape)
        x = Flatten()(i)
        x = Dense(64,activation = 'relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128,activation = 'relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64,activation = 'relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(K,activation = 'softmax')(x)
        self.model = Model(i,x)      
        self.model.compile(optimizer = Adam(learning_rate=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
        
        self.r = self.model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), epochs = 50, batch_size = 32 )
        print(self.model.summary())
        return self.r
    

    def rnn_model(self):
        K = len(set(self.y_train))
        i = Input(shape = self.x_train[0].shape)
        x = LSTM(128, return_sequences=True)(i)
        x = Dense(64,activation = 'relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(K,activation = 'softmax')(x)
        self.model = Model(i,x)      
        self.model.compile(optimizer = Adam(learning_rate=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
        self.r = self.model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), epochs = 50, batch_size = 32 )
        print(self.model.summary())
        return self.r
   
    def draw(self):
        f1 = plt.figure(1)
        plt.title('Loss')
        plt.plot(self.r.history['loss'], label='loss')
        plt.plot(self.r.history['val_loss'], label='val_loss')
        plt.legend()
        f1.show()
    
        f2 = plt.figure(2)
        plt.title('Accuracy')
        plt.plot(self.r.history['accuracy'], label='accuracy')
        plt.plot(self.r.history['val_accuracy'], label='val_accuracy')
        plt.legend()
        f2.show()
        
    def con_matrix(self):
        K = len(set(self.y_train))
        self.y_pred = self.model.predict(self.x_test).argmax(axis=1)
        cm = confusion_matrix(self.y_test,self.y_pred)
        self.plot_confusion_matrix(cm,list(range(K)))
        

    def plot_confusion_matrix(self, cm, classes, normalize = False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")
        print(cm)
        f3 = plt.figure(3)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd' 
        thresh = cm.max()/2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment = "center",
                     color = "white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('predicted label')
            f3.show()
                


if __name__ == "__main__":
    model_name = "dnn" # can be cnn/dnn/rnn
    path = "./uci_har.h5"
    har = models(path)
    print("read h5 file....")
    har.read_h5()
    
    if model_name == "cnn":
        har.cnn_model()
    elif model_name == "dnn":
        har.dnn_model()
    elif model_name == "rnn":
        har.rnn_model()
    har.draw()
    har.con_matrix()