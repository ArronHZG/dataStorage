from keras import models
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l2, l1


class Model():
    def __init__(self,input_dim=10):
        self.input_dim=input_dim
        self.model=self.model()
        # Now compile the network.
        optimizer = Adam(lr=1e-5, decay=1e-6)
        metrics = ['accuracy']
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    def model(self):
        model = models.Sequential()
        model.add(Dense(20, activation='relu', input_dim=self.input_dim))
        model.add(Dense(30, activation='relu',
                        kernel_regularizer=l2(0.01),
                        activity_regularizer=l1(0.01),
                        bias_regularizer=l2(0.001)))
        model.add(Dense(30, activation='relu',
                        kernel_regularizer=l2(0.01),
                        activity_regularizer=l1(0.01),
                        bias_regularizer=l2(0.001)))
        model.add(Dense(30, activation='relu',
                        kernel_regularizer=l2(0.01),
                        activity_regularizer=l1(0.01),
                        bias_regularizer=l2(0.001)))
        model.add(Dense(20, activation='relu',
                        kernel_regularizer=l2(0.01),
                        activity_regularizer=l1(0.01),
                        bias_regularizer=l2(0.001)))
        model.add(Dense(1, activation='sigmoid'))
        return model