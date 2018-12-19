"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import Model
from data import DataSet
import time
import os.path
import os

def train(saved_model=None,
          load_to_memory=False,
          batch_size=32,
          nb_epoch=100):
    # Helper: Save the model.
    if not os.path.exists(os.path.join('data', 'checkpoints')):
        os.makedirs(os.path.join('data', 'checkpoints'))

    if not os.path.exists(os.path.join('data', 'logs')):
        os.makedirs(os.path.join('data', 'logs'))

    if not os.path.exists(os.path.join('data', 'checkpoints')):
        os.makedirs(os.path.join('data', 'checkpoints'))

    check_pointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints','val_loss-{val_loss:.3f}_val_acc-{val_acc:3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs'))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=2)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs','training-' + str(timestamp) + '.log'))

    # Get the data and process it.

    data = DataSet()
    # X,Y,X_test,Y_test,generator,val_generator=None


    if load_to_memory:
        # Get data.
        X, Y = data.get_memory('train')
        X_test, Y_test = data.get_memory('test')
    else:
        # Get generators.
        generator = data.generator(batch_size, 'train')
        val_generator = data.generator(batch_size, 'test')

    # Get the model.
    model=Model(10).model
    # Fit!
    if load_to_memory:
        # Use standard fit.
        model.fit(
            X,
            Y,
            batch_size=batch_size,
            validation_data=(X_test, Y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, check_pointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        steps_per_epoch = 3500 // batch_size
        model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, check_pointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=10,
            epochs=nb_epoch)


def main():

    saved_model = None
    load_to_memory = True  # pre-load the sequences into memory
    batch_size = 32
    nb_epoch = 1000  # 一般使用early_stopper = EarlyStopping(patience=10)提前终止运行

    train(saved_model=saved_model,
          load_to_memory=load_to_memory,
          batch_size=batch_size,
          nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
