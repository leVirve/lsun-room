import os
import keras

from lsun_room import Phase
from dataset import DataGenerator

from fcn_models.fcn32s_vgg16 import fcn_vggbase
from fcn_models.fcn_score import (
    softmax_sparse_crossentropy_ignoring_last_label,
    sparse_accuracy_ignoring_last_label
)


def main():

    experiment_name = 'vggbase_adam_lr1e-4'

    size = 512
    workers = 16
    epochs = 50
    batch_size = 8

    optimizer = keras.optimizers.Adam(lr=1e-4)

    callbacks = [
        keras.callbacks.CSVLogger('output/training.log'),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8),
        keras.callbacks.ModelCheckpoint(
            filepath='output/weights.{epoch:02d}.hdf5',
            verbose=1, save_best_only=True, save_weights_only=True),
        keras.callbacks.TensorBoard(
            log_dir='./logs/%s' % experiment_name,
            batch_size=batch_size, write_graph=False)
    ]

    data_gen = DataGenerator()
    train_generator = data_gen.flow_from_directory(
        directory='../data', phase=Phase.TRAIN,
        target_size=(size, size),
        batch_size=batch_size, shuffle=True)
    validation_generator = data_gen.flow_from_directory(
        directory='../data', phase=Phase.VALIDATE,
        target_size=(size, size),
        batch_size=batch_size)
    os.makedirs('output/', exist_ok=True)

    model = fcn_vggbase(
        input_shape=(size, size, 3),
        pretrained_weights='../model_June13_sgd_60kitr.h5')
    model.summary()
    model.compile(
        optimizer=optimizer,
        loss=[softmax_sparse_crossentropy_ignoring_last_label],
        metrics=[sparse_accuracy_ignoring_last_label])
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=(train_generator.samples + 1) // batch_size,
        validation_data=validation_generator,
        validation_steps=(validation_generator.samples + 1) // batch_size,
        epochs=epochs,
        workers=workers,
        callbacks=callbacks
    )


if __name__ == '__main__':
    main()
