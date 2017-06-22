import os
import keras

from lsun_room import Phase
from dataset import DataGenerator


from fcn_models.fcn32s_vgg16 import fcn_vggbase
from fcn_models.fcn_score import (
    sparse_pixelwise_accuracy,
    corssentropy2d
)


def main():
    experiment_name = 'vggbase_go_adam_lr1e-4'
    initial_weight_path = '../model_June13_sgd_60kitr.h5'

    dataset_root = '../data'
    size = 512
    workers = 16
    epochs = 50
    batch_size = 8

    optimizer = keras.optimizers.Adam(lr=1e-4)

    callbacks = [
        keras.callbacks.CSVLogger('%s/training.log' % experiment_name),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8),
        keras.callbacks.ModelCheckpoint(
            filepath='%s/weights.{epoch:02d}.hdf5' % experiment_name,
            verbose=1, save_best_only=True, save_weights_only=True),
        keras.callbacks.TensorBoard(
            log_dir='./logs/%s' % experiment_name,
            batch_size=batch_size, write_graph=False)
    ]

    os.makedirs(experiment_name, exist_ok=True)

    data_gen = DataGenerator()
    train_generator = data_gen.flow_from_directory(
        directory=dataset_root, phase=Phase.TRAIN,
        target_size=(size, size),
        batch_size=batch_size, shuffle=True)
    validation_generator = data_gen.flow_from_directory(
        directory=dataset_root, phase=Phase.VALIDATE,
        target_size=(size, size),
        batch_size=batch_size)

    model = fcn_vggbase(
        input_shape=(size, size, 3),
        pretrained_weights=initial_weight_path)
    model.summary()
    model.compile(
        optimizer=optimizer,
        loss=[corssentropy2d],
        metrics=[sparse_pixelwise_accuracy])
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
