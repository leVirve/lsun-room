from keras.optimizers import RMSprop

from lsun_room import Phase
from dataset import DataGenerator

from fcn_models.fcn32s_vgg16 import fcn_vggbase
from fcn_models.fcn_score import (
    softmax_sparse_crossentropy_ignoring_last_label,
    sparse_accuracy_ignoring_last_label
)


def main():

    size = 512
    workers = 32
    epochs = 2
    batch_size = 16

    data_gen = DataGenerator()
    train_generator = data_gen.flow_from_directory(
        directory='../data', phase=Phase.TRAIN,
        target_size=(size, size),
        batch_size=4, shuffle=True)
    validation_generator = data_gen.flow_from_directory(
        directory='../data', phase=Phase.VALIDATE,
        target_size=(size, size),
        batch_size=4)

    model = fcn_vggbase(
        input_shape=(size, size, 3),
        pretrained_weights='../model_June13_sgd_60kitr.h5')
    model.summary()
    model.compile(
        optimizer=RMSprop(lr=1e-4),
        loss=[softmax_sparse_crossentropy_ignoring_last_label],
        metrics=[sparse_accuracy_ignoring_last_label])
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=(train_generator.samples + 1) // batch_size,
        validation_data=validation_generator,
        validation_steps=(validation_generator.samples + 1) // batch_size,
        epochs=epochs,
        workers=workers
    )


if __name__ == '__main__':
    main()
