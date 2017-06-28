import os
import numpy as np

import config as cfg
from lsun_room import Phase
from dataset import DataGenerator


from fcn import (
    fcn32s,
    sparse_pixelwise_accuracy,
    corssentropy2d
)


def main():
    os.makedirs('output/' + cfg.experiment_name, exist_ok=True)
    print('==> <Experiment>:', cfg.experiment_name)

    data_gen = DataGenerator()
    train_generator = data_gen.flow_from_directory(
        directory=cfg.dataset_root, phase=Phase.TRAIN,
        target_size=(cfg.size, cfg.size),
        batch_size=cfg.batch_size, shuffle=True)
    validation_generator = data_gen.flow_from_directory(
        directory=cfg.dataset_root, phase=Phase.VALIDATE,
        target_size=(cfg.size, cfg.size),
        batch_size=1)

    validation_data = [
        next(validation_generator)
        for _ in range(validation_generator.samples)]
    val_data_X = np.array([e[0][0] for e in validation_data])
    val_data_Y = np.array([e[1][0] for e in validation_data])

    model = fcn32s(
        input_shape=(cfg.size, cfg.size, 3), num_class=5,
        weights=cfg.initial_weight_path)
    model.summary()
    model.compile(
        optimizer=cfg.optimizer,
        loss=[corssentropy2d],
        metrics=[sparse_pixelwise_accuracy])
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=(train_generator.samples + 1) // cfg.batch_size,
        validation_data=(val_data_X, val_data_Y),
        epochs=cfg.epochs,
        workers=cfg.workers,
        callbacks=cfg.callbacks
    )


if __name__ == '__main__':
    main()
