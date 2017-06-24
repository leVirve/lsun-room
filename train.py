import os

import config as cfg
from lsun_room import Phase
from dataset import DataGenerator


from fcn_models.fcn32s_vgg16 import fcn_vggbase
from fcn_models.fcn_score import (
    sparse_pixelwise_accuracy,
    corssentropy2d
)


def main():
    os.makedirs(cfg.experiment_name, exist_ok=True)

    data_gen = DataGenerator()
    train_generator = data_gen.flow_from_directory(
        directory=cfg.dataset_root, phase=Phase.TRAIN,
        target_size=(cfg.size, cfg.size),
        batch_size=cfg.batch_size, shuffle=True)
    validation_generator = data_gen.flow_from_directory(
        directory=cfg.dataset_root, phase=Phase.VALIDATE,
        target_size=(cfg.size, cfg.size),
        batch_size=cfg.batch_size)

    model = fcn_vggbase(
        input_shape=(cfg.size, cfg.size, 3),
        pretrained_weights=cfg.initial_weight_path)
    model.summary()
    model.compile(
        optimizer=cfg.optimizer,
        loss=[corssentropy2d],
        metrics=[sparse_pixelwise_accuracy])
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=(train_generator.samples + 1) // cfg.batch_size,
        validation_data=validation_generator,
        validation_steps=(validation_generator.samples + 1) // cfg.batch_size,
        epochs=cfg.epochs,
        workers=cfg.workers,
        callbacks=cfg.callbacks
    )


if __name__ == '__main__':
    main()
