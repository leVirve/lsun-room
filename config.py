import keras

experiment_name = 'vgg16_adam'
initial_weight_path = '../model_June13_sgd_60kitr.h5'

dataset_root = '../data'
size = 512
workers = 4
epochs = 50
batch_size = 8

optimizer = keras.optimizers.adam(lr=1e-4)

callbacks = [
    keras.callbacks.CSVLogger('output/%s/training.log' % experiment_name),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8),
    keras.callbacks.ModelCheckpoint(
        filepath='output/%s/weights.{epoch:02d}.hdf5' % experiment_name,
        verbose=1, save_best_only=True, save_weights_only=True),
    keras.callbacks.TensorBoard(
        log_dir='./logs/%s' % experiment_name,
        batch_size=batch_size,
        write_images=True, histogram_freq=1)
]
