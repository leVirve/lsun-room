import keras

experiment_name = 'vggbase_sgd_scratch_lrsched'
initial_weight_path = '../model_June13_sgd_60kitr.h5'

dataset_root = '../data'
size = 512
workers = 16
epochs = 50
batch_size = 8

optimizer = keras.optimizers.SGD(lr=1e-4, decay=0.0, momentum=0.99, nesterov=True)

callbacks = [
    keras.callbacks.CSVLogger('%s/training.log' % experiment_name),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-8),
    keras.callbacks.ModelCheckpoint(
        filepath='%s/weights.{epoch:02d}.hdf5' % experiment_name,
        verbose=1, save_best_only=True, save_weights_only=True),
    keras.callbacks.TensorBoard(
        log_dir='./logs/%s' % experiment_name,
        batch_size=batch_size,
        write_images=True, write_graph=False, histogram_freq=1)
]
