# This is a training script for T1 generation to be run in a distributed manner on a DGX Station with x gpus

import nobrainer
import tensorflow as tf
from pathlib import Path

latent_size = 1024
g_fmap_base = 4096
d_fmap_base = 4096
num_gpus = 4
num_parallel_calls = 4
iterations = int(200e3)
lr = 1.25e-4
save_dir = 'results/run_t1'

save_dir = Path(save_dir)s
generated_dir = save_dir.joinpath('generated')
model_dir = save_dir.joinpath('saved_models')
log_dir = save_dir.joinpath('logs')

save_dir.mkdir(exist_ok=True)
generated_dir.mkdir(exist_ok=True)
model_dir.mkdir(exist_ok=True)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    generator, discriminator = nobrainer.models.progressivegan(latent_size, label_size=label_size, g_fmap_base=g_fmap_base, d_fmap_base=d_fmap_base)

resolution_batch_size_map = {8: 32, 16: 16, 32: 8, 64: 8, 128: 4, 256: 1}
resolution_batch_size_map = {k : v*num_gpus for k,v in resolution_batch_size_map.items()}
resolutions = sorted(list(resolution_batch_size_map.keys()))

for resolution in resolutions:

    dataset_train = nobrainer.dataset.get_dataset(
        file_pattern="/tfrecords/kwyk_biobank/*res-%03d*.tfrec*"%(resolution),
        batch_size=resolution_batch_size_map[resolution],
        num_parallel_calls=num_parallel_calls,
        volume_shape=(resolution, resolution, resolution),
        n_classes=label_size, # dummy labels
        scalar_label=True,
        standardize=False
    )

    with strategy.scope():
        generator.add_resolution()
        discriminator.add_resolution()

        progressive_gan_trainer = nobrainer.training.ProgressiveGANTrainer(
            generator=generator,
            discriminator=discriminator,
            gradient_penalty=True)

        progressive_gan_trainer.compile(
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
            g_loss_fn=nobrainer.losses.wasserstein,
            d_loss_fn=nobrainer.losses.wasserstein
            )

    steps_per_epoch = iterations//resolution_batch_size_map[resolution]

    logger = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), update_freq='batch')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(str(model_dir), save_weights_only=True, save_freq=10, save_best_only=False)

    print('Resolution : {}'.format(resolution))

    print('Transition phase')
    progressive_gan_trainer.fit(
        dataset_train,
        phase='transition',
        resolution=resolution,
        steps_per_epoch=steps_per_epoch,
        callbacks=[model_checkpoint_callback])

    print('Resolution phase')
    progressive_gan_trainer.fit(
        dataset_train,
        phase='resolution',
        resolution=resolution,
        steps_per_epoch=steps_per_epoch,
        callbacks=[model_checkpoint_callback])

    generator.save(str(model_dir.joinpath('generator_final_res_{}'.format(resolution))))