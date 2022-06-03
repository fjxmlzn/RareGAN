from gpu_task_scheduler.gpu_task import GPUTask


class Task(GPUTask):
    def main(self):
        import random
        import tensorflow.compat.v1 as tf
        import os
        import sys
        import numpy as np
        from lib.gan.raregan import RareGAN
        from lib.gan.networks import ConditionalGenerator, ACDiscriminator
        from lib.data.load_data import load_data
        from lib.gan.metrics import FrechetInceptionDistance, NearestRealDistance

        random.seed(self._config['run'])
        np.random.seed(random.randint(0, 1000000))

        dataset = load_data(
            self._config['dataset'],
            data_high_fraction=self._config['data_high_frc'])
        sys.stdout.flush()
        height, width, depth = dataset.image_dims
        generator = ConditionalGenerator(
            output_width=width, output_height=height, output_depth=depth,
            mg=self._config['mg'])
        discriminator = ACDiscriminator(num_classes=2, mg=self._config['mg'])

        checkpoint_dir = os.path.join(self._work_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        sample_dir = os.path.join(self._work_dir, 'sample')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        time_path = os.path.join(self._work_dir, 'time.txt')
        metric_path = os.path.join(self._work_dir, 'metrics.csv')

        run_config = tf.ConfigProto()
        with tf.Session(config=run_config) as sess:
            metric_data = dataset.get_data(1)
            metric_callbacks = [
                FrechetInceptionDistance(sess=sess, real_images=metric_data),
                NearestRealDistance(sess=sess, real_images=metric_data)]
            gan = RareGAN(
                sess=sess,
                dataset=dataset,
                request_budget=self._config['bgt'],
                generator=generator,
                discriminator=discriminator,
                batch_size=self._config['batch_size'],
                z_dim=self._config['z_dim'],
                balanced_disc_weights=self._config['bal_disc_weights'],
                high_fraction_multiple=self._config['high_frc_mul'],
                class_loss_with_fake=self._config['class_loss_with_fake'],
                balanced_class_weights=self._config['bal_class_weights'],
                initial_random_budget=self._config['ini_rnd_bgt'],
                budget_per_step=self._config['bgt_per_step'],
                num_iterations_per_step=self._config['num_iters_per_step'],
                checkpoint_dir=checkpoint_dir,
                sample_dir=sample_dir,
                time_path=time_path,
                extra_iteration_checkpoint_freq=self._config[
                    'extra_iteration_checkpoint_freq'],
                iteration_log_freq=self._config['iteration_log_freq'],
                visualization_freq=self._config['visualization_freq'],
                metric_callbacks=metric_callbacks,
                metric_freq=self._config['metric_freq'],
                metric_path=metric_path,
                gen_lr=self._config['gen_lr'],
                gen_beta1=self._config['gen_beta1'],
                disc_lr=self._config['disc_lr'],
                disc_beta1=self._config['disc_beta1'],
                disc_disc_coe=self._config['disc_disc_coe'],
                gen_disc_coe=self._config['gen_disc_coe'])
            gan.build()
            gan.train()

            numpy_inputs = gan.sample_high(
                num_samples=self._config['num_generated_samples'])

            folder = os.path.join(self._work_dir, "generated_data")
            if not os.path.exists(folder):
                os.makedirs(folder)
            np.savez(
                os.path.join(folder, 'data.npz'),
                data=numpy_inputs)
