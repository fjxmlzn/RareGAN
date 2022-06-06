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
        from lib.gan.amplification_utils import get_target_threshold_from_config
        from lib.data.dataset import Dataset
        from lib.data.input_definition import InputDefinition

        from blackboxes.blackbox_utils import create_blackbox_from_config, close_blackbox_from_config

        random.seed(self._config['run'])
        np.random.seed(random.randint(0, 1000000))

        input_definition = InputDefinition(
            self._config['input_definition_file'])
        blackbox, auxiliary = create_blackbox_from_config(
            self._config, input_definition)

        if 'dataset_folder' in self._config:
            dataset = Dataset(self._config['dataset_folder'])
        else:
            dataset = Dataset()
        generator = ConditionalGenerator(
            num_layers=self._config['gen_num_layers'],
            l_dim=self._config['gen_l_dim'],
            input_definition=input_definition)
        discriminator = ACDiscriminator(
            num_shared_layers=self._config['disc_num_shared_layers'],
            num_disc_layers=self._config['disc_num_disc_layers'],
            num_class_layers=self._config['disc_num_class_layers'],
            l_dim=self._config['disc_l_dim'],
            num_classes=2)

        target_threshold = get_target_threshold_from_config(
            self._config, blackbox, input_definition)
        print('Target threshold: {}'.format(target_threshold))
        sys.stdout.flush()

        checkpoint_dir = os.path.join(self._work_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        time_path = os.path.join(self._work_dir, 'time.txt')

        run_config = tf.ConfigProto()
        with tf.Session(config=run_config) as sess:
            gan = RareGAN(
                sess=sess,
                blackbox=blackbox,
                dataset=dataset,
                input_definition=input_definition,
                request_budget=self._config['bgt'],
                target_threshold=target_threshold,
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
                oversampling_ratio=self._config['oversampling_ratio'],
                num_iterations_per_step=self._config['num_iters_per_step'],
                checkpoint_dir=checkpoint_dir,
                time_path=time_path,
                extra_iteration_checkpoint_freq=self._config[
                    'extra_iteration_checkpoint_freq'],
                iteration_log_freq=self._config['iteration_log_freq'],
                gen_lr=self._config['gen_lr'],
                gen_beta1=self._config['gen_beta1'],
                disc_lr=self._config['disc_lr'],
                disc_beta1=self._config['disc_beta1'],
                disc_gp_coe=self._config['disc_gp_coe'],
                disc_disc_coe=self._config['disc_disc_coe'],
                gen_disc_coe=self._config['gen_disc_coe'])
            gan.build()
            gan.train()

            generated_dataset = Dataset()
            numpy_inputs = gan.sample(
                num_samples=self._config['num_generated_samples'],
                condition=1)[0]

            field_dict_inputs = input_definition.numpy_to_field_dict(
                numpy_inputs)
            amplifications = blackbox.query(field_dict_inputs)
            generated_dataset.add_data(
                numpy_inputs=numpy_inputs,
                amplifications=amplifications,
                steps=np.zeros(numpy_inputs.shape[0], dtype=np.int32),
                conditions=np.ones(numpy_inputs.shape[0], dtype=np.int32))

            folder = os.path.join(self._work_dir, "generated_data")
            generated_dataset.dump_to_folder(folder)

        close_blackbox_from_config(self._config, blackbox, auxiliary)
