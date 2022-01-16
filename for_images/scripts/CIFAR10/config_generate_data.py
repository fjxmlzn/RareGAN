config = {
    'scheduler_config': {
        'gpu': ['0'],
        'temp_folder': 'temp',
        'scheduler_log_file_path': 'scheduler.log',
        'log_file': 'worker.log',
        'config_string_value_maxlen': 1000,
        'ignored_keys_for_folder_name': []
    },

    'global_config': {
        'batch_size': 100,
        'z_dim': 100,

        'mg': 256,

        'gen_lr': 0.0002,
        'gen_beta1': 0.5,
        'disc_lr': 0.0002,
        'disc_beta1': 0.5,

        'extra_iteration_checkpoint_freq': 50000,
        'iteration_log_freq': 50000,
        'visualization_freq': 200,
        'metric_freq': 400,

        'class_loss_with_fake': False,
        'bal_class_weights': False,

        'num_generated_samples': 500000,
    },

    'test_config': [
        {
            'method': ['raregan'],
            'dataset': ['CIFAR10'],
            'bgt': [10000],
            'data_high_frc': [0.1, 0.08, 0.05],
            'run': [0, 1, 2, 3, 4],

            'ini_rnd_bgt': [5000],
            'bgt_per_step': [5000],

            'high_frc_mul': [3.0],

            'bal_disc_weights': [True],
            'num_iters_per_step': [100000],
            'disc_disc_coe': [1.0],
            'gen_disc_coe': [1.0],
        },
        {
            'method': ['raregan'],
            'dataset': ['CIFAR10'],
            'bgt': [5000],
            'data_high_frc': [0.1],
            'run': [0, 1, 2, 3, 4],

            'ini_rnd_bgt': [2500],
            'bgt_per_step': [2500],

            'high_frc_mul': [3.0],

            'bal_disc_weights': [True],
            'num_iters_per_step': [100000],
            'disc_disc_coe': [1.0],
            'gen_disc_coe': [1.0],
        },
        {
            'method': ['raregan'],
            'dataset': ['CIFAR10'],
            'bgt': [8000],
            'data_high_frc': [0.1],
            'run': [0, 1, 2, 3, 4],

            'ini_rnd_bgt': [4000],
            'bgt_per_step': [4000],

            'high_frc_mul': [3.0],

            'bal_disc_weights': [True],
            'num_iters_per_step': [100000],
            'disc_disc_coe': [1.0],
            'gen_disc_coe': [1.0],
        },
    ]
}
