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

        'mg': 64,

        'gen_lr': 0.0002,
        'gen_beta1': 0.5,
        'disc_lr': 0.0002,
        'disc_beta1': 0.5,

        'extra_iteration_checkpoint_freq': 5000,
        'iteration_log_freq': 5000,
        'visualization_freq': 200,
        'metric_freq': 200,

        'class_loss_with_fake': False,
        'bal_class_weights': False,

        'num_generated_samples': 500000,
    },

    'test_config': [
        {
            'method': ['raregan'],
            'dataset': ['MNIST'],
            'bgt': [5000],
            'data_high_frc': [0.02, 0.01, 0.005, 0.002],
            'run': [0, 1, 2, 3, 4],

            'ini_rnd_bgt': [2500],
            'bgt_per_step': [2500],

            'high_frc_mul': [3.0],

            'bal_disc_weights': [True],
            'num_iters_per_step': [15000],
            'disc_disc_coe': [1.0],
            'gen_disc_coe': [1.0],
        },
        {
            'method': ['raregan'],
            'dataset': ['MNIST'],
            'bgt': [10000],
            'data_high_frc': [0.01],
            'run': [0, 1, 2, 3, 4],

            'ini_rnd_bgt': [5000],
            'bgt_per_step': [5000],

            'high_frc_mul': [3.0],

            'bal_disc_weights': [True],
            'num_iters_per_step': [15000],
            'disc_disc_coe': [1.0],
            'gen_disc_coe': [1.0],
        },
        {
            'method': ['raregan'],
            'dataset': ['MNIST'],
            'bgt': [2000],
            'data_high_frc': [0.01],
            'run': [0, 1, 2, 3, 4],

            'ini_rnd_bgt': [1000],
            'bgt_per_step': [1000],

            'high_frc_mul': [3.0],

            'bal_disc_weights': [True],
            'num_iters_per_step': [15000],
            'disc_disc_coe': [1.0],
            'gen_disc_coe': [1.0],
        },
        {
            'method': ['raregan'],
            'dataset': ['MNIST'],
            'bgt': [1000],
            'data_high_frc': [0.01],
            'run': [0, 1, 2, 3, 4],

            'ini_rnd_bgt': [500],
            'bgt_per_step': [500],

            'high_frc_mul': [3.0],

            'bal_disc_weights': [True],
            'num_iters_per_step': [15000],
            'disc_disc_coe': [1.0],
            'gen_disc_coe': [1.0],
        }
    ]
}
