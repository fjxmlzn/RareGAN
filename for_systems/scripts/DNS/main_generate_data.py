if __name__ == "__main__":
    from .task_generate_data import Task
    from .config_generate_data import config
    from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler
    
    scheduler = GPUTaskScheduler(config=config, gpu_task_class=Task)
    scheduler.start()
