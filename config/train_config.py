num_step = 5000

trian = {
    "pretrain" : {
        "lr" : 0.0005,
        "test_step" : 5000,
        "num_step" : 3000,
    },
    "train" : {
        "focal_gamma": 4,
        "focal_alpha" : None,
        "lr" : 0.003,
        "num_step" : num_step,
        "scheduler": {
            "first_cycle_steps": num_step,
            "max_lr" : 0.003,
            "min_lr" : 0.0001,
            "warmup_steps" : int(num_step * 0.1)
        }

    },
    "use_wandb" : True 
}
