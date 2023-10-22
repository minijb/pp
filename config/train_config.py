num_step = 30000

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
            "t_initial": num_step,
            "lr_min": 0.0001,
            "warmup_lr_init" : 0.0005,
            "warmup_t": num_step//5,
        }
    },
    "use_wandb" : True 
}
