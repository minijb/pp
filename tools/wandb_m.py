import wandb


def wandb_init(project_name: str, model_name: str, cfg):
    wandb.init(project = project_name, name = model_name, config = cfg)
    
def wandb_log_eval(eval_metrics, step):
    wandb.log(eval_metrics, step=step)
    
def wandb_log_train(train_information, step):
    wandb.log(train_information, step=step)
