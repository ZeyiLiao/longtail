import wandb
import numpy as np
import random


# 🐝 Step 1: Define training function that takes in hyperparameter
# values from `wandb.config` and uses them to train a model and return metric
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main():
    # Use the wandb.init() API to generate a background process
    # to sync and log data as a Weights and Biases run.
    # Optionally provide the name of the project.
    config = {'lr': 0.001, "bs": 16, "epochs": 6}
    wandb.init(project='test',
                     entity='zeyiliao',
                     tags=['what is this'],
                     config=config,name = 'what if have conflict name')

    # note that we define values from `wandb.config` instead of
    # defining hard values
    lr = wandb.config.lr
    bs = wandb.config.bs
    epochs = wandb.config.epochs
    train_loss_all = []
    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        train_loss_all.append(train_loss)
        val_acc, val_loss = evaluate_one_epoch(epoch)
        # wandb.define_metric("train_loss", summary="min")
        # # define a metric we are interested in the maximum of
        # wandb.define_metric("train_acc", summary="max")
        # wandb.define_metric("val_loss", summary="min")
        # # define a metric we are interested in the maximum of
        # wandb.define_metric("val_acc", summary="max")
        my_table = wandb.Table(columns=["a", "b_test"], data=[["1a", "1b"], ["2a", "2b"]])

        wandb.log({
            'epoch': epoch,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_loss': val_loss,
            "table_key": my_table

        })
    # wandb.run.summary['min_loss_train'] = np.min(train_loss_all)
    # print(wandb.run.summary)

main()