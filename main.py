from data_acquisition import load_breastmnist
from training import trainCNN
from plotting import plot_loss_acc


seed=[0,1,2,3,4]
result = []
train, val, test = load_breastmnist()
for s in seed:


    train_loss, train_acc, val_loss, val_acc, epoch = trainCNN(0.0001, 100, train, val, 1, 2, s)

    result.append({
        "seed":s,
        "train_loss":train_loss,
        "train_acc" : train_acc,
        "val_loss" : val_loss,
        "val_acc" : val_acc
    })
plot_loss_acc(result, 'initial')



