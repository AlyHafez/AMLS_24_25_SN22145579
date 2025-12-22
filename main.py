from data_acquisition import load_breastmnist
from training import trainCNN
from plotting import plot_train_loss
train, val, test = load_breastmnist()


train_loss, train_acc, val_loss, val_acc, epoch = trainCNN(0.001, 40, train, val, 1, 2)
plot_train_loss(train_loss, train_acc, val_loss,val_acc, epoch, 'initial')



