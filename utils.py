import matplotlib.pyplot as plt

def save_loss_plot(train,test):
    plt.plot(train)
    plt.plot(test)
    plt.legend(['train','test'])
    plt.savefig('plots/loss.png')
    plt.close('all')