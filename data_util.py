import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model


# Model visualization
def visual_model(model):
    plot_model(model, to_file='./image/model.png', show_shapes=True)


# Training history visualization

# Plot training & validation accuracy values
def visual_accuracy(history, title, url):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig(url)
    plt.show()


# Plot training & validation loss values
def visual_loss(history, title, url):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig(url)
    plt.show()


# Plot PR curve
def visual_pr(recall, precision, url):
    plt.plot(recall, precision, '-')
    plt.title('PR曲线')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig(url)


