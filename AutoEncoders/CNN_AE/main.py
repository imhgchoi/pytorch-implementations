from AutoEncoder.CNN_AE.cnn_ae import CNN_AE
from AutoEncoder.CNN_AE.data import Data
from AutoEncoder.CNN_AE.learner import Learner
from AutoEncoder.CNN_AE.visualizer import  Visualizer

if __name__ == '__main__':
    DATA_DIR = 'D:/rawDataFiles/digit_train.csv'

    LEARNING_RATE = 0.003
    EPOCHS = 200
    BATCH_SIZE = 3000

    data = Data(DATA_DIR)
    target, input = data.import_data()
    cae = CNN_AE(LEARNING_RATE)
    target = target.to(cae.device)
    input = input.to(cae.device)
    print(target.shape)
    print(input.shape)

    learner = Learner(cae, input, target, batch_size=BATCH_SIZE, epochs=EPOCHS)
    model = learner.learn()

    viz = Visualizer(target, model)
    viz.viz()
