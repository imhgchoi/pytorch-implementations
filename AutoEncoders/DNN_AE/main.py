from AutoEncoder.DNN_AE.dnn_ae import DNN_AE
from AutoEncoder.DNN_AE.data import Data
from AutoEncoder.DNN_AE.learner import Learner
from AutoEncoder.DNN_AE.visualizer import  Visualizer

if __name__ == '__main__':
    DATA_DIR = 'D:/rawDataFiles/digit_train.csv'

    LEARNING_RATE = 3e-5
    EPOCHS = 500
    BATCH_SIZE = 3000

    data = Data(DATA_DIR)
    target, input = data.import_data()
    dae = DNN_AE(LEARNING_RATE)
    target = target.to(dae.device)
    input = input.to(dae.device)
    print(target.shape)
    print(input.shape)

    learner = Learner(dae, input, target, batch_size=BATCH_SIZE, epochs=EPOCHS)
    model = learner.learn()

    viz = Visualizer(target, model)
    viz.viz()
