from AutoEncoder.sparse_AE.sae import SAE
from AutoEncoder.sparse_AE.data import Data
from AutoEncoder.sparse_AE.learner import Learner
from AutoEncoder.sparse_AE.visualizer import  Visualizer

if __name__ == '__main__':
    DATA_DIR = 'D:/rawDataFiles/digit_train.csv'

    BETA = 2
    LEARNING_RATE = 0.001
    EPOCHS = 300
    BATCH_SIZE = 3000

    data = Data(DATA_DIR)
    target, input = data.import_data()
    sae = SAE(LEARNING_RATE, BATCH_SIZE)
    target = target.to(sae.device)
    input = input.to(sae.device)
    print(target.shape)
    print(input.shape)

    learner = Learner(sae, input, target, batch_size=BATCH_SIZE, epochs=EPOCHS)
    model = learner.learn(BETA)

    viz = Visualizer(target, model)
    viz.viz()
