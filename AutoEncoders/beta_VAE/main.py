from AutoEncoder.VAE.vae import VAE
from AutoEncoder.VAE.data import Data
from AutoEncoder.VAE.learner import Learner
from AutoEncoder.VAE.visualizer import  Visualizer

if __name__ == '__main__':
    DATA_DIR = 'D:/rawDataFiles/digit_train.csv'

    LEARNING_RATE = 0.0005
    EPOCHS = 500
    BATCH_SIZE = 3000

    data = Data(DATA_DIR)
    target, input = data.import_data()
    vae = VAE(LEARNING_RATE)
    target = target.to(vae.device)
    input = input.to(vae.device)
    print(target.shape)
    print(input.shape)

    learner = Learner(vae, input, target, batch_size=BATCH_SIZE, epochs=EPOCHS)
    model = learner.learn()

    viz = Visualizer(target, model)
    viz.viz()
