import matplotlib.pyplot as plt
import numpy as np
import torch as T

class Visualizer :
    def __init__(self, target, model):
        self.target = target
        self.model = model
        self.samples = self.random_select()

    def random_select(self):
        rndm = np.random.randint(0,self.target.shape[0]-1,10)
        return rndm

    def viz(self):
        for _, i in enumerate(self.samples):
            output = self.model.forward(self.target[i])[1].view(28,28)
            cat_img = T.cat((self.target[i].view(28,28),output), 1)
            plt.imshow(cat_img.cpu().detach().numpy())
            plt.title('Sparse AutoEncoder target vs output')
            plt.savefig('./output_images/sample'+str(_+1))
            plt.close()
