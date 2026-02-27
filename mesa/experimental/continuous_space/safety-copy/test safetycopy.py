from mesa.experimental.continuous_space.property_layer import PropertyLayer, HasPropertyLayers

import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import time 
from mesa.experimental.continuous_space import ContinuousSpaceAgent, ContinuousSpace
import numpy as np
from mesa.model import Model 


def plotheatmap(m, name):
    plt.imshow(m, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Kernel Weight')
    plt.title('Kernel Matrix Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(name) 
    plt.close() 

class AgentT(ContinuousSpaceAgent):
    def __init__(self, model):
        super().__init__(model.space, model)
        self.model=model
        self.space=model.space
        self.position = tuple(
            np.random.uniform(low, high) for (low, high) in self.space.dimensions
        )

    def step(self):
        # self.position = tuple(
        #     np.random.uniform(low, high) for (low, high) in self.space.dimensions
        # )
        self.position+=tuple([1,1])

class ModelT(Model, HasPropertyLayers):
    def __init__(self, rng, width, height, n_agents):
        super().__init__()

        self.space = ContinuousSpace(
            [[0, width], [0, height]],
            torus=True,
            random=self.random,
            n_agents=n_agents
        )

        self.create_property_layer(
            name="step_count",
            space=self.space,
            resolution=2,
            default_value=0,
            dtype=float
        )

        AgentT.create_agents(model=self, n=n_agents)

    def step(self):
        self.agents.shuffle_do("step")
        pos=self.space.agent_positions
        self.step_count.deposit_splat(
            pos=pos, 
            value=2,
            mode="add",
            kernel="gaussian",
            spread=1,
            torus=False
        )
        self.step_count.decay(
            T=100,
            type="exponential",
            k=1.0
        )

        if "randomL" in self._property_layers:
            self.randomL.data[:]+=np.indices((200, 200)).sum(axis=0)
            mask=self.randomL.get_neighborhood_mask(pos[0], 3, True)
            self.randomL.data[mask]=0



if __name__ == "__main__":

    d=np.random.rand(50, 50)

    model=ModelT(
        rng=None,
        width=50,
        height=50,
        n_agents=5
    )

    layer2=PropertyLayer(
        name="randomL",
        bounds=(50,50),
        resolution=4,
        default_value=2,
        dtype=float
    )
    layer3=PropertyLayer.from_data(
        name="from_data",
        data=d,
        bounds=[[0,50], [0,50]],
        resolution=1
    )

    model.run_for(100)
    plotheatmap(model.step_count.data, "step_count")
    model.add_property_layer(layer2, model.space)
    model.run_for(5)
    plotheatmap(model.randomL.data, "randomL")
    model.add_property_layer(layer3, model.space)
    plotheatmap(model.from_data.data, "from-data")

    # print(model.step_count.aggregate(operation=np.mean))




    # print(model._property_layers)
    # model.remove_property_layer("step_count")
    # print("---")
    # print(model._property_layers)

    # layer=PropertyLayer(
    #     name="trial",
    #     bounds=(100,100),
    #     resolution=2,
    #     default_value=1,
    #     dtype=float
    # )   


    # layer.deposit_splat(
    #     pos=[(1.2,66.8), (91.7, 55.4), (32, 78)], 
    #     value=5, 
    #     mode="add",
    #     alpha=None, 
    #     kernel="gaussian",
    #     spread=10, 
    #     torus=False
    # )
    # mask=layer.get_neighborhood_mask((78.9,25.4), 8, False)
    # T=5
    # for _ in range (T):
    #     layer.decay(T, "exponential", 2)
    #     plotheatmap(layer.data)
    #     time.sleep(5)

