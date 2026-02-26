from mesa.experimental.continuous_space.property_layer import PropertyLayer, HasPropertyLayers

import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import time 
from mesa.experimental.continuous_space import ContinuousSpaceAgent, ContinuousSpace
import numpy as np
from mesa.model import Model 


def plotheatmap(m):
    plt.imshow(m, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Kernel Weight')
    plt.title('Kernel Matrix Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig("plot") 
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
        self.position = tuple(
            np.random.uniform(low, high) for (low, high) in self.space.dimensions
        )

class ModelT(Model, HasPropertyLayers):
    def __init__(self, rng, width, height, n_agents):
        super().__init__()

        self.space = ContinuousSpace(
            [[0, width], [0, height]],
            torus=False,
            random=self.random,
            n_agents=n_agents,
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
            kernel="epanechnikov",
            spread=1,
            torus=False
        )
        self.step_count.decay(
            T=100,
            type="exponential",
            k=1.0
        )



if __name__ == "__main__":

    model=ModelT(
        rng=None,
        width=50,
        height=50,
        n_agents=5
    )

    model.run_for(100)
    plotheatmap(model.step_count.data)


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

