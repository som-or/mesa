from mesa.experimental.continuous_space.property_layer import PropertyLayer, HasPropertyLayers

import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import time 
from mesa.experimental.continuous_space import ContinuousSpaceAgent, ContinuousSpace
import numpy as np
from mesa.model import Model 


from mesa.visualization import SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle,  PropertyLayerStyle
import solara
from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter



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
        self.position+=tuple([0.5,0.5])

class ModelT(Model):
    def __init__(self, rng, width, height, n_agents):
        super().__init__()

        self.space = ContinuousSpace(
            [[0, width], [0, height]],
            torus=True,
            random=self.random,
            n_agents=n_agents
        )

        self.space.create_property(
            name="step_count",
            resolution=5,
            default_value=0,
            dtype=float
        )
        d=np.random.rand(10, 10)
        self.space.add_property("from_data", d)

        AgentT.create_agents(model=self, n=n_agents)

    def step(self):
        self.agents.shuffle_do("step")
        pos=self.space.agent_positions
        self.space.step_count.deposit_splat(
            pos=pos, 
            value=2,
            mode="add",
            kernel="gaussian",
            spread=0.5
        )
        self.space.step_count.decay(
            T=100,
            type="exponential",
            k=1.0
        )

        d=np.random.rand(10, 10)
        self.space.from_data.data=d

        if "randomL" in self.space._property:
            self.space.randomL.data+=np.indices((200, 200)).sum(axis=0)
            mask=self.space.randomL.get_neighborhood_mask(pos[0], 3, True)
            self.space.randomL.data[mask]=0



if __name__ == "__main__":

    d=np.random.rand(10, 10)

    model=ModelT(
        rng=None,
        width=10,
        height=10,
        n_agents=5
    )

    layer2=PropertyLayer(
        name="randomL",
        bounds=(10,10),
        resolution=4,
        default_value=2
    )

    # layer3=PropertyLayer.from_data(
    #     name="from_data",
    #     data=d,
    #     bounds=[[0,50], [0,50]],
    #     resolution=1
    # )

    # model.run_for(100)
    # plotheatmap(model.space.step_count.data, "step_count")
    # model.space._attach_property(layer2)
    # model.run_for(5)
    # plotheatmap(model.space.randomL.data, "randomL")
    # model.space.add_property("from_data", d)
    # plotheatmap(model.space.from_data.data, "from-data")

    # print(model.step_count.aggregate(operation=np.mean))

    def agent_portrayal(agent):
        return AgentPortrayalStyle(size=10, marker='o', color='red')
        

    def propertylayer_portrayal(layer):
        if layer.name=="step_count":
            return PropertyLayerStyle(color='blue', alpha=0.8, colorbar=True)
        if layer.name=="from_data":
            return PropertyLayerStyle(color='green', alpha=0.5, colorbar=True)
        
        
        
    model_params={
        "rng":None,
        "width":10,
        "height":10,
        "n_agents":5
    }

    renderer = (
        SpaceRenderer(
            model,
            backend="altair",
        )
    )
    renderer.draw_agents(agent_portrayal)
    renderer.draw_propertylayer(propertylayer_portrayal)

    page = SolaraViz(
        model,
        renderer,
        model_params=model_params,
        name="Actice Walker Model",
    )

    page

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

