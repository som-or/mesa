from mesa.experimental.continuous_space.property_layer import PropertyLayer

import matplotlib.pyplot as plt
from mesa.experimental.continuous_space import ContinuousSpaceAgent, ContinuousSpace
import numpy as np
from mesa.model import Model 


from mesa.visualization import SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle,  PropertyLayerStyle



def plotheatmap(m, name):
    plt.imshow(m, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Kernel Weight')
    plt.title(name)
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
            name="trace_layer",
            resolution=5,
            default_value=0,
            dtype=float
        )
        d=np.random.rand(10, 10)
        self.space.add_property("random_layer", d)

        AgentT.create_agents(model=self, n=n_agents)

    def step(self):
        self.agents.shuffle_do("step")
        pos=self.space.agent_positions
        self.space.trace_layer.deposit_splat(
            pos=pos, 
            value=2,
            mode="add",
            kernel="gaussian",
            spread=0.5
        )
        self.space.trace_layer.decay(
            T=100,
            type="exponential",
            k=1.0
        )

        d=np.random.rand(10, 10)
        self.space.random_layer.data=d

        if "buffer_layer" in self.space._property:
            self.space.buffer_layer.data+=np.indices((40, 40)).sum(axis=0)
            mask=self.space.buffer_layer.get_neighborhood_mask(pos[0], 1.5, True)
            self.space.buffer_layer.data[mask]=0



if __name__ == "__main__":

    d=np.random.rand(10, 10)

    model=ModelT(
        rng=None,
        width=10,
        height=10,
        n_agents=5
    )

    layer2=PropertyLayer(
        name="buffer_layer",
        bounds=(10,10),
        resolution=4,
        default_value=2
    )

    # model.run_for(100)
    # plotheatmap(model.space.trace_layer.data, "trace_layer")
    # model.space._attach_property(layer2)
    # model.run_for(5)
    # plotheatmap(model.space.buffer_layer.data, "buffer_layer")
    # plotheatmap(model.space.random_layer.data, "random_layer")

    # print(model.space.trace_layer.aggregate(operation=np.mean))

    def agent_portrayal(agent):
        return AgentPortrayalStyle(size=10, marker='o', color='red')
        

    def propertylayer_portrayal(layer):
        if layer.name=="trace_layer":
            return PropertyLayerStyle(color='blue', alpha=0.8, colorbar=True)
        if layer.name=="random_layer":
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

    
