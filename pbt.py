#create unitary q-bit matrix
from scipy.stats import unitary_group
import numpy as np
from qgd_python.decomposition.qgd_N_Qubit_State_Preparation_adaptive import qgd_N_Qubit_State_Preparation_adaptive as N_Qubit_State_Preparation_adaptive
from qgd_python import utils as utils
import time
import sys
from qgd_python.gates.qgd_Gates_Block import qgd_Gates_Block as Gates_Block
from tqdm import tqdm
import numpyro.distributions as dist
import numpyro
from jax import random


class Population_Based_Trainer:
    def __init__(self, State, PBT_config, rng_key, level_limit_max=8, level_limit_min=0, topology=None, config={}):
        self.State = State
        self.PBT_config = PBT_config
        self.level_limit_max = level_limit_max
        self.level_limit_min = level_limit_min
        self.topology = topology 
        self.config = config
        self.agents = [0]*PBT_config["num_of_pops"]
        self.cost_funcs = [0]*PBT_config["num_of_pops"]
        self.rng_key = rng_key

    def create_agent(self, agent_config, parameters):
        
        cDecompose = N_Qubit_State_Preparation_adaptive( self.State, level_limit_max=self.level_limit_max, level_limit_min=self.level_limit_min, topology = self.topology, config=agent_config)
        cDecompose.set_Optimizer("AGENTS")
        cDecompose.set_Gate_Structure_From_Binary(self.PBT_config["circuit"])
        cDecompose.set_Optimized_Parameters(parameters)
        return cDecompose
        
    def check_bounds(self, parameter_idx, parameter):
    
        parameter = min(self.PBT_config["hb"][parameter_idx], parameter)
        parameter = max(self.PBT_config["lb"][parameter_idx], parameter)
        return parameter
    
    def explore(self, config):
        
        config_new = config.copy()
        for idx in range(len(self.PBT_config["parameters"])):
            parameter = self.PBT_config["parameters"][idx]
            theta = config_new[parameter]
            theta += float(self.PBT_config["perturbation_radius"][idx].sample(random.PRNGKey(np.random.randint(10000))))
            if self.PBT_config["type"][idx] == "discrete":
                theta = int(theta)
            else:
                theta = float(theta)
            theta = self.check_bounds(idx,theta)
            config_new[parameter] = theta
        return config_new
    
    def exploit(self, best_agent_idx):
        
        config_best = self.agents[best_agent_idx].config
        config_new = self.explore(config_best)
        return config_best
    
    def synchronize(self):
        
        for idx in range(self.PBT_config["num_of_pops"]):
            agent = self.agents[idx]
            self.cost_funcs = agent.Optimization_Problem(agent.get_Optimized_Parameters())
        best_agent_idx = np.argmin(self.cost_funcs)
        parameters = self.agents[best_agent_idx].get_Optimized_Parameters()
        best_agent_config = self.agents[best_agent_idx].config
        
        for agent_idx in range(self.PBT_config["num_of_pops"]):
            if agent_idx != best_agent_idx:
                config_new = self.exploit(best_agent_idx)
            else: 
                config_new = best_agent_config
            config_new["max_inner_iterations"] = self.PBT_config["episode_iter"] - (self.PBT_config["episode_iter"]%config_new["agent_lifetime"]) + 1
            self.agents[idx] = self.create_agent(config_new, parameters)
            self.agents[idx].set_Project_Name(str(agent_idx))
        return 
    
    def tune_loop(self):
        
        for agent_idx in range(self.PBT_config["num_of_pops"]):
            self.agents[agent_idx].get_Initial_Circuit()
        self.synchronize()
        return 
    
    def initialize(self):
        
        config_new = self.config.copy()
        for idx in range(len(self.PBT_config["parameters"])):
            parameter = self.PBT_config["parameters"][idx]
            theta = self.PBT_config["parameter_distributions"][idx].sample(random.PRNGKey(np.random.randint(10000)))
            if self.PBT_config["type"][idx] == "discrete":
                theta = int(theta)
            else:
                theta = float(theta)
            config_new[parameter] = theta
        config_new["max_inner_iterations"] = self.PBT_config["episode_iter"] - (self.PBT_config["episode_iter"]%config_new["agent_lifetime"]) + 1
        return config_new
    
    def tune(self):
        
        for agent_idx in range(self.PBT_config["num_of_pops"]):
            config_agent = self.initialize()
            self.agents[agent_idx] = self.create_agent(config_agent,np.zeros(self.PBT_config["parameter_num"]))
            
        for agent_idx in range(self.PBT_config["num_of_pops"]):
            self.agents[agent_idx].get_Initial_Circuit()
        self.synchronize()
        for episode in tqdm(range(self.PBT_config["episode_num"]-1)):
            self.tune_loop()
            print(f"BEST AGENT NUM:{np.argmin(self.cost_funcs)} , DECOMPOSITION ERROR:{np.min(self.cost_funcs)}")
        return self.agents[np.argmin(self.cost_funcs)].config

            
