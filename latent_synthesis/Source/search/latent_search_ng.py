from __future__ import annotations
from functools import partial
from multiprocessing import Pool
import os
import time
import numpy as np
import torch
import nevergrad as ng

from dsl import DSL
from dsl.base import Program
from search.top_down import TopDownSearch
from vae.models.base_vae import BaseVAE
from logger.stdout_logger import StdoutLogger
from tasks.task import Task
from config import Config
from vae.models.sketch_vae import SketchVAE


def execute_program(program_str: str, task_envs: list[Task],
                    dsl: DSL) -> tuple[Program, int, float]:
    try:
        program = dsl.parse_str_to_node(program_str)
    except AssertionError: # In case of invalid program (e.g. does not have an ending token)
        return Program(), 0, -float('inf')
    # If program is a sketch
    if not program.is_complete():
        # Let TDS complete and evaluate programs
        tds = TopDownSearch()
        tds_result = tds.synthesize(program, dsl, task_envs, Config.datagen_sketch_iterations)
        program, num_evaluations, mean_reward = tds_result
        if program is None: # Failsafe for TDS result
            return program, num_evaluations, -float('inf')
        if not program.is_complete(): # TDS failed to complete program
            return program, num_evaluations, -float('inf')
    # If program is a complete program
    else:
        # Evaluate single program
        mean_reward = 0.
        for task_env in task_envs:
            mean_reward += task_env.evaluate_program(program)
        num_evaluations = 1
        mean_reward /= len(task_envs)
    return program, num_evaluations, mean_reward


class LatentSearchNG:
    """Implements the CEM method from LEAPS paper.
    """
    def __init__(self, model: BaseVAE, task_cls: type[Task], dsl: DSL):
        self.model = model
        if issubclass(type(self.model), SketchVAE):
            self.dsl = dsl.extend_dsl()
        else:
            self.dsl = dsl
        self.device = self.model.device
        self.population_size = Config.search_population_size
        self.elitism_rate = Config.search_elitism_rate
        self.n_elite = int(Config.search_elitism_rate * self.population_size)
        self.number_executions = Config.search_number_executions
        self.number_iterations = Config.search_number_iterations
        self.sigma = Config.search_sigma
        self.model_hidden_size = Config.model_hidden_size
        self.task_envs = [task_cls(i) for i in range(self.number_executions)]
        self.program_filler = TopDownSearch()
        self.filler_iterations = Config.search_topdown_iterations
        output_dir = os.path.join('output', Config.experiment_name, 'latent_search')
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, f'seed_{Config.model_seed}.csv')
        self.trace_file = os.path.join(output_dir, f'seed_{Config.model_seed}.gif')
        self.restart_timeout = Config.search_restart_timeout


    def execute_single_latent(self, latent_program: np.ndarray) -> float:
        """Evaluates the given latent program in the environment.

        Args:
            latent_program (torch.Tensor): Latent program as a tensor.

        Returns:
            float: Mean reward of the program.
        """
        latent_program_torch = torch.tensor(latent_program, dtype=torch.float32, device=self.device)
        program_list = self.model.decode_vector(latent_program_torch.unsqueeze(0))[0]
        try:
            program_nodes = self.dsl.parse_int_to_node(program_list)
        except AssertionError:
            return float('inf')
        sum_rewards = 0.
        for task_env in self.task_envs:
            sum_rewards += task_env.evaluate_program(program_nodes)
        return - sum_rewards / len(self.task_envs)

    
    def search(self) -> tuple[str, bool, int]:
        """Main search method. Searches for a program using the specified DSL that yields the
        highest reward at the specified task.

        Returns:
            tuple[str, bool]: Best program in string format and a boolean value indicating
            if the search has converged.
        """
        parameterization = ng.p.Array(shape=(self.model_hidden_size,), lower=-1., upper=1.)
        optimizer = ng.optimizers.PSO(parametrization=parameterization, budget=100000)
        recommendation = optimizer.minimize(self.execute_single_latent, verbosity=1)
        print(recommendation)
        # with open(self.output_file, mode='w') as f:
        #     f.write('time,num_evaluations,best_reward,best_program\n')

        # for iteration in range(1, self.number_iterations + 1):
        #     programs, num_eval, rewards = self.execute_population(population)
        #     best_indices = torch.topk(rewards, self.n_elite).indices
        #     elite_population = population[best_indices]
        #     mean_elite_reward = torch.mean(rewards[best_indices])
        #     num_evaluations += num_eval

        #     if torch.max(rewards) > best_reward:
        #         best_reward = torch.max(rewards)
        #         best_program = programs[torch.argmax(rewards)]
        #         StdoutLogger.log('Latent Search',f'New best reward: {best_reward}')
        #         StdoutLogger.log('Latent Search',f'New best program: {best_program}')
        #         with open(self.output_file, mode='a') as f:
        #             t = time.time() - start_time
        #             f.write(f'{t},{num_evaluations},{best_reward},{best_program}\n')

        #     StdoutLogger.log('Latent Search',f'Iteration {iteration}.')
        #     StdoutLogger.log('Latent Search',f'Mean elite reward: {mean_elite_reward}')
        #     StdoutLogger.log('Latent Search',f'Number of evaluations in this iteration: {num_eval}')
            
        #     if best_reward >= 1.0:
        #         converged = True
        #         break
        #     if mean_elite_reward.cpu().numpy() == prev_mean_elite_reward:
        #         counter_for_restart += 1
        #     else:
        #         counter_for_restart = 0
        #     if counter_for_restart >= self.restart_timeout and self.restart_timeout > 0:
        #         population = self.init_population()
        #         counter_for_restart = 0
        #         StdoutLogger.log('Latent Search','Restarted population.')
        #     else:
        #         new_indices = torch.ones(elite_population.size(0), device=self.device).multinomial(
        #             self.population_size, replacement=True)
        #         if Config.search_reduce_to_mean:
        #             elite_population = torch.mean(elite_population, dim=0).repeat(self.n_elite, 1)
        #         new_population = []
        #         for index in new_indices:
        #             sample = elite_population[index]
        #             new_population.append(
        #                 sample + self.sigma * torch.randn_like(sample, device=self.device)
        #             )
        #         population = torch.stack(new_population)
        #     prev_mean_elite_reward = mean_elite_reward.cpu().numpy()
        
        # best_program_nodes = self.dsl.parse_str_to_node(best_program)
        # self.task_envs[0].trace_program(best_program_nodes, self.trace_file, 1000)
        
        return None, False, 0
