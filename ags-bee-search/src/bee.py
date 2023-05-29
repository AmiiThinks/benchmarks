import logging
import os
import sys
from datetime import datetime, timedelta
import heapq
import bisect
import math

import numpy as np
import tensorflow.keras.models as keras_model
from scipy.interpolate import CubicSpline

from sygus_string_dsl import *
from sygus_parser import StrParser
from utils import *

import argparse
from Levenshtein import distance


limit_decimal_places = True


class ProgramList:
    def __init__(self, string_variables_list, integer_variables_list, input_output):
        """
        self.plist has a structure like this:
        {"cost":
            {"type1": [] programs,
             "type2": [] programs,
             ...},
         "cost2":
            {"type1": [] programs,
            "type2": [] programs,
            ...},
            ...
        }
        """
        self.number_reheapify_calls = 0
        self.plist = {}
        self.number_programs = 0
        self.parent_input_output = input_output
        self.string_variables = string_variables_list
        self.integer_variables = integer_variables_list
        self.parent_ps = []
        self.batch_jobs = []
        self.property_encodings = {
            AllTrue: EncodedAllTrue,
            AllFalse: EncodedAllFalse,
            Mixed: EncodedMixed,
            Padding: EncodedPadding,
        }
        """
         self.pq_store has a structure like this:
        {
            'arity': [
                [cost, order, combination],...
                ],
                ...
            ],
        }
        Arity = No. of inputs to any operation. 
        We combine them based on arity. This reduces the number of operations we need to consider.
        Order is just maintained to break the tie when cost is same.
        Combination is the actual combination of operations based on indices of the costs from costs_list.
        """
        self.pq_store = {}

        """
        Maintains popped elements from pq_store, indexed by arity. So that we can perform the grow step.
        """
        self.pq_popped = {}
        self.cost_list = []
        self.order = 0

        """
        Maintains costs of programs generated in the last iteration.
        """
        self.last_costs_generated = []

        """
        pq_last_max maintains the max cost of tuples generated for each arity in last iteration.
        Used for heapify operation.
        Next three variables are used as keys for the pq_last_max dictionary. 
        COST: max cost of tuples generated for each arity in last iteration.
        HEAPIFY: (bool) whether heapify operation is needed for the arity.
        MAX_INDEX: max index in any tuple generated in the last iteration.
        """
        self.pq_last_max = {}
        self.COST = "cost"
        self.HEAPIFY = "heapify"
        self.MAX_INDEX = "max_index"

        calculate_ps_for_problem(self, string_variables_list, integer_variables_list)

    def insert(self, program):
        self.batch_jobs.append(program)

    def process_batch_jobs(self):
        batch_size = 100000
        total_jobs = len(self.batch_jobs)

        # Used to store the costs of the programs for the current batch - needed to heapify.
        self.last_costs_generated = []

        # step by batch size
        for job_index in range(0, total_jobs, batch_size):
            current_batch = self.batch_jobs[job_index : job_index + batch_size]
            current_batch_ps = []  # property signatures of the current batch

            for program in current_batch:
                # Copy the parent property signature.
                test_row = self.parent_ps.copy()
                child_input_outputs = []

                # Run the program on all the input-output pairs.
                for index, parent_input in enumerate(self.parent_input_output):
                    child_input_output = parent_input.copy()
                    child_output = program.interpret(child_input_output)
                    child_input_output["cout"] = child_output
                    child_input_output["out"] = self.parent_input_output[index]["out"]
                    child_input_outputs.append(child_input_output)

                # get the output of the subexpression
                outputs = [output["cout"] for output in child_input_outputs]

                # get ps for the sub-program.
                populate_sub_program_ps(
                    self,
                    program,
                    test_row,
                    outputs,
                    child_input_outputs,
                    STR_TYPES,
                    INT_TYPES,
                    BOOL_TYPES,
                )

                # Append the test row to the current batch.
                current_batch_ps.append(test_row)

            # Predict the probability of the current batch.
            current_batch_predictions = BustleModel.predict(np.array(current_batch_ps))

            for program_index, program in enumerate(current_batch):
                # Predicted probability of the program by the model.
                program_probability = current_batch_predictions[program_index]
                # Penalization based on W(unbound)
                additional_weight = -math.log(program_probability[0], 2)
                if not additional_weight > 0:
                    # Safe check, not happens in practice.
                    additional_weight = 0.01
                # Updated size of the program while avoiding floating point errors.
                # New size = old size + W(unbound)
                program.size = decimal_place_converter(program.size + additional_weight)
                # program.size = program.size + additional_weight

                # Storing the program.
                if program.size not in self.plist:
                    self.plist[program.size] = {}

                if program.size not in self.cost_list:
                    # bisect.insrot is essentially a binary search - inserts el in a sorted list.
                    bisect.insort(self.last_costs_generated, program.size)
                    bisect.insort(self.cost_list, program.size)

                if program.getReturnType() not in self.plist[program.size]:
                    self.plist[program.size][program.getReturnType()] = []

                self.plist[program.size][program.getReturnType()].append(program)
                self.number_programs += 1

        # Code for heapifying the PQs.
        needToHeapify = False
        if len(self.last_costs_generated) > 0:
            for arity in range(1, MAX_ARITY + 1):
                # If the last cost generated is less than the max cost in the PQ, then we need to heapify the PQ.
                if self.last_costs_generated[0] < self.pq_last_max[arity][self.COST]:
                    self.pq_last_max[arity][self.HEAPIFY] = True
                    needToHeapify = True
        if needToHeapify:
            self.number_reheapify_calls += 1
            self.heapify_pqs()

        # Clearing the batch jobs.
        self.batch_jobs.clear()
        return self.number_reheapify_calls

    def heapify_pqs(self):
        for arity in range(1, MAX_ARITY + 1):
            if self.pq_last_max[arity][self.HEAPIFY]:
                self.pq_last_max[arity][self.HEAPIFY] = False
                pq_for_current_arity = self.pq_store[arity]
                # pq_for_current_arity = copy.deepcopy(pq_for_current_arity)
                pq_for_current_arity = [
                    [
                        decimal_place_converter(
                            sum(self.index_to_cost_comb(comb[2])) + 1
                        ),
                        comb[1],
                        comb[2],
                    ]
                    for comb in pq_for_current_arity
                ]
                heapq.heapify(pq_for_current_arity)
                self.pq_store[arity] = pq_for_current_arity

    # Initialize the program list with the literals and variables.
    def init_insert(self, program):
        if program.size not in self.plist:
            self.plist[program.size] = {}

        if program.getReturnType() not in self.plist[program.size]:
            self.plist[program.size][program.getReturnType()] = []

        self.plist[program.size][program.getReturnType()].append(program)
        self.number_programs += 1

    def init_plist(
        self,
        string_literals_list,
        integer_literals_list,
        boolean_literals,
        string_variables_list,
        integer_variables_list,
        partial_programs=[],
    ):
        for string_literal in string_literals_list:
            init_program = StrLiteral(string_literal)
            self.init_insert(init_program)

        for integer_literal in integer_literals_list:
            init_program = IntLiteral(integer_literal)
            self.init_insert(init_program)

        for boolean_literal in boolean_literals:
            init_program = BoolLiteral(boolean_literal)
            self.init_insert(init_program)

        for str_var in string_variables_list:
            init_program = StrVar(str_var)
            self.init_insert(init_program)

        for int_var in integer_variables_list:
            init_program = IntVar(int_var)
            self.init_insert(init_program)

        if len(partial_programs) > 0:
            for pprog in partial_programs:
                pprog.size = 1
                self.init_insert(pprog)

        self.cost_list.append(1)
        self.initialize_pq_store()

    def get_programs_all(self, size):
        if size in self.plist:
            programs = []
            for value in self.plist[size].values():
                programs.extend(value)
            return programs

        return []

    def get_programs(self, size, return_type):
        if size in self.plist:
            if return_type in self.plist[size]:
                return self.plist[size][return_type]

        return []

    def get_number_programs(self):
        return self.number_programs

    """
    Returns combination of costs for a given combination of indices.
    @param combination: Combination of indices.
    @return: Combination of costs.
    e.g., converts (0,1) => (2, 4) if cost_list = [2, 4, 6, 8]
    """

    def index_to_cost_comb(self, combination):
        return [self.cost_list[val] for val in combination]

    """
    Initializes the priority queues.
    - For each arity, we have a priority queue to keep things simple.
    - Initially it is (0,...,k) where k is the arity.
    """

    def initialize_pq_store(self):
        for arity in range(1, MAX_ARITY + 1):
            self.pq_popped[arity] = []
            combination = [0] * arity
            cost = (arity * self.cost_list[0]) + 1  # Initial cost.
            combinationx = [cost, self.order, combination]
            self.order += 1  # Tie breaker.
            self.pq_store[arity] = [combinationx]
            self.pq_last_max[arity] = {
                self.COST: 0,
                self.HEAPIFY: False,
                self.MAX_INDEX: 0,
            }

    """
    Returns a list of the next cheapest combination of programs. Each item in array is of the form: [cost, arity]
    """

    def get_next_cheapest(self):
        comb_container = []
        smallest_cost = float("inf")

        for arity in range(1, MAX_ARITY + 1):
            item = self.pq_store[arity]
            if len(item) == 0:
                continue

            current_cost = item[0][0]

            if current_cost <= smallest_cost:
                if current_cost < smallest_cost:
                    comb_container = []
                smallest_cost = current_cost
                comb_container.append(arity)

        return [comb_container, smallest_cost]

    def generate_next_set_of_combinations(self):
        for arity in range(1, MAX_ARITY + 1):
            # If no element was popped from this arity.
            if len(self.pq_popped[arity]) == 0:
                continue

            pq_for_current_arity = self.pq_store[arity]
            popped_combinations = self.pq_popped[arity]
            max_index = len(self.cost_list) - 1  # Max index allowed.

            # Maximum index that has been used in the pq of this arity.
            maximum_index_used = self.pq_last_max[arity][self.MAX_INDEX]

            """
            Basic idea:
            - Add one to each index of the popped combination separately.
            e.g., (0,0) --> (1,0) and (0,1)

            We are using a small trick to not maintain an open and closed list for tuples and each tuple is only generated once.
            We are using the fact that the tuples are generated in increasing order. e.g., (1,0) and (0,1) can both generate (1,1).
            We want to avoid generating (1,1) twice. We do this by checking if the index that we have incremented is greater than 1, we stop.
            
            e.g.,
            (0,0) --> (1,0), (0,1)
            (1,0) --> (2,0) 
            (0,1) --> (1,1), (0,2)

            similarly,
            (0,0,0) --> (1,0,0), (0,1,0), (0,0,1)
            (1,0,0) --> (2,0,0)
            (0,1,0) --> (1,1,0), (0,2,0)
            (0,0,1) --> (1,0,1), (0,1,1), (0,0,2)
            """

            for comb in popped_combinations:
                for i in range(arity):
                    cc = comb.copy()

                    # Does not happen in practice. Sanity check.
                    if cc[i] == max_index:
                        break

                    # Incrementing the tuple index at ith position.
                    cc[i] = cc[i] + 1

                    # Updating the maximum_index_used in this arity.
                    if cc[i] > maximum_index_used:
                        maximum_index_used = cc[i]

                    # Calculating the cost of the new combination.
                    cost = decimal_place_converter(sum(self.index_to_cost_comb(cc)) + 1)

                    # Inserting the new combination into the priority queue.
                    heapq.heappush(pq_for_current_arity, [cost, self.order, cc])
                    self.order += 1

                    # If the value at ith position is greater than 1, we do not need to expand it further.
                    if cc[i] > 1:
                        break

            self.pq_last_max[arity][self.MAX_INDEX] = maximum_index_used
            self.pq_last_max[arity][self.COST] = self.cost_list[maximum_index_used]
            self.pq_popped[arity] = []  # Emptying it for next iteration


class BeeSearch:
    def __init__(self, string_variables_list, integer_variables_list, input_output):
        self._variables = string_variables_list + integer_variables_list
        self._input_output = input_output
        self.plist = ProgramList(
            string_variables_list, integer_variables_list, input_output
        )
        self._outputs = set()
        self.number_evaluations = 0

    def is_correct(self, p):
        is_program_correct = True

        for inout in self._input_output:
            env = self.init_env(inout)
            out = p.interpret(env)
            if out != inout["out"]:
                is_program_correct = False

        return is_program_correct

    # addapted from @thirupathi-reddy's code on github
    def is_partial_solution(self, p):
        is_partial_program = False
        average_scores = float("inf")

        for inout in self._input_output:
            env = self.init_env(inout)
            out = p.interpret(env)
            if out == inout["out"]:
                is_partial_program = True

        if is_partial_program:
            eval_scores = []
            for inout in self._input_output:
                env = self.init_env(inout)
                out = p.interpret(env)

                if not type(out) == str:
                    out = str(out)
                if not type(inout["out"]) == str:
                    inout["out"] = str(inout["out"])

                eval_scores.append(distance(out, inout["out"]))
            average_scores = sum(eval_scores) / len(eval_scores)

        return is_partial_program, average_scores

    def init_env(self, inout):
        env = {}
        for v in self._variables:
            env[v] = inout[v]
        return env

    def has_equivalent(self, program):
        p_out = []
        for inout in self._input_output:
            env = self.init_env(inout)
            out = program.interpret(env)
            if out is not None:
                p_out.append(out)
            else:
                return True

        tuple_out = tuple(p_out)

        if tuple_out not in self._outputs:
            self._outputs.add(tuple_out)
            return False
        return True

    def grow(self, cheapest_combinations, next_cheapest_cost):
        new_programs = []
        for cheapest_combination in cheapest_combinations:
            # Pick the cheapest entry.
            smallest_cost_arity = cheapest_combination

            # Get all tthe combinations against that op
            all_comb_of_arity = self.plist.pq_store[smallest_cost_arity]

            # Remove all the cheapest combinations from the list.
            while (
                len(all_comb_of_arity) and all_comb_of_arity[0][0] == next_cheapest_cost
            ):
                smallest_combination = heapq.heappop(all_comb_of_arity)

                # Add that to the popped combinations.
                self.plist.pq_popped[smallest_cost_arity].append(
                    smallest_combination[2]
                )

                # Since programs are indexed by cost in the plist.
                cost_combination = self.plist.index_to_cost_comb(
                    smallest_combination[2]
                )

                for operation in NON_TERMINALS:
                    if operation.ARITY != smallest_cost_arity:
                        continue
                    for new_program in operation.grow(self.plist, cost_combination):
                        self.number_evaluations += 1
                        if not self.has_equivalent(new_program):
                            new_programs.append(new_program)
                            yield new_program

        # For evaluating it later on w neural network.
        for new_program in new_programs:
            self.plist.insert(new_program)

        self.number_heapify_calls += self.plist.process_batch_jobs()

    def search(
        self,
        bound,
        string_literals_list,
        integer_literals_list,
        boolean_literals,
        string_variables_list,
        integer_variables_list,
    ):
        self.plist.init_plist(
            string_literals_list,
            integer_literals_list,
            boolean_literals,
            string_variables_list,
            integer_variables_list,
        )

        current_size = 0
        prev_cheapest = 0
        self.number_heapify_calls = 0

        # More of a check on # of iterations rather than a size bound.
        while current_size <= bound:
            cheapest_combinations, next_cheapest_cost = self.plist.get_next_cheapest()
            # Sanity checks to make sure that the implementation is correct.
            if math.isinf(next_cheapest_cost):
                logging.info("Error in implementation.")
                break

            if next_cheapest_cost < prev_cheapest:
                logging.info("Error in implementation.")
                break

            for new_program in self.grow(cheapest_combinations, next_cheapest_cost):
                is_p_correct = self.is_correct(new_program)
                if is_p_correct:
                    return (
                        new_program,
                        self.number_evaluations,
                        self.number_heapify_calls,
                    )

            self.plist.generate_next_set_of_combinations()

            prev_cheapest = next_cheapest_cost

            current_size += 1

        return None, self.number_evaluations, self.number_heapify_calls

    def augmented_search(
        self,
        program_limit,
        string_literals_list,
        integer_literals_list,
        boolean_literals,
        string_variables_list,
        integer_variables_list,
        current_iteration=1,
        partial_programs=[],
    ):
        self.plist.init_plist(
            string_literals_list,
            integer_literals_list,
            boolean_literals,
            string_variables_list,
            integer_variables_list,
            partial_programs,
        )

        current_size = 0
        prev_cheapest = 0
        self.number_heapify_calls = 0  # this is for one search
        number_evaluations_Onesearch = 0

        # list of tuples of the form (score, program)
        augmented_programs = []

        # More of a check on # of iterations rather than a size bound.
        while number_evaluations_Onesearch <= program_limit:
            cheapest_combinations, next_cheapest_cost = self.plist.get_next_cheapest()
            # Sanity checks to make sure that the implementation is correct.
            if math.isinf(next_cheapest_cost):
                logging.info("Error in implementation.")
                break

            if next_cheapest_cost < prev_cheapest:
                logging.info("Error in implementation.")
                break

            for new_program in self.grow(cheapest_combinations, next_cheapest_cost):
                is_p_correct = self.is_correct(new_program)
                number_evaluations_Onesearch += 1
                if is_p_correct:
                    return (
                        [new_program],
                        True,  # True if augmented search found a solution.
                        self.number_evaluations,
                        self.number_heapify_calls,
                    )
                # partial program, add to the partial program list with program
                else:
                    is_p_partial_program, score = self.is_partial_solution(new_program)
                    if is_p_partial_program:
                        augmented_programs.append((score, new_program))

                # if number of programs evaluated is greater than the program limit, break.
                if number_evaluations_Onesearch > program_limit:
                    break

            self.plist.generate_next_set_of_combinations()

            prev_cheapest = next_cheapest_cost

            current_size += 1

        # sort the augmented programs by score in ascending order and return the top current_iteration programs.
        augmented_programs.sort(key=lambda x: x[0])
        return_programs = [
            program[1] for program in augmented_programs[:current_iteration]
        ]

        logging.info(
            "Augmented search did not find a solution. Best programs found: "
            + return_programs[0].toString()
            + " with score: "
            + str(augmented_programs[0][0])
        )

        return (
            return_programs,
            False,
            self.number_evaluations,
            self.number_heapify_calls,
        )

    def synthesize(
        self,
        bound,
        operations,
        string_literals_list,
        integer_literals_list,
        boolean_literals,
        string_variables_list,
        integer_variables_list,
    ):
        BustlePCFG.initialize(
            operations,
            string_literals_list,
            integer_literals_list,
            boolean_literals,
            string_variables_list,
            integer_variables_list,
        )

        program_solution, evaluations, reheapifies = self.search(
            bound,
            string_literals_list,
            integer_literals_list,
            boolean_literals,
            string_variables_list,
            integer_variables_list,
        )

        return program_solution, evaluations, reheapifies

    def augmented_synthesize(
        self,
        program_limit,
        operations,
        string_literals_list,
        integer_literals_list,
        boolean_literals,
        string_variables_list,
        integer_variables_list,
        current_iteration=1,
        partial_programs=[],
    ):
        BustlePCFG.initialize(
            operations,
            string_literals_list,
            integer_literals_list,
            boolean_literals,
            string_variables_list,
            integer_variables_list,
        )

        return_program_list = []
        (
            return_program_list,
            is_solved,
            evaluations,
            reheapifies,
        ) = self.augmented_search(
            program_limit,
            string_literals_list,
            integer_literals_list,
            boolean_literals,
            string_variables_list,
            integer_variables_list,
            current_iteration,
            partial_programs,
        )

        if is_solved:
            return is_solved, return_program_list[0], evaluations, reheapifies
        else:
            return is_solved, return_program_list, evaluations, reheapifies


def load_bustle_model(models_filename="bustle_model_01.hdf5"):
    global BustleModel
    # can be changed to load different models
    model_filename = models_directory + models_filename
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    BustleModel = keras_model.load_model(model_filename)


if __name__ == "__main__":
    """
    Should take three arguments:
    1. TaskId (1-89) - Total number of tasks is 89 in bustle_benchmarks.txt
    2. Hard or Easy - 0 for easy, 1 for hard, if not specified, defaults to easy.
    3. Log filename - if not specified, defaults to bee-search.log
    4. Model filename - if not specified, defaults to bustle_model_01.hdf5
    5. Benchmarks filename - if not specified, defaults to bustle_benchmarks.txt
    6. Augmented - 0 for non-augmented, 1 for augmented. If not specified, defaults to non-augmented.
    7. Program Limit - Number of programs evaluated before augmenting. If not specified, defaults to 14 million.
    8. Time Limit - In floating point hours. Time limit for the search. If not specified, defaults to 1 hour
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--task_id",
        dest="TaskId",
        default=57,
        type=int,
        help="Define the task id from bustle_benchmarks.txt to solve.",
    )

    parser.add_argument(
        "-d",
        "--difficulty",
        dest="difficulty",
        default="0",
        type=int,
        help="Hard or Easy - 0 for easy, 1 for hard, if not specified, defaults to easy.",
    )

    parser.add_argument(
        "-l",
        "--log",
        dest="log_filename",
        default="bee-search.log",
        type=str,
        help="Log filename - if not specified, defaults to bee-search.log",
    )

    parser.add_argument(
        "-m",
        "--model",
        dest="model_filename",
        default="bustle_model_01.hdf5",
        type=str,
        help="Model filename - if not specified, defaults to bustle_model_01.hdf5",
    )

    parser.add_argument(
        "-b",
        "--benchmark",
        dest="benchmark_filename",
        default="bustle_benchmarks.txt",
        type=str,
        help="Benchmarks filename - if not specified, defaults to bustle_benchmarks.txt",
    )

    parser.add_argument(
        "-a",
        "--is_augmenting",
        dest="is_augmenting",
        default=0,
        type=int,
        help="Augmented - 0 for non-augmented, 1 for augmented. If not specified, defaults to non-augmented.",
    )

    parser.add_argument(
        "-p",
        "--program_limit",
        dest="program_limit",
        default=14000000,
        type=int,
        help="Program Limit - Number of programs evaluated before augmenting. If not specified, defaults to 14 million.",
    )

    # parser.add_argument(
    #     "-tl",
    #     "--time_limit",
    #     dest="time_limit",
    #     default=1,
    #     type=float,
    #     help="Time Limit - In floating point hours. Time limit for the search. If not specified, defaults to 1 hour",
    # )

    parameters = parser.parse_args()
    TaskId = parameters.TaskId - 1
    log_filename = logs_directory + parameters.log_filename
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    difficulty = parameters.difficulty
    model_filename = parameters.model_filename
    benchmark_filename = parameters.benchmark_filename
    program_limit = parameters.program_limit
    is_augmenting = parameters.is_augmenting

    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        format="[Task: " + str(TaskId) + "] " + "%(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    load_bustle_model(model_filename)

    with open(config_directory + parameters.benchmark_filename) as f:
        benchmarks = f.read().splitlines()

        string_variables = []
        string_literals = []
        integer_variables = []
        integer_literals = []

    accumulate_all = difficulty == 1  # 0 for easy, 1 for hard

    if accumulate_all:
        for index, filename in enumerate(benchmarks):
            specification_parser = StrParser(filename)
            specifications = specification_parser.parse()
            string_variables = list(set(string_variables + specifications[0]))
            string_literals = list(set(string_literals + specifications[1]))
            integer_variables = list(set(integer_variables + specifications[2]))
            integer_literals = list(set(integer_literals + specifications[3]))
        lowercase_alphabets = set(
            [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
            ]
        )
        uppercase_alphabets = set(
            [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
            ]
        )
        all_alphabets = list(uppercase_alphabets.union(uppercase_alphabets))
        string_literals = list(set(string_literals + all_alphabets))

    benchmark = None
    filename = benchmarks[TaskId]

    benchmark = filename

    specification_parser = StrParser(benchmark)
    specifications = specification_parser.parse()
    logging.info("\n")
    # Sygus grammar.
    dsl_functions = [
        StrConcat,
        StrReplace,
        StrSubstr,
        StrIte,
        StrIntToStr,
        StrCharAt,
        StrLower,
        StrUpper,
        IntStrToInt,
        IntPlus,
        IntMinus,
        IntLength,
        IntIteInt,
        IntIndexOf,
        IntFirstIndexOf,
        IntMultiply,
        IntModulo,
        BoolEqual,
        BoolContain,
        BoolSuffixof,
        BoolPrefixof,
        BoolGreaterThan,
        BoolLessThan,
    ]

    if not accumulate_all:
        string_variables = specifications[0]
        string_literals = specifications[1]
        integer_variables = specifications[2]
        integer_literals = specifications[3]
    else:
        string_variables = specifications[0]
        integer_variables = specifications[2]
        string_literals = list(set(specifications[1] + string_literals))
        integer_literals = list(set(specifications[3] + integer_literals))

    input_output_examples = specifications[4]

    begin_time = datetime.now()

    solution, num, reheapifies = None, 0, None

    if is_augmenting == 1:
        current_iteration = 1
        partial_programs = []
        is_solved = False

        while not is_solved and current_iteration <= 10:
            synthesizer = BeeSearch(
                string_variables, integer_variables, input_output_examples
            )

            synthesizer.number_evaluations = num

            (
                is_solved,
                solution,
                num_eval,
                reheapifies,
            ) = synthesizer.augmented_synthesize(
                program_limit,
                dsl_functions,
                string_literals,
                integer_literals,
                [True, False],
                string_variables,
                integer_variables,
                current_iteration,
                partial_programs,
            )
            num += num_eval
            current_iteration += 1
            if not is_solved:
                partial_programs.extend(solution)
        if not is_solved:
            solution = None

    else:
        synthesizer = BeeSearch(
            string_variables, integer_variables, input_output_examples
        )

        solution, num, reheapifies = synthesizer.synthesize(
            float("inf"),
            dsl_functions,
            string_literals,
            integer_literals,
            [True, False],
            string_variables,
            integer_variables,
        )

    if solution is not None:
        logging.info("Benchmark: " + str(benchmark))
        logging.info("Model: " + str(model_filename))
        logging.info("Difficulty: " + "Easy" if difficulty == 0 else "Hard")
        logging.info("Result: Success")
        logging.info("Augmented: " + str(is_augmenting == 1))
        logging.info("Program: " + solution.toString())
        logging.info("Number of evaluations: " + str(num))
        logging.info(str(datetime.now()))
        logging.info("Time taken: " + str(datetime.now() - begin_time))
        logging.info("Number of calls to heapify: " + str(reheapifies))
        # create a file named taskid_model_augmented/not_augmented.txt inside results folder
        with open(
            "../results/"
            + str(TaskId + 1)
            + "_"
            + str(model_filename)
            + "_"
            + str(is_augmenting == 1)
            + ".txt",
            "w",
        ) as f:
            f.write("Program: " + solution.toString() + "\n")
            f.write("Number of evaluations: " + str(num) + "\n")
            f.close()

    else:
        logging.info("Benchmark: " + str(benchmark))
        logging.info("Model: " + str(model_filename))
        logging.info("Difficulty: " + "Easy" if difficulty == 0 else "Hard")
        logging.info("Result: Fail")
        logging.info("Augmented: " + str(is_augmenting == 1))
        logging.info("Program: None")
        logging.info("Number of evaluations: " + str(num))
        logging.info(str(datetime.now()))
        logging.info("Time taken: " + str(datetime.now() - begin_time))
        logging.info("Number of calls to heapify: " + str(reheapifies))

    logging.info("\n\n")
