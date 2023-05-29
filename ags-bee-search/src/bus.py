from datetime import datetime
import logging
import os
from utils import *
from sygus_parser import StrParser
from itertools import product
import sys
from sygus_string_dsl import *


class ProgramsList():
    def __init__(self):
        self.plist = {}

    def get_programs_all(self, size):
        if size in self.plist:
            programs = []
            for value in self.plist[size].values():
                programs.extend(value)
            return programs

        return []

    def get_programs(self, size, type):
        return self.plist[size][type] if size in self.plist and type in self.plist[size] else []


class Search():

    def __init__(self):
        self.output = set()
        self.evals = 0
        self.plist = ProgramsList()
        self.TEST_OUT_STR = 'out'
        self.str_literals = []
        self.str_var = []
        self.int_literals = []
        self.int_var = []
        self.bool_literals = []

    """
        Returns [is_correct, is_equivalent]
    """

    def eval_and_equivalence_check(self, program, test_cases):
        outputs = self.evaluate(program, test_cases)

        if (outputs == None):
            return False, True

        iscorrect, _ = self.is_correct(outputs, test_cases)

        if (iscorrect):
            return True, False

        outputs_tuple = self.transform_output(outputs)
        outputs_exists = outputs_tuple in self.output

        if (not outputs_exists):
            self.output.add(outputs_tuple)

        return False, outputs_exists

    # Yield all possible combinations of the given list and size
    def findCartesianProduct(self, list, size):
        for combination in product(list, repeat=size):
            yield combination

    def grow(self, nt_operations, test_cases, allowed_size):

        for operation in nt_operations:
            for combination in self.findCartesianProduct(self.plist.plist.keys(), operation.ARITY):
                if (sum(list(combination)) + 1) != allowed_size:
                    continue

                for program in operation.grow(self.plist, combination):
                    is_correct, is_equivalent = self.eval_and_equivalence_check(
                        program, test_cases)

                    if (is_correct):
                        return is_correct, program

                    if (not is_equivalent):
                        if (allowed_size not in self.plist.plist):
                            self.plist.plist[allowed_size] = {}

                        if (program.getReturnType() not in self.plist.plist[allowed_size]):
                            self.plist.plist[allowed_size][program.getReturnType()] = [
                            ]

                        self.plist.plist[allowed_size][program.getReturnType()
                                                       ].append(program)
        return False, None

    def evaluate(self, program, test_cases):
        self.evals += 1
        outputs = []
        for test_case in test_cases:
            try:
                output = program.interpret(test_case)
                outputs.append(output)
            except:
                return None
        return outputs

    def is_correct(self, outputs, test_cases):
        results = [(output == test_cases[index][self.TEST_OUT_STR])
                   for index, output in enumerate(outputs)]
        if all([result == True for result in results]):
            return [True, results]
        # returns [is_correct, is_partially_correct, results]
        return [False, results]

    def transform_output(self, outputs):
        new_outputs = []
        for output in outputs:
            if (type(output) is bool):
                new_outputs.append(str(output))
            else:
                new_outputs.append(output)
        return tuple(new_outputs)

    def transform_terminals(self, terminals, type):
        if (len(terminals) == 0):
            return []
        new_terminals = []

        if (type == 'strvar'):
            for terminal in terminals:
                new_terminals.append(StrVar(terminal))
        elif (type == 'strlit'):
            for terminal in terminals:
                new_terminals.append(StrLiteral(terminal))
        elif (type == 'intvar'):
            for terminal in terminals:
                new_terminals.append(IntVar(terminal))
        elif (type == 'intlit'):
            for terminal in terminals:
                new_terminals.append(IntLiteral(terminal))
        elif (type == 'boollit'):
            for terminal in terminals:
                val = True if terminal == "True" else False
                new_terminals.append(BoolLiteral(val))

        return new_terminals

    def synthesize(self, bound, grammar_nt, str_var, str_literals, int_var, int_literals, test_cases):

        # Create program from for constants and args.
        bool_literals = [str(True), str(False)]
        bool_literals = self.transform_terminals(bool_literals, 'boollit')
        str_literals = self.transform_terminals(str_literals, 'strlit')
        str_var = self.transform_terminals(str_var, 'strvar')
        int_literals = self.transform_terminals(int_literals, 'intlit')
        int_var = self.transform_terminals(int_var, 'intvar')

        terminals = str_literals + str_var + int_literals + int_var + bool_literals

        self.plist.plist[1] = {}
        for terminal in terminals:
            is_correct, is_equivalent = self.eval_and_equivalence_check(
                terminal, test_cases)
            if (is_correct):  # if the terminal is correct, return it
                return terminal, self.evals

            if (not is_equivalent):
                if (terminal.getReturnType() not in self.plist.plist[1]):
                    self.plist.plist[1][terminal.getReturnType()] = []
                self.plist.plist[1][terminal.getReturnType()].append(terminal)

        current_size = 2
        while (current_size <= bound):
            prog_found, prog = self.grow(
                grammar_nt, test_cases, current_size)
            current_size += 1
            if (prog_found):
                return prog, self.evals

        return None, self.evals


if __name__ == "__main__":

    TaskId = None
    log_filename = logs_directory + "/bus.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    """
    Should take three arguments:
    1. TaskId (1-205) - Total number of tasks is 205 in SyGuS - sygus_string_benchmarks.txt
    2. Hard or Easy - 0 for easy, 1 for hard, if not specified, defaults to easy.
    """
    # Assert that the number of arguments is correct.
    assert len(sys.argv) == 2 or len(sys.argv) == 3
    # Assert that the task id is correct.
    assert int(sys.argv[1]) >= 1 and int(sys.argv[1]) <= 205
    # Assert that the difficulty is correct.
    if len(sys.argv) == 3:
        assert int(sys.argv[2]) == 0 or int(sys.argv[2]) == 1

    difficulty = int(sys.argv[2]) if len(sys.argv) == 3 else 0

    slurm_task_id = sys.argv[1]
    TaskId = int(slurm_task_id) - 1
    logging.basicConfig(filename=log_filename,
                        filemode='a',
                        format="[Task: " +
                        str(TaskId) + "] " + '%(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    with open(config_directory+"sygus_string_benchmarks.txt") as f:
        benchmarks = f.read().splitlines()

        string_variables = []
        string_literals = []
        integer_variables = []
        integer_literals = []

    accumulate_all = difficulty == 1  # 0 for easy, 1 for hard

    dsl_functions = [StrConcat, StrReplace, StrSubstr, StrIte, StrIntToStr, StrCharAt, StrLower, StrUpper, IntStrToInt,
                     IntPlus, IntMinus, IntLength, IntIteInt, IntIndexOf, IntFirstIndexOf, IntMultiply, IntModulo,
                     BoolEqual, BoolContain, BoolSuffixof, BoolPrefixof, BoolGreaterThan, BoolLessThan]

    if accumulate_all:
        for index, filename in enumerate(benchmarks):
            specification_parser = StrParser(filename)
            specifications = specification_parser.parse()
            string_variables = list(set(string_variables + specifications[0]))
            string_literals = list(set(string_literals + specifications[1]))
            integer_variables = list(
                set(integer_variables + specifications[2]))
            integer_literals = list(set(integer_literals + specifications[3]))
        lowercase_alphabets = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                                  'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
        uppercase_alphabets = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
        all_alphabets = list(uppercase_alphabets.union(uppercase_alphabets))
        string_literals = list(set(string_literals + all_alphabets))

    # Get the problem.
    benchmark = None
    filename = benchmarks[TaskId]

    benchmark = filename

    specification_parser = StrParser(benchmark)
    specifications = specification_parser.parse()
    logging.info("\n")

    if (not accumulate_all):
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

    # Synthesizer
    synthesizer = Search()

    begin_time = datetime.now()

    # passing bound as 1000
    solution, num = synthesizer.synthesize(1000, dsl_functions, string_variables, string_literals,
                                           integer_variables, integer_literals, input_output_examples)

    time_taken = str(datetime.now() - begin_time)

    if solution is not None:
        logging.info("Benchmark: " + str(benchmark))
        logging.info("Result: Success")
        logging.info("Program: " + solution.toString())
        logging.info("Number of evaluations: " + str(num))
        logging.info(str(datetime.now()))
        logging.info("Time taken: " + str(datetime.now() - begin_time))
    else:
        logging.info("Benchmark: " + str(benchmark))
        logging.info("Result: Fail")
        logging.info("Program: None")
        logging.info("Number of evaluations: " + str(num))
        logging.info(str(datetime.now()))
        logging.info("Time taken: " + str(datetime.now() - begin_time))

    logging.info("\n\n")
