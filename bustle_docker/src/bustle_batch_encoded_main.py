import logging
import os
import sys
from datetime import datetime
import time

import numpy as np
import tensorflow.keras.models as keras_model

from bustle_properties import *
from bustle_string_dsl import *
from sygus_parser import StrParser
from utils import *


def populate_property_value(property_signature, property_encoding):
    for encoding in property_encoding:
        property_signature.append(encoding)


class ProgramList:

    def __init__(self, string_variables_list, integer_variables_list, input_output):
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
            Padding: EncodedPadding
        }

        max_string_variables = 3

        # input strings with string only properties
        for string_variable in string_variables_list:
            input_strings = [parent_input.get(string_variable) for parent_input in self.parent_input_output]
            for StringProperty in StringProperties:
                property_value = StringProperty(input_strings)
                populate_property_value(self.parent_ps, self.property_encodings[property_value])

        for padding_index in range(0, max_string_variables - len(string_variables_list)):
            for _ in StringProperties:
                property_value = Padding
                populate_property_value(self.parent_ps, self.property_encodings[property_value])

        # output string with string only properties
        output_strings = [parent_output['out'] for parent_output in self.parent_input_output]
        for StringProperty in StringProperties:
            property_value = StringProperty(output_strings)
            populate_property_value(self.parent_ps, self.property_encodings[property_value])

        # input strings and output string with string-string properties
        for string_variable in string_variables_list:
            for InputStringOutputStringProperty in InputStringOutputStringProperties:
                property_value = InputStringOutputStringProperty(self.parent_input_output, string_variable)
                populate_property_value(self.parent_ps, self.property_encodings[property_value])

        for padding_index in range(0, max_string_variables - len(string_variables_list)):
            for _ in InputStringOutputStringProperties:
                property_value = Padding
                populate_property_value(self.parent_ps, self.property_encodings[property_value])

    def insert(self, program):
        self.batch_jobs.append(program)

    def process_batch_jobs(self):

        batch_size = 100000
        total_jobs = len(self.batch_jobs)

        for job_index in range(0, total_jobs, batch_size):

            current_batch = self.batch_jobs[job_index:job_index + batch_size]
            current_batch_ps = []

            for program in current_batch:

                test_row = self.parent_ps.copy()
                child_input_outputs = []

                for index, parent_input in enumerate(self.parent_input_output):
                    child_input_output = parent_input.copy()
                    child_output = program.interpret(child_input_output)
                    child_input_output['cout'] = child_output
                    child_input_output['out'] = self.parent_input_output[index]['out']
                    child_input_outputs.append(child_input_output)

                outputs = [output['cout'] for output in child_input_outputs]

                # boolean output of subexpression with boolean only properties
                if program.getReturnType() == BOOL_TYPES['type']:
                    for BooleanProperty in BooleanProperties:
                        property_value = BooleanProperty(outputs)
                        populate_property_value(test_row, self.property_encodings[property_value])
                else:
                    for _ in BooleanProperties:
                        property_value = Padding
                        populate_property_value(test_row, self.property_encodings[property_value])

                # integer output of expression with integer only properties -
                if program.getReturnType() == INT_TYPES['type']:
                    for IntegerProperty in IntegerProperties:
                        property_value = IntegerProperty(outputs)
                        populate_property_value(test_row, self.property_encodings[property_value])
                else:
                    for _ in IntegerProperties:
                        property_value = Padding
                        populate_property_value(test_row, self.property_encodings[property_value])

                # string output of subexpression with string only properties
                if program.getReturnType() == STR_TYPES['type']:
                    for StringProperty in StringProperties:
                        property_value = StringProperty(outputs)
                        populate_property_value(test_row, self.property_encodings[property_value])
                else:
                    for _ in StringProperties:
                        property_value = Padding
                        populate_property_value(test_row, self.property_encodings[property_value])

                # integer output of subexpression and string output of main expression with integer-string properties
                if program.getReturnType() == INT_TYPES['type']:
                    for InputInterOutputStringProperty in InputIntegerOutputStringProperties:
                        property_value = InputInterOutputStringProperty(child_input_outputs, 'cout')
                        populate_property_value(test_row, self.property_encodings[property_value])
                else:
                    for _ in InputIntegerOutputStringProperties:
                        property_value = Padding
                        populate_property_value(test_row, self.property_encodings[property_value])

                # string output of subexpression and string output of main expression with string-string properties
                if program.getReturnType() == STR_TYPES['type']:
                    for InputStringOutputStringProperty in InputStringOutputStringProperties:
                        property_value = InputStringOutputStringProperty(child_input_outputs, 'cout')
                        populate_property_value(test_row, self.property_encodings[property_value])
                else:
                    for _ in InputStringOutputStringProperties:
                        property_value = Padding
                        populate_property_value(test_row, self.property_encodings[property_value])

                current_batch_ps.append(test_row)

            current_batch_predictions = BustleModel.predict(np.array(current_batch_ps))

            for program_index, program in enumerate(current_batch):

                program_size = program.size

                # Reweighing the size of the program as per BUSTLE algorithm using the neural model

                program_probability = current_batch_predictions[program_index]

                if program_probability <= 0.1:
                    program_size += 5
                elif program_probability <= 0.2:
                    program_size += 4
                elif program_probability <= 0.3:
                    program_size += 3
                elif program_probability <= 0.4:
                    program_size += 2
                elif program_probability <= 0.6:
                    program_size += 1

                program.size = program_size

                if program.size not in self.plist:
                    self.plist[program.size] = {}

                if program.getReturnType() not in self.plist[program.size]:
                    self.plist[program.size][program.getReturnType()] = []

                self.plist[program.size][program.getReturnType()].append(program)
                self.number_programs += 1

        self.batch_jobs.clear()

    def init_insert(self, program):

        if program.size not in self.plist:
            self.plist[program.size] = {}

        if program.getReturnType() not in self.plist[program.size]:
            self.plist[program.size][program.getReturnType()] = []

        self.plist[program.size][program.getReturnType()].append(program)
        self.number_programs += 1

    def init_plist(self, string_literals_list, integer_literals_list,
                   string_variables_list, integer_variables_list):
        for string_literal in string_literals_list:
            init_program = StrLiteral(string_literal)
            self.init_insert(init_program)

        for integer_literal in integer_literals_list:
            init_program = IntLiteral(integer_literal)
            self.init_insert(init_program)

        for str_var in string_variables_list:
            init_program = StrVar(str_var)
            self.init_insert(init_program)

        for int_var in integer_variables_list:
            init_program = IntVar(int_var)
            self.init_insert(init_program)

        # self.process_batch_jobs()

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


class BottomUpSearch:

    def __init__(self, string_variables_list, integer_variables_list, input_output):
        self._variables = string_variables_list + integer_variables_list
        self._input_output = input_output
        self.plist = ProgramList(string_variables_list, integer_variables_list, input_output)
        self._outputs = set()
        self.closed_list = set()

    def is_correct(self, p):
        is_program_correct = True

        for inout in self._input_output:
            env = self.init_env(inout)
            out = p.interpret(env)
            if out != inout['out']:
                is_program_correct = False

        return is_program_correct

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

    def grow(self, operations, size):
        new_programs = []
        for operation in operations:
            for new_program in operation.grow(self.plist, size):
                # print(p.toString)
                if new_program.toString() not in self.closed_list and not self.has_equivalent(new_program):
                    self.closed_list.add(new_program)
                    new_programs.append(new_program)
                    yield new_program

        for new_program in new_programs:
            self.plist.insert(new_program)

        self.plist.process_batch_jobs()

    def search(self, bound, operations, string_literals_list, integer_literals_list,
               string_variables_list,
               integer_variables_list):

        self.plist.init_plist(string_literals_list, integer_literals_list, string_variables_list,
                              integer_variables_list)

        logging.info('Number of programs: ' + str(self.plist.get_number_programs()))

        number_evaluations = 0
        current_size = 0

        while current_size <= bound:

            number_evaluations_bound = 0

            for new_program in self.grow(operations, current_size):
                number_evaluations += 1
                number_evaluations_bound += 1
                is_p_correct = self.is_correct(new_program)
                if is_p_correct:
                    return new_program, number_evaluations

            logging.info('Size: ' + str(current_size) + ' Evaluations: ' + str(number_evaluations_bound))
            current_size += 1

        return None, number_evaluations

    def synthesize(self, bound, operations, string_literals_list, integer_literals_list,
                   string_variables_list,
                   integer_variables_list):

        BustlePCFG.initialize(operations, string_literals_list, integer_literals_list,
                              string_variables_list)

        program_solution, evaluations = self.search(bound, operations, string_literals_list, integer_literals_list,
                                                    string_variables_list, integer_variables_list)

        return program_solution, evaluations


def load_bustle_model():
    logging.info("Loading bustle model....")
    global BustleModel
    model_filename = models_directory + "EncodedBustleModelForPS.hdf5"
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    BustleModel = keras_model.load_model(model_filename)


if __name__ == "__main__":

    TaskId = None
    log_filename = logs_directory + "encoded_bustle_output.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    if len(sys.argv) == 2:
        slurm_task_id = sys.argv[1]
        TaskId = int(slurm_task_id) - 1
        logging.basicConfig(filename=log_filename,
                            filemode='a',
                            format="[Task: " + str(TaskId) + "] " + '%(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    else:
        logging.basicConfig(filename=log_filename,
                            filemode='a',
                            format='%(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    load_bustle_model()

    # with open("simple_benchmarks_bustle.txt") as f:
    with open(config_directory + "bustle_benchmarks.txt") as f:
        benchmarks = f.read().splitlines()

    logging.info('TaskId: ' + str(TaskId))
    benchmark = None
    experiment_begin_time = round(time.time())
    while True:
        current_time = round(time.time())
        if current_time - experiment_begin_time > 200:
            break
        for count, filename in enumerate(benchmarks):
            if count > 5:
                break

            benchmark = filename

            specification_parser = StrParser(benchmark)
            specifications = specification_parser.parse()
            logging.info('Count: ' + str(count))
            logging.info("\n")

            dsl_functions = [IntPlus, StrConcat, IntFirstIndexOf, IntIndexOf, StrLeftSubstr,
                             IntLength, StrSubstr, IntMinus, StrReplaceAdd, StrRightSubstr,
                             StrTrim, StrLower, StrUpper, StrProper, StrRepeat, StrReplace,
                             StrReplaceOccurence, StrIntToStr, StrIte, BoolEqual, BoolGreaterThan,
                             BoolGreaterThanEqual]

            string_variables = specifications[0]
            string_literals = specifications[1]
            integer_variables = specifications[2]
            integer_literals = specifications[3]

            input_output_examples = specifications[4]

            synthesizer = BottomUpSearch(string_variables, integer_variables, input_output_examples)
            logging.info(str(datetime.now()))
            begin_time = datetime.now()
            solution, num = synthesizer.synthesize(40, dsl_functions,
                                                   string_literals,
                                                   integer_literals,
                                                   string_variables,
                                                   integer_variables)

            # print(synthesizer.plist.get_number_programs())
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
