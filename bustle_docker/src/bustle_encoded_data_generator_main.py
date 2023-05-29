import logging
import os
import random
import sys
from datetime import datetime

import pandas as pd

from bustle_data import bustle_inputs_dict_generator, bustle_constants_extractor
from bustle_properties import *
from bustle_string_dsl import *
from utils import *


class ProgramList:

    def __init__(self):
        self.plist = {}
        self.number_programs = 0

    def insert(self, program):
        if program.size not in self.plist:
            self.plist[program.size] = {}

        if program.getReturnType() not in self.plist[program.size]:
            self.plist[program.size][program.getReturnType()] = []

        self.plist[program.size][program.getReturnType()].append(program)
        self.number_programs += 1

    def init_plist(self, string_literals_list, integer_literals_list,
                   string_variables_list):
        for string_literal in string_literals_list:
            init_program = StrLiteral(string_literal)
            self.insert(init_program)

        for integer_literal in integer_literals_list:
            init_program = IntLiteral(integer_literal)
            self.insert(init_program)

        for str_var in string_variables_list:
            init_program = StrVar(str_var)
            self.insert(init_program)

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

    def __init__(self, string_variables_list, input_output):
        self._variables = string_variables_list
        self._input_output = input_output
        self.plist = ProgramList()
        self._outputs = set()
        self.closed_list = set()

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
        for op in operations:
            for new_program in op.grow(self.plist, size):
                if new_program.toString() not in self.closed_list and not self.has_equivalent(new_program):
                    self.closed_list.add(new_program)
                    new_programs.append(new_program)
                    yield new_program

        for new_program in new_programs:
            self.plist.insert(new_program)

    def search(self, bound, operations, string_literals_list, integer_literals_list,
               string_variables_list):

        self.plist.init_plist(string_literals_list, integer_literals_list, string_variables_list)

        logging.info('Number of programs: ' + str(self.plist.get_number_programs()))

        number_evaluations = 0
        current_size = 0

        while current_size <= bound:

            number_evaluations_bound = 0

            for _ in self.grow(operations, current_size):
                number_evaluations += 1
                number_evaluations_bound += 1

            logging.info('Size: ' + str(current_size) + ' Evaluations: ' + str(number_evaluations_bound))
            current_size += 1

        return None, number_evaluations

    def synthesize(self, bound, operations, string_literals_list,
                   integer_literals_list,
                   string_variables_list):

        BustlePCFG.initialize(operations, string_literals_list, integer_literals_list,
                              string_variables_list)

        solution, evaluations = self.search(bound, operations, string_literals_list, integer_literals_list,
                                            string_variables_list)

        return solution, evaluations


def find_random_programs(string_variables_list, all_programs, existing_programs_set, number_of_programs):
    programs = []
    for program in all_programs:
        program_string = program.toString()
        all_string_variables_present = True
        for string_variable in string_variables_list:
            if string_variable not in program_string:
                all_string_variables_present = False

        if all_string_variables_present:
            programs.append(program)

    if len(programs) < number_of_programs:
        return

    for program in random.sample(programs, number_of_programs):
        existing_programs_set.add(program)


def populate_property_value(data_point, property_index, property_encoding):
    property_str = "prop" + str(property_index) + "_"
    for encoding_index, encoding_value in enumerate(property_encoding):
        data_point[property_str + str(encoding_index)] = encoding_value


def generate_training_data(program_synthesizer, gen_training_data_set, random_inputs,
                           string_variables_list):
    programs = program_synthesizer.plist.get_programs(8, STR_TYPES['type'])
    random_programs = set()
    property_encodings = {
        AllTrue: EncodedAllTrue,
        AllFalse: EncodedAllFalse,
        Mixed: EncodedMixed,
        Padding: EncodedPadding
    }

    """

            Property Signature composition:

                    main_program:

                    1. input strings with string only properties (3*14 = 42)
                    2. input integer with integer only properties (1*7 = 7)
                    3. output string with string only properties (1*14 = 14)
                    4. input integer and output string with integer-string properties (1*7 =7)
                    5. input strings and output string with string-string properties (3*17 = 51)

                    sub_program:

                    6. boolean output with boolean only properties (1*1 = 1)
                    7. integer output with integer only properties (1*7 = 7)
                    8. string output with string only properties (1*14 = 14)
                    9. integer output and string output of main program with integer-string properties (1*7 = 7)
                    10. string output and string output of main program with string-string properties (1*17 = 17)

    """

    find_random_programs(string_variables_list, programs, random_programs, 100)
    if len(random_programs) == 0:
        return
    random_programs = list(random_programs)

    for program in random_programs:
        positive_training_data_point = {}
        negative_training_data_point = {}

        parent_input_outputs = []

        for random_input in random_inputs:
            input_output = random_input.copy()
            output = program.interpret(input_output)
            input_output['out'] = output
            parent_input_outputs.append(input_output)

        property_index = 0
        max_string_variables = 3

        # PARENT PROPERTY SIGNATURE CALCULATION

        # Input strings string-only properties
        for string_variable_index, string_variable in enumerate(string_variables_list):
            input_strings = [parent_input_output.get(string_variable)
                             for parent_input_output in parent_input_outputs]
            for StringProperty in StringProperties:
                property_value = StringProperty(input_strings)
                property_index += 1
                populate_property_value(positive_training_data_point, property_index,
                                        property_encodings[property_value])
                populate_property_value(negative_training_data_point, property_index,
                                        property_encodings[property_value])

        for padding_index in range(0, max_string_variables - len(string_variables_list)):
            for _ in StringProperties:
                property_value = Padding
                property_index += 1
                populate_property_value(positive_training_data_point, property_index,
                                        property_encodings[property_value])
                populate_property_value(negative_training_data_point, property_index,
                                        property_encodings[property_value])

        # output string with string only properties
        output_strings = [parent_input_output['out'] for parent_input_output in parent_input_outputs]
        for StringProperty in StringProperties:
            property_value = StringProperty(output_strings)
            property_index += 1
            populate_property_value(positive_training_data_point, property_index,
                                    property_encodings[property_value])
            populate_property_value(negative_training_data_point, property_index,
                                    property_encodings[property_value])

        # input strings and output string with string-string properties
        for string_variable_index, string_variable in enumerate(string_variables_list):
            for StringProperty in InputStringOutputStringProperties:
                property_value = StringProperty(parent_input_outputs, string_variable)
                property_index += 1
                populate_property_value(positive_training_data_point, property_index,
                                        property_encodings[property_value])
                populate_property_value(negative_training_data_point, property_index,
                                        property_encodings[property_value])

        for padding_index in range(0, max_string_variables - len(string_variables_list)):
            for _ in InputStringOutputStringProperties:
                property_value = Padding
                property_index += 1
                populate_property_value(positive_training_data_point, property_index,
                                        property_encodings[property_value])
                populate_property_value(negative_training_data_point, property_index,
                                        property_encodings[property_value])

        # FINDING POSITIVE CHILD PROGRAM
        positive_child = None
        child_programs_set = set()
        program.getProgramIds(child_programs_set)
        child_size = 0
        while child_size < 4:
            positive_child = random.choice(list(child_programs_set))
            child_size = positive_child.size

        # FINDING NEGATIVE CHILD PROGRAM
        children = random.sample(program_synthesizer.plist.get_programs_all(6), 2)
        negative_child = children[0]
        if children[0] in child_programs_set:
            negative_child = children[1]

        positive_child_input_outputs = []
        negative_child_input_outputs = []
        pos_property_index = property_index
        neg_property_index = property_index

        for random_input_index, random_input in enumerate(random_inputs):
            positive_input_output = random_input.copy()
            negative_input_output = random_input.copy()
            positive_output = positive_child.interpret(positive_input_output)
            negative_output = negative_child.interpret(negative_input_output)
            positive_input_output['cout'] = positive_output
            negative_input_output['cout'] = negative_output
            positive_input_output['out'] = parent_input_outputs[random_input_index]['out']
            negative_input_output['out'] = parent_input_outputs[random_input_index]['out']
            positive_child_input_outputs.append(positive_input_output)
            negative_child_input_outputs.append(negative_input_output)

        # POSITIVE CHILD PROPERTY SIGNATURE CALCULATION

        outputs = [output['cout'] for output in positive_child_input_outputs]

        # boolean output of subexpression with boolean only properties
        if positive_child.getReturnType() == BOOL_TYPES['type']:
            for StringProperty in BooleanProperties:
                property_value = StringProperty(outputs)
                pos_property_index += 1
                populate_property_value(positive_training_data_point, pos_property_index,
                                        property_encodings[property_value])

        else:
            for _ in BooleanProperties:
                property_value = Padding
                pos_property_index += 1
                populate_property_value(positive_training_data_point, pos_property_index,
                                        property_encodings[property_value])

        # integer output of expression with integer only properties
        if positive_child.getReturnType() == INT_TYPES['type']:
            for StringProperty in IntegerProperties:
                property_value = StringProperty(outputs)
                pos_property_index += 1
                populate_property_value(positive_training_data_point, pos_property_index,
                                        property_encodings[property_value])

        else:
            for _ in IntegerProperties:
                property_value = Padding
                pos_property_index += 1
                populate_property_value(positive_training_data_point, pos_property_index,
                                        property_encodings[property_value])

        # string output of subexpression with string only properties
        if positive_child.getReturnType() == STR_TYPES['type']:
            for StringProperty in StringProperties:
                property_value = StringProperty(outputs)
                pos_property_index += 1
                populate_property_value(positive_training_data_point, pos_property_index,
                                        property_encodings[property_value])

        else:
            for _ in StringProperties:
                property_value = Padding
                pos_property_index += 1
                populate_property_value(positive_training_data_point, pos_property_index,
                                        property_encodings[property_value])

        # integer output of subexpression and string output of main expression with integer-string properties
        if positive_child.getReturnType() == INT_TYPES['type']:
            for StringProperty in InputIntegerOutputStringProperties:
                property_value = StringProperty(positive_child_input_outputs, 'cout')
                pos_property_index += 1
                populate_property_value(positive_training_data_point, pos_property_index,
                                        property_encodings[property_value])
        else:
            for _ in InputIntegerOutputStringProperties:
                property_value = Padding
                pos_property_index += 1
                populate_property_value(positive_training_data_point, pos_property_index,
                                        property_encodings[property_value])

        # string output of subexpression and string output of main expression with string-string properties
        if positive_child.getReturnType() == STR_TYPES['type']:
            for StringProperty in InputStringOutputStringProperties:
                property_value = StringProperty(positive_child_input_outputs, 'cout')
                pos_property_index += 1
                populate_property_value(positive_training_data_point, pos_property_index,
                                        property_encodings[property_value])
        else:
            for _ in InputStringOutputStringProperties:
                property_value = Padding
                pos_property_index += 1
                populate_property_value(positive_training_data_point, pos_property_index,
                                        property_encodings[property_value])

        # NEGATIVE CHILD PROPERTY SIGNATURE CALCULATION

        outputs = [output['cout'] for output in negative_child_input_outputs]

        # boolean output of subexpression with boolean only properties
        if negative_child.getReturnType() == BOOL_TYPES['type']:
            for StringProperty in BooleanProperties:
                property_value = StringProperty(outputs)
                neg_property_index += 1
                populate_property_value(negative_training_data_point, neg_property_index,
                                        property_encodings[property_value])
        else:
            for _ in BooleanProperties:
                property_value = Padding
                neg_property_index += 1
                populate_property_value(negative_training_data_point, neg_property_index,
                                        property_encodings[property_value])

        # integer output of expression with integer only properties
        if negative_child.getReturnType() == INT_TYPES['type']:
            for StringProperty in IntegerProperties:
                property_value = StringProperty(outputs)
                neg_property_index += 1
                populate_property_value(negative_training_data_point, neg_property_index,
                                        property_encodings[property_value])
        else:
            for _ in IntegerProperties:
                property_value = Padding
                neg_property_index += 1
                populate_property_value(negative_training_data_point, neg_property_index,
                                        property_encodings[property_value])

        # string output of subexpression with string only properties
        if negative_child.getReturnType() == STR_TYPES['type']:
            for StringProperty in StringProperties:
                property_value = StringProperty(outputs)
                neg_property_index += 1
                populate_property_value(negative_training_data_point, neg_property_index,
                                        property_encodings[property_value])
        else:
            for _ in StringProperties:
                property_value = Padding
                neg_property_index += 1
                populate_property_value(negative_training_data_point, neg_property_index,
                                        property_encodings[property_value])

        # integer output of subexpression and string output of main expression with integer-string properties
        if negative_child.getReturnType() == INT_TYPES['type']:
            for StringProperty in InputIntegerOutputStringProperties:
                property_value = StringProperty(negative_child_input_outputs, 'cout')
                neg_property_index += 1
                populate_property_value(negative_training_data_point, neg_property_index,
                                        property_encodings[property_value])
        else:
            for _ in InputIntegerOutputStringProperties:
                property_value = Padding
                neg_property_index += 1
                populate_property_value(negative_training_data_point, neg_property_index,
                                        property_encodings[property_value])

        # string output of subexpression and string output of main expression with string-string properties
        if negative_child.getReturnType() == STR_TYPES['type']:
            for StringProperty in InputStringOutputStringProperties:
                property_value = StringProperty(negative_child_input_outputs, 'cout')
                neg_property_index += 1
                populate_property_value(negative_training_data_point, neg_property_index,
                                        property_encodings[property_value])
        else:
            for _ in InputStringOutputStringProperties:
                property_value = Padding
                neg_property_index += 1
                populate_property_value(negative_training_data_point, neg_property_index,
                                        property_encodings[property_value])

        positive_training_data_point['label'] = 1
        negative_training_data_point['label'] = 0

        gen_training_data_set.append(positive_training_data_point)
        gen_training_data_set.append(negative_training_data_point)


if __name__ == "__main__":

    log_filename = logs_directory + "encoded_data_generator.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    if len(sys.argv) == 2:
        slurm_task_id = sys.argv[1]
        logging.basicConfig(filename=log_filename,
                            filemode='a',
                            format="[Task: " + str(slurm_task_id) + "] " + '%(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    else:
        logging.basicConfig(filename=log_filename,
                            filemode='a',
                            format='%(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    dsl_functions = [IntPlus, StrConcat, IntFirstIndexOf, IntIndexOf, StrLeftSubstr,
                     IntLength, StrSubstr, IntMinus, StrReplaceAdd, StrRightSubstr,
                     StrTrim, StrLower, StrUpper, StrProper, StrRepeat, StrReplace,
                     StrReplaceOccurence, StrIntToStr, StrIte, BoolEqual, BoolGreaterThan,
                     BoolGreaterThanEqual]

    string_literals = ['', '/', ' ', ',', '-', '.']
    integer_literals = [0, 1, 2, 3, 99]

    training_data_set = []
    property_data_set = []

    for search_index in range(0, 5):

        begin_time = datetime.now()
        logging.info("start time: " + str(datetime.now()))
        # number_of_examples = random.choice([1, 2, 3])
        number_of_examples = 3
        number_of_inputs = random.choice([1, 2, 3])

        random_io = bustle_inputs_dict_generator(number_of_inputs, number_of_examples)
        string_variables = list(random_io.keys())
        io_pairs = []
        io_strings = []

        for index in range(number_of_examples):
            io_pair = {}
            for input_index in range(number_of_inputs):
                io_pair[string_variables[input_index]] = random_io[string_variables[input_index]][index]
                io_strings.append(random_io[string_variables[input_index]][index])
            io_pairs.append(io_pair.copy())

        logging.info("examples: " + str(io_pairs))

        string_literals = bustle_constants_extractor(io_strings)
        synthesizer = BottomUpSearch(string_variables, io_pairs)
        p, num = synthesizer.synthesize(8, dsl_functions,
                                        string_literals,
                                        integer_literals,
                                        string_variables)

        generate_training_data(synthesizer, training_data_set, io_pairs, string_variables)
        logging.info("end time: " + str(datetime.now()))
        logging.info("Time taken: " + str(datetime.now() - begin_time))
        logging.info("training dataset size: " + str(len(training_data_set)))

    logging.info("training dataset size: " + str(len(training_data_set)))
    if len(training_data_set) != 0:
        dataFrame = pd.DataFrame(training_data_set)
        logging.info("Saving the training data to csv file...")
        dataFrame.to_csv(data_directory + "sygus_bustle_encoded_training_data.csv", index=False, mode='a',
                         header=not os.path.exists(data_directory + "sygus_bustle_encoded_training_data.csv"))
