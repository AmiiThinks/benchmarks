import re
from property_signatures import *

# paths relative to the main.py file.
PATH_TO_STR_BENCHMARKS = "../sygus_string_tasks/"
config_directory = "../config/"
models_directory = "../models/"
logs_directory = "../logs/"

# sygus parser constants
NT_STRING = "ntString String"
NT_INT = "ntInt Int"
CONSTRAINT = "constraint"
STRING_VAR = "string"
INTEGER_VAR = "integer"
EMPTY_STRING = "\"\""

MAX_ARITY = 4

# Regex properties for property signatures
regex_digit = re.compile('\d')
regex_only_digits = re.compile('^\d+$')
regex_alpha = re.compile('[a-zA-Z]')
regex_alpha_only = re.compile('^[a-zA-Z]+$')


# Util functions for beesearch
def decimal_place_converter(number):
    return float("{:.2f}".format(number))


def populate_property_value(property_signature, property_encoding):
    for encoding in property_encoding:
        property_signature.append(encoding)


# Calculate property signatures for the problem
def calculate_ps_for_problem(
    self, string_variables_list, integer_variables_list
):
    """

        # Please do not change the code below.
        # This part essentially computes and save the property signatures for the benchmark.
        Property Signature composition:
        main_program:

        1. input strings with string only properties (4*14 = 56)
        # b/c max 4 str vars allowed in dsl.
        # if the number of vars are less, we pad the property signature with padding property.
        2. input integer with integer only properties (1*7 = 7)
        3. output string with string only properties (1*14 = 14)
        4. output integer with integer only properties (1*7 = 7)
        5. output boolean with boolean only properties (1*1 = 1)
        6. input integer and output string with integer-string properties (1*7 =7)
        7. input integer and output integer with integer-integer properties (1*13 = 13)
        8. input integer and output boolean with integer-boolean properties (1*3 = 3)
        9. input strings and output string with string-string properties (4*17 = 68)
        10. input strings and output integer with string-integer properties (4*7 = 28)
        11. input strings and output boolean with string-boolean properties (4*3 = 12)

        sub_program: (the ones we encounter during our search)
        12. boolean output with boolean only properties (1*1 = 1)
        13. integer output with integer only properties (1*7 = 7)
        14. string output with string only properties (1*14 = 14)
        15. integer output and string output of main program with integer-string properties (1*7 = 7)
        16. string output and string output of main program with string-string properties (1*17 = 17)
        17. integer output and integer output of main program with integer-string properties (1*13 = 13)
        18. string output and integer output of main program with string-string properties (1*7 = 7)
        19. integer output and boolean output of main program with integer-bool properties (1*3 = 3)
        20. string output and boolean output of main program with string-bool properties (1*3 = 3)

    # Process: We try to compute each property based on the type, if type matches, we compute property, else we pad with padding property.

    """

    # For property signatures
    max_string_variables = 4
    max_integer_variables = 1
    # Adding property signatures for the benchmark.

    # For inputs.
    # 1- input strings with string only properties
    for string_variable in string_variables_list:
        input_strings = [parent_input.get(
            string_variable) for parent_input in self.parent_input_output]
        for StringProperty in StringProperties:
            property_value = StringProperty(input_strings)
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    for padding_index in range(0, max_string_variables - len(string_variables_list)):
        for _ in StringProperties:
            property_value = Padding
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    # 2- input integer with integer only properties
    for integer_variable in integer_variables_list:
        input_integers = [parent_input.get(
            integer_variable) for parent_input in self.parent_input_output]
        for IntegerProperty in IntegerProperties:
            property_value = IntegerProperty(input_integers)
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    for padding_index in range(0, max_integer_variables - len(integer_variables_list)):
        for _ in IntegerProperties:
            property_value = Padding
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    # For outputs, based on the type of the output.
    task_outputs = [parent_output['out']
                    for parent_output in self.parent_input_output]
    output_type = "str"
    if isinstance(task_outputs[0], int):
        output_type = "int"
    elif isinstance(task_outputs[0], bool):
        output_type = "bool"

    self.parent_output_type = output_type

    # 3- output string with string only properties
    output_strings = [parent_output['out']
                      for parent_output in self.parent_input_output]
    if output_type == "str":
        for StringProperty in StringProperties:
            property_value = StringProperty(output_strings)
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])
    else:  # padding
        for _ in StringProperties:
            property_value = Padding
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    # 4- output integer with integer only properties
    output_integers = [parent_output['out']
                       for parent_output in self.parent_input_output]
    if output_type == "int":
        for IntegerProperty in IntegerProperties:
            property_value = IntegerProperty(output_integers)
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])
    else:
        for _ in IntegerProperties:
            property_value = Padding
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    # 5- output boolean with boolean only properties
    output_bools = [parent_output['out']
                    for parent_output in self.parent_input_output]
    if output_type == "bool":
        for BooleanProperty in BooleanProperties:
            property_value = BooleanProperty(output_bools)
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])
    else:
        for _ in BooleanProperties:
            property_value = Padding
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    # 6- input integer and output string with integer-string properties
    for integer_variable in integer_variables_list:
        if output_type == "str":
            for InputIntegerOutputStringProperty in InputIntegerOutputStringProperties:
                property_value = InputIntegerOutputStringProperty(
                    self.parent_input_output, integer_variable)
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])
        else:
            for _ in InputIntegerOutputStringProperties:
                property_value = Padding
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])

    for padding_index in range(0, max_integer_variables - len(integer_variables_list)):
        for _ in InputIntegerOutputStringProperties:
            property_value = Padding
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    # 7- input integer and output integer with integer-integer properties
    for integer_variable in integer_variables_list:
        if output_type == "int":
            for InputIntegerOutputIntegerProperty in InputIntegerOutputIntegerProperties:
                property_value = InputIntegerOutputIntegerProperty(
                    self.parent_input_output, integer_variable)
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])
        else:
            for _ in InputIntegerOutputIntegerProperties:
                property_value = Padding
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])

    for padding_index in range(0, max_integer_variables - len(integer_variables_list)):
        for _ in InputIntegerOutputIntegerProperties:
            property_value = Padding
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    # 8- input integer and output boolean with integer-boolean properties
    for integer_variable in integer_variables_list:
        if output_type == "bool":
            for InputIntegerOutputBoolProperty in InputIntegerOutputBoolProperties:
                property_value = InputIntegerOutputBoolProperty(
                    self.parent_input_output, integer_variable)
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])
        else:
            for _ in InputIntegerOutputBoolProperties:
                property_value = Padding
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])

    for padding_index in range(0, max_integer_variables - len(integer_variables_list)):
        for _ in InputIntegerOutputBoolProperties:
            property_value = Padding
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    # 9- input strings and output string with string-string properties
    for string_variable in string_variables_list:
        if output_type == "str":
            for InputStringOutputStringProperty in InputStringOutputStringProperties:
                property_value = InputStringOutputStringProperty(
                    self.parent_input_output, string_variable)
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])
        else:
            for _ in InputStringOutputStringProperties:
                property_value = Padding
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])

    for padding_index in range(0, max_string_variables - len(string_variables_list)):
        for _ in InputStringOutputStringProperties:
            property_value = Padding
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    # 10- input strings and output integer with string-integer properties
    for string_variable in string_variables_list:
        if output_type == "int":
            for InputStringOutputIntegerProperty in InputStringOutputIntegerProperties:
                property_value = InputStringOutputIntegerProperty(
                    self.parent_input_output, string_variable)
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])
        else:
            for _ in InputStringOutputIntegerProperties:
                property_value = Padding
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])

    for padding_index in range(0, max_string_variables - len(string_variables_list)):
        for _ in InputStringOutputIntegerProperties:
            property_value = Padding
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])

    # 11- input strings and output boolean with string-bool properties
    for string_variable in string_variables_list:
        if output_type == "bool":
            for InputStringOutputBoolProperty in InputStringOutputBoolProperties:
                property_value = InputStringOutputBoolProperty(
                    self.parent_input_output, string_variable)
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])
        else:
            for _ in InputStringOutputBoolProperties:
                property_value = Padding
                populate_property_value(
                    self.parent_ps, self.property_encodings[property_value])

    for padding_index in range(0, max_string_variables - len(string_variables_list)):
        for _ in InputStringOutputBoolProperties:
            property_value = Padding
            populate_property_value(
                self.parent_ps, self.property_encodings[property_value])


# Calculate ps for each sub-program
def populate_sub_program_ps(self, program, test_row, outputs, child_input_outputs, STR_TYPES, INT_TYPES, BOOL_TYPES):
    # Calculate the property signatures of the current program.
    # boolean output of subexpression with boolean only properties
    if program.getReturnType() == BOOL_TYPES['type']:
        for BooleanProperty in BooleanProperties:
            property_value = BooleanProperty(outputs)
            populate_property_value(
                test_row, self.property_encodings[property_value])
    else:
        for _ in BooleanProperties:
            property_value = Padding
            populate_property_value(
                test_row, self.property_encodings[property_value])

    # integer output of expression with integer only properties -
    if program.getReturnType() == INT_TYPES['type']:
        for IntegerProperty in IntegerProperties:
            property_value = IntegerProperty(outputs)
            populate_property_value(
                test_row, self.property_encodings[property_value])
    else:
        for _ in IntegerProperties:
            property_value = Padding
            populate_property_value(
                test_row, self.property_encodings[property_value])

    # string output of subexpression with string only properties
    if program.getReturnType() == STR_TYPES['type']:
        for StringProperty in StringProperties:
            property_value = StringProperty(outputs)
            populate_property_value(
                test_row, self.property_encodings[property_value])
    else:
        for _ in StringProperties:
            property_value = Padding
            populate_property_value(
                test_row, self.property_encodings[property_value])

    # integer output of subexpression and string output of main expression with integer-string properties
    if program.getReturnType() == INT_TYPES['type'] and self.parent_output_type == "str":
        for InputIntegerOutputStringProperty in InputIntegerOutputStringProperties:
            property_value = InputIntegerOutputStringProperty(
                child_input_outputs, 'cout')
            populate_property_value(
                test_row, self.property_encodings[property_value])
    else:
        for _ in InputIntegerOutputStringProperties:
            property_value = Padding
            populate_property_value(
                test_row, self.property_encodings[property_value])

    # string output of subexpression and string output of main expression with string-string properties
    if program.getReturnType() == STR_TYPES['type'] and self.parent_output_type == "str":
        for InputStringOutputStringProperty in InputStringOutputStringProperties:
            property_value = InputStringOutputStringProperty(
                child_input_outputs, 'cout')
            populate_property_value(
                test_row, self.property_encodings[property_value])
    else:
        for _ in InputStringOutputStringProperties:
            property_value = Padding
            populate_property_value(
                test_row, self.property_encodings[property_value])

    # integer output of subexpression and integer output of main expression with integer-integer properties
    if program.getReturnType() == INT_TYPES['type'] and self.parent_output_type == "int":
        for InputIntegerOutputIntegerProperty in InputIntegerOutputIntegerProperties:
            property_value = InputIntegerOutputIntegerProperty(
                child_input_outputs, 'cout')
            populate_property_value(
                test_row, self.property_encodings[property_value])
    else:
        for _ in InputIntegerOutputIntegerProperties:
            property_value = Padding
            populate_property_value(
                test_row, self.property_encodings[property_value])

    # string output of subexpression and integer output of main expression with string-integer properties
    if program.getReturnType() == STR_TYPES['type'] and self.parent_output_type == "int":
        for InputStringOutputIntegerProperty in InputStringOutputIntegerProperties:
            property_value = InputStringOutputIntegerProperty(
                child_input_outputs, 'cout')
            populate_property_value(
                test_row, self.property_encodings[property_value])
    else:
        for _ in InputStringOutputIntegerProperties:
            property_value = Padding
            populate_property_value(
                test_row, self.property_encodings[property_value])

    # integer output of subexpression and boolean output of main expression with integer-bool properties
    if program.getReturnType() == INT_TYPES['type'] and self.parent_output_type == "bool":
        for InputIntegerOutputBoolProperty in InputIntegerOutputBoolProperties:
            property_value = InputIntegerOutputBoolProperty(
                child_input_outputs, 'cout')
            populate_property_value(
                test_row, self.property_encodings[property_value])
    else:
        for _ in InputIntegerOutputBoolProperties:
            property_value = Padding
            populate_property_value(
                test_row, self.property_encodings[property_value])

    # string output of subexpression and boolean output of main expression with string-bool properties
    if program.getReturnType() == STR_TYPES['type'] and self.parent_output_type == "bool":
        for InputStringOutputBoolProperty in InputStringOutputBoolProperties:
            property_value = InputStringOutputBoolProperty(
                child_input_outputs, 'cout')
            populate_property_value(
                test_row, self.property_encodings[property_value])
    else:
        for _ in InputStringOutputBoolProperties:
            property_value = Padding
            populate_property_value(
                test_row, self.property_encodings[property_value])
