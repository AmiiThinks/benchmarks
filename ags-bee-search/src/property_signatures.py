import re
regex_digit = re.compile('\d')
regex_only_digits = re.compile('^\d+$')
regex_alpha = re.compile('[a-zA-Z]')
regex_alpha_only = re.compile('^[a-zA-Z]+$')

# How we represent properties
# Padding is needed when some properties are not applicable to a given benchmark
AllTrue = -1
Mixed = 0
AllFalse = 1
Padding = 2

# Encoded properties
EncodedMixed = [1, 0, 0, 0]
EncodedAllFalse = [0, 1, 0, 0]
EncodedAllTrue = [0, 0, 1, 0]
EncodedPadding = [0, 0, 0, 1]

# Properties acting only on the string
def is_string_empty(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if len(program_input) == 0:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_single_char(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if len(program_input.strip()) == 1:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_short_string(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if len(program_input.strip()) <= 5:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_lower_case(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if program_input.islower():
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_upper_case(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if program_input.isupper():
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_contains_space(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if " " in program_input:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_contains_comma(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if "," in program_input:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_contains_period(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if "." in program_input:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_contains_dash(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if "-" in program_input:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_contains_slash(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if "/" in program_input:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_contains_digit(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if regex_digit.search(program_input) is not None:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_contains_only_digits(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if regex_only_digits.search(program_input) is not None:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_contains_letters(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if regex_alpha.search(program_input) is not None:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_contains_letters_only(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if regex_alpha_only.search(program_input) is not None:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


# Properties acting only on the integer
def is_zero(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if program_input == 0:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_one(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if program_input == 1:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_two(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if program_input == 2:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_negative(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if program_input < 0:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_small(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if 0 < program_input <= 3:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_medium(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if 3 < program_input <= 9:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_large(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if program_input > 9:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

# Properties acting only on Boolean
def is_true(inputs):
    is_true_present = False
    is_false_present = False
    for program_input in inputs:
        if program_input:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

# Properties acting on input integer and output string
def is_less_than_output_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input < len(output):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_less_than_or_equal_output_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input <= len(output):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_equal_output_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input == len(output):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_greater_than_or_equal_output_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input >= len(output):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_greater_than_output_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input > len(output):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse
# don't mind the name of the function
def is_very_closer_to_output_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if abs(program_input - len(output)) <= 1:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_closer_to_output_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if abs(program_input - len(output)) <= 3:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


# Properties acting on input string and output integer
def is_less_than_input_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if output < len(program_input):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_less_than_or_equal_input_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if output <= len(program_input):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_equal_input_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if output == len(program_input):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_greater_than_or_equal_input_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if output >= len(program_input):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_greater_than_input_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if output > len(program_input):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_very_closer_to_input_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if abs(len(program_input) - output) <= 1:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def is_closer_to_input_length(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if abs(len(program_input) - output) <= 3:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

# Properties acting on input string and output string
def output_contains_input(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input in output:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def output_starts_with_input(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if output.startswith(program_input):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def output_ends_with_input(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if output.endswith(program_input):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_contains_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if output in program_input:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_starts_with_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input.startswith(output):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_ends_with_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input.endswith(output):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def output_contains_input_ignorecase(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input.lower() in output.lower():
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def output_starts_with_input_ignorecase(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if output.lower().startswith(program_input.lower()):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def output_ends_with_input_ignorecase(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if output.lower().endswith(program_input.lower()):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_contains_output_ignorecase(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if output.lower() in program_input.lower():
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_starts_with_output_ignorecase(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input.lower().startswith(output.lower()):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_ends_with_output_ignorecase(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input.lower().endswith(output.lower()):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_equals_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input == output:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_equals_output_ignorecase(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input.lower() == output.lower():
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_same_length_as_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if len(program_input) == len(output):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_shorter_than_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if len(program_input) < len(output):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_longer_than_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if len(program_input) > len(output):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

# Properties acting on input integer and output integer
def output_perfectly_divisble_by_input(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if (program_input != 0) and (output % program_input == 0):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_perfectly_divisble_by_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if (output != 0) and (program_input % output == 0):
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_less_than_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input < output:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_less_than_or_equal_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input <= output:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_equal_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input == output:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_greater_than_or_equal_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input >= output:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_greater_than_output(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input > output:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_output_both_zero(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input == 0 and output == 0:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

def input_output_both_one(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input == 1 and output == 1:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


def input_output_both_even(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input % 2 == 0 and output % 2 == 0:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


def input_output_both_odd(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if program_input % 2 == 1 and output % 2 == 1:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


def input_output_very_close(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if abs(program_input - output) <= 1:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


def input_output_close(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if abs(program_input - output) <= 3:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


# Properties for integer inputs and boolean outputs
def is_parsed_input_and_output_true(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if bool(program_input) == True and output == True:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


def is_parsed_input_and_output_false(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if bool(program_input) == False and output == False:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


def is_parsed_input_and_output_same(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if bool(program_input) == output:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse

# Properties for string inputs and boolean outputs
def is_parsed_input_len_and_output_true(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if bool(len(program_input)) == True and output == True:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


def is_parsed_input_len_and_output_false(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if bool(len(program_input)) == False and output == False:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


def is_parsed_input_len_and_output_same(input_output, key):
    is_true_present = False
    is_false_present = False
    for in_out in input_output:
        program_input = in_out[key]
        output = in_out['out']
        if bool(len(program_input)) == output:
            is_true_present = True
        else:
            is_false_present = True

    if is_true_present and is_false_present:
        return Mixed
    elif is_true_present:
        return AllTrue
    else:
        return AllFalse


StringProperties = [
    is_string_empty,
    is_single_char,
    is_short_string,
    is_lower_case,
    is_upper_case,
    is_contains_space,
    is_contains_comma,
    is_contains_period,
    is_contains_dash,
    is_contains_slash,
    is_contains_digit,
    is_contains_only_digits,
    is_contains_letters,
    is_contains_letters_only
]

IntegerProperties = [
    is_zero,
    is_one,
    is_two,
    is_negative,
    is_small,
    is_medium,
    is_large
]

BooleanProperties = [is_true]

InputStringOutputStringProperties = [
    output_contains_input,
    output_starts_with_input,
    output_ends_with_input,
    input_contains_output,
    input_starts_with_output,
    input_ends_with_output,
    output_contains_input_ignorecase,
    output_starts_with_input_ignorecase,
    output_ends_with_input_ignorecase,
    input_contains_output_ignorecase,
    input_starts_with_output_ignorecase,
    input_ends_with_output_ignorecase,
    input_equals_output,
    input_equals_output_ignorecase,
    input_same_length_as_output,
    input_shorter_than_output,
    input_longer_than_output
]

InputIntegerOutputStringProperties = [
    is_less_than_output_length,
    is_less_than_or_equal_output_length,
    is_equal_output_length,
    is_greater_than_or_equal_output_length,
    is_greater_than_output_length,
    is_very_closer_to_output_length,
    is_closer_to_output_length
]

# new properties for all sygus tasks
InputStringOutputIntegerProperties = [
    is_less_than_input_length,
    is_less_than_or_equal_input_length,
    is_equal_input_length,
    is_greater_than_or_equal_input_length,
    is_greater_than_input_length,
    is_very_closer_to_input_length,
    is_closer_to_input_length
]


InputIntegerOutputIntegerProperties = [
    output_perfectly_divisble_by_input,
    input_perfectly_divisble_by_output,
    input_less_than_output,
    input_less_than_or_equal_output,
    input_equal_output,
    input_greater_than_or_equal_output,
    input_greater_than_output,
    input_output_both_zero,
    input_output_both_one,
    input_output_both_even,
    input_output_both_odd,
    input_output_very_close,
    input_output_close
]

InputIntegerOutputBoolProperties = [
    is_parsed_input_and_output_true,
    is_parsed_input_and_output_false,
    is_parsed_input_and_output_same
]

InputStringOutputBoolProperties = [
    is_parsed_input_len_and_output_true,
    is_parsed_input_len_and_output_false,
    is_parsed_input_len_and_output_same

]
