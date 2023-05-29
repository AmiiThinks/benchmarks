import re
from os.path import expanduser

scratch_directory = "../scratch/"
PATH_TO_STR_BENCHMARKS = "../sygus_string_tasks/"
config_directory = "../config/"
models_directory = "../models/"
logs_directory = "../logs/"
data_directory = "../data/"

# sygus parser constants
NT_STRING = "ntString String"
NT_INT = "ntInt Int"
CONSTRAINT = "constraint"
STRING_VAR = "string"
INTEGER_VAR = "integer"
EMPTY_STRING = "\"\""

# Regex properties
regex_digit = re.compile('\d')
regex_only_digits = re.compile('^\d+$')
regex_alpha = re.compile('[a-zA-Z]')
regex_alpha_only = re.compile('^[a-zA-Z]+$')

# String type
STR_TYPES = {'type': 'str'}

# Integer type
INT_TYPES = {'type': 'integer'}

# Boolean type
BOOL_TYPES = {'type': 'boolean'}
