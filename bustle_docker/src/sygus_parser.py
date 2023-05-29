from utils import *


class StrParser:
    """
    Returns:
            str_var: variables to represent input string data
            str_literals: string constants
            int_var: variables to represent input int data
            int_literals: integer literals
            input: input values (for str_var or int_var)
            output: output string database
    """

    def __init__(self, filename):
        self.str_var = []
        self.str_literals = []
        self.int_var = []
        self.int_literals = []
        self.input = []
        self.output = []
        self.problem = filename
        self.test_cases = []
        self.nt_ops = []

    def reset(self):
        self.str_var = []
        self.str_literals = []
        self.int_var = []
        self.int_literals = []
        self.input = []
        self.output = []
        self.test_cases = []
        self.nt_ops = []

    def parse_str_literals(self, line):
        indices = [index for index, character in enumerate(
            line) if character == '"']
        for i in range(0, len(indices), 2):
            self.str_literals.append(line[indices[i] + 1:indices[i + 1]])

    def parse_vars(self, line, var_type):
        temp = line.strip().split(" ")
        bank = []
        if len(temp) == 1 and temp[0] == '':
            bank = []
        else:
            bank = temp

        if var_type == STRING_VAR:
            self.str_var = bank
        elif var_type == INTEGER_VAR:
            self.int_var = bank

    def parse_int_literals(self, line):
        temp = line.strip().split(" ")
        if len(temp) == 1 and temp[0] == '':
            self.int_literals = []
        else:
            self.int_literals = [int(integer) for integer in temp]

    def parse_io_pair(self, line):
        # (constraint (= (f "1/17/16-1/18/17" 1) "1/17/16")) ==> "1/17/16-1/18/17" 1) "1/17/16"))
        io = line.split("(f")[1].strip()

        # "1/17/16-1/18/17" 1) "1/17/16")) ==> ["1/17/16-1/18/17" 1, "1/17/16", ..]
        io_splitted = io.split(")", 1)

        inp = io_splitted[0].strip()

        out = io_splitted[1].strip()[:-2]

        inp = self.process_input(inp)
        out = self.process_output(out)

        self.input += [inp]
        self.output.append(out)

    def process_input(self, p_input):

        if "\"" in p_input:
            if "\"\"" in p_input:
                p_input = p_input.replace("\"\"", "\"empty_string_here_sa\"")
            inp = [i.strip() for i in p_input.split("\"")]

        else:
            inp = [i.strip() for i in p_input.split(" ")]

        inp = list(filter(lambda x: x != "", inp))
        inp = [i.replace("empty_string_here_sa", "") for i in inp]

        inp = list(map(self.parse_type, inp))
        temp = []
        count = 0
        for _ in range(len(self.str_var)):
            temp.append(str(inp[count]))
            count += 1
        for _ in range(len(self.int_var)):
            temp.append(int(inp[count]))
            count += 1
        return temp

    def parse_type(self, value):
        try:
            value = int(value)
        except:
            pass
        return value

    def process_output(self, output):
        if "\"" in output:
            return output.replace("\"", "")
        try:
            output = int(output)
        except:
            if 'true' in output:
                output = True
            elif 'false' in output:
                output = False
        return output

    def read(self, filename):
        # self.reset()
        f = open(filename, "r")

        lines = f.readlines()
        for step, line in enumerate(lines):

            if NT_STRING in line:
                str_var = lines[step + 1].strip()
                str_literals = lines[step + 2].strip()

                self.parse_vars(str_var, STRING_VAR)
                self.parse_str_literals(str_literals)

            if NT_INT in line:
                int_var = lines[step + 1].strip()
                int_literals = lines[step + 2].strip()

                self.parse_vars(int_var, INTEGER_VAR)
                self.parse_int_literals(int_literals)

            if CONSTRAINT in line:
                io_pair = line.strip()
                self.parse_io_pair(io_pair)

        f.close()

    def transform_outputs(self):
        test_cases = []
        for index, input_value in enumerate(self.input):
            test_case = {}
            count = 0
            for v in self.str_var:
                test_case[v] = input_value[count]
                count += 1

            for v in self.int_var:
                test_case[v] = input_value[count]
                count += 1

            test_case['out'] = self.output[index]

            test_cases.append(test_case)

        self.test_cases = test_cases

    def get_attrs(self):

        self.transform_outputs()
        return [self.str_var, self.str_literals, self.int_var, self.int_literals, self.test_cases, self.problem]

    def parse(self):

        self.read(PATH_TO_STR_BENCHMARKS + '/' + self.problem)
        # self.problem = self.filename
        return self.get_attrs()
