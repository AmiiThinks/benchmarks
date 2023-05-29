class BustlePCFG:
    __instance = None

    @staticmethod
    def get_instance():
        if BustlePCFG.__instance is None:
            raise Exception("Need to initialize the grammar first")

        return BustlePCFG.__instance

    @staticmethod
    def initialize(operations, string_literals, integer_literals, boolean_literals, string_variables,
                   integer_variables):
        BustlePCFG(operations, string_literals, integer_literals,
                   boolean_literals, string_variables, integer_variables)

    def __init__(self, operations, string_literals, integer_literals, boolean_literals, string_variables,
                 integer_variables):
        self.uniform_grammar = {}
        self.grammar = {}
        self.number_rules = len(operations) + len(string_literals) + len(integer_literals) + len(
            boolean_literals) + len(string_variables) + len(integer_variables)
        self.program_id = 0
        BustlePCFG.__instance = self

    def get_cost(self, p):
        return 1

    def get_cost_by_name(self, name):
        return 1

    def get_program_id(self):
        self.program_id += 1
        return self.program_id
