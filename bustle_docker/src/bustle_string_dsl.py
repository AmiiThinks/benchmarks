import copy
import itertools

from cfg import BustlePCFG
from utils import *


class Str:
    def __init__(self):
        self.size = 0

    def getReturnType(self):
        return STR_TYPES['type']

    @classmethod
    def name(cls):
        return cls.__name__


class StrLiteral(Str):
    def __init__(self, value):
        self.value = value
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return '\"' + self.value + '\"'

    def interpret(self, env):
        return self.value

    def getProgramIds(self, program_ids):
        pass


class StrVar(Str):
    def __init__(self, name):
        self.value = name
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.value

    def interpret(self, env):
        return copy.deepcopy(env[self.value])

    def getProgramIds(self, program_ids):
        pass


class StrConcat(Str):
    ARITY = 2

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = x.size + y.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return 'concat(' + self.x.toString() + ", " + self.y.toString() + ")"

    def interpret(self, env):
        return self.x.interpret(env) + self.y.interpret(env)

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.x.getProgramIds(program_ids)
        self.y.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):

        combinations = list(itertools.product(range(1, size - 1), repeat=2))

        for combination in combinations:
            # skip if the cost combination exceeds the limit
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(StrConcat.name()) != size:
                continue

            # retrive bank of programs with costs c[0] and c[1]
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])

            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral) and prog1.toString() == EMPTY_STRING: continue
                for prog2 in layer2_prog:
                    if isinstance(prog2, StrLiteral) and prog2.toString() == EMPTY_STRING: continue
                    program = StrConcat(prog1, prog2)
                    yield program


class StrReplace(Str):
    ARITY = 3

    def __init__(self, input_str, old, new):
        self.str = input_str
        self.old = old
        self.new = new
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + old.size + new.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + '.replace(' + self.old.toString() + ", " + self.new.toString() + ")"

    def interpret(self, env):
        old_str = self.old.interpret(env)
        if old_str == "":
            return None
        return self.str.interpret(env).replace(old_str, self.new.interpret(env))

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)
        self.old.getProgramIds(program_ids)
        self.new.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 2), repeat=3))
        for combination in combinations:
            layer1, layer2, layer3 = combination
            if layer1 + layer2 + layer3 + BustlePCFG.get_instance().get_cost_by_name(StrReplace.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])
            layer3_prog = plist.get_programs(layer3, STR_TYPES['type'])
            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral):
                    continue
                for prog2 in layer2_prog:
                    p2_str = prog2.toString()
                    if p2_str == EMPTY_STRING:
                        continue
                    is_p2_var = isinstance(prog2, StrVar)
                    if is_p2_var:
                        continue
                    for prog3 in layer3_prog:
                        if prog3.toString() == p2_str:
                            continue
                        yield StrReplace(prog1, prog2, prog3)


class StrSubstr(Str):
    ARITY = 3

    def __init__(self, input_str, start, length):
        self.str = input_str
        self.start = start
        self.length = length
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + start.size + length.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + ".Substr(" + self.start.toString() + ":" + self.start.toString() + "+" + self.length.toString() + ")"

    def interpret(self, env):
        start_pos = self.start.interpret(env)
        substr_len = self.length.interpret(env)
        super_str = self.str.interpret(env)
        if start_pos <= 0 or substr_len <= 0:
            return None
        return super_str[start_pos: start_pos + substr_len]

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)
        self.start.getProgramIds(program_ids)
        self.length.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 2), repeat=3))
        for combination in combinations:
            layer1, layer2, layer3 = combination
            if layer1 + layer2 + layer3 + BustlePCFG.get_instance().get_cost_by_name(StrSubstr.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
            layer3_prog = plist.get_programs(layer3, INT_TYPES['type'])

            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral):
                    continue
                for prog2 in layer2_prog:
                    for prog3 in layer3_prog:
                        if prog2.toString() == prog3.toString():
                            continue
                        yield StrSubstr(prog1, prog2, prog3)


class StrIte(Str):
    ARITY = 3

    def __init__(self, condition, true_case, false_case):
        self.condition = condition
        self.true_case = true_case
        self.false_case = false_case
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = condition.size + true_case.size + false_case.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return "(if" + self.condition.toString() + " then " + self.true_case.toString() + " else " + self.false_case.toString() + ")"

    def interpret(self, env):
        if self.condition.interpret(env):
            return self.true_case.interpret(env)
        else:
            return self.false_case.interpret(env)

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.condition.getProgramIds(program_ids)
        self.true_case.getProgramIds(program_ids)
        self.false_case.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 2), repeat=3))
        for combination in combinations:
            layer1, layer2, layer3 = combination
            if layer1 + layer2 + layer3 + BustlePCFG.get_instance().get_cost_by_name(StrIte.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, BOOL_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])
            layer3_prog = plist.get_programs(layer3, STR_TYPES['type'])

            for prog1 in layer1_prog:
                for prog2 in layer2_prog:
                    for prog3 in layer3_prog:
                        yield StrIte(prog1, prog2, prog3)


class StrIntToStr(Str):
    ARITY = 1

    def __init__(self, input_int):
        self.int = input_int
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_int.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.int.toString() + ".IntToStr()"

    def interpret(self, env):
        return str(self.int.interpret(env))

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.int.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = range(1, size)
        for combination in combinations:
            layer1 = combination
            if layer1 + BustlePCFG.get_instance().get_cost_by_name(StrIntToStr.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, INT_TYPES['type'])

            for prog1 in layer1_prog:
                yield StrIntToStr(prog1)


# bustle additional classes
class StrReplaceAdd(Str):
    ARITY = 4

    def __init__(self, input_str, start, length, add_str):
        self.str = input_str
        self.start = start
        self.length = length
        self.add_str = add_str
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + start.size + length.size + add_str.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + "[:" + self.start.toString() + "] + " + self.add_str.toString() + " + " + self.str.toString() + "[" + self.start.toString() + "+" + self.length.toString() + ":]"

    def interpret(self, env):
        start_pos = self.start.interpret(env)
        replace_len = self.length.interpret(env)
        if start_pos <= 0 or replace_len < 0:
            return None
        super_str = self.str.interpret(env)
        replace_str = self.add_str.interpret(env)
        return super_str[:start_pos] + replace_str + super_str[start_pos + replace_len:]

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)
        self.start.getProgramIds(program_ids)
        self.length.getProgramIds(program_ids)
        self.add_str.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 3), repeat=4))
        for combination in combinations:
            layer1, layer2, layer3, layer4 = combination
            if layer1 + layer2 + layer3 + layer4 + BustlePCFG.get_instance().get_cost_by_name(StrSubstr.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
            layer3_prog = plist.get_programs(layer3, INT_TYPES['type'])
            layer4_prog = plist.get_programs(layer4, STR_TYPES['type'])

            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral):
                    continue
                for prog2 in layer2_prog:
                    for prog3 in layer3_prog:
                        if prog2.toString() == prog3.toString():
                            continue
                        for prog4 in layer4_prog:
                            yield StrReplaceAdd(prog1, prog2, prog3, prog4)


class StrReplaceOccurence(Str):
    ARITY = 4

    def __init__(self, input_str, old, new, count):
        self.str = input_str
        self.old = old
        self.new = new
        self.count = count
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + old.size + new.size + count.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + '.replace(' + self.old.toString() + ", " + self.new.toString() + "," + self.count.toString() + ")"

    def interpret(self, env):
        super_str = self.str.interpret(env)
        occurence_count = self.count.interpret(env)
        old_str = self.old.interpret(env)
        new_str = self.new.interpret(env)
        if old_str == "" or occurence_count <= 0:
            return None

        index = -1
        for _ in range(occurence_count):
            index += 1
            index = super_str.find(old_str, index)
            if index == -1:
                break

        if index == -1:
            return None

        return super_str[:index] + new_str + super_str[index + len(old_str):]

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)
        self.old.getProgramIds(program_ids)
        self.new.getProgramIds(program_ids)
        self.count.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 3), repeat=4))
        for combination in combinations:
            layer1, layer2, layer3, layer4 = combination
            if layer1 + layer2 + layer3 + layer4 + BustlePCFG.get_instance().get_cost_by_name(
                    StrReplaceOccurence.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])
            layer3_prog = plist.get_programs(layer3, STR_TYPES['type'])
            layer4_prog = plist.get_programs(layer4, INT_TYPES['type'])
            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral):
                    continue
                for prog2 in layer2_prog:
                    p2_str = prog2.toString()
                    if p2_str == EMPTY_STRING:
                        continue
                    is_p2_var = isinstance(prog2, StrVar)
                    if is_p2_var:
                        continue
                    for prog3 in layer3_prog:
                        if prog3.toString() == p2_str:
                            continue
                        for prog4 in layer4_prog:
                            yield StrReplaceOccurence(prog1, prog2, prog3, prog4)


class StrTrim(Str):
    ARITY = 1

    def __init__(self, input_str):
        self.str = input_str
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + ".strip()"

    def interpret(self, env):
        return self.str.interpret(env).strip()

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = range(1, size)
        for combination in combinations:
            layer1 = combination
            if layer1 + BustlePCFG.get_instance().get_cost_by_name(StrTrim.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])

            for prog1 in layer1_prog:
                if isinstance(prog1, StrTrim):
                    continue
                yield StrTrim(prog1)


class StrLower(Str):
    ARITY = 1

    def __init__(self, input_str):
        self.str = input_str
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + ".lower()"

    def interpret(self, env):
        return self.str.interpret(env).lower()

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = range(1, size)
        for combination in combinations:
            layer1 = combination
            if layer1 + BustlePCFG.get_instance().get_cost_by_name(StrLower.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])

            for prog1 in layer1_prog:
                if isinstance(prog1, StrLower):
                    continue
                yield StrLower(prog1)


class StrUpper(Str):
    ARITY = 1

    def __init__(self, input_str):
        self.str = input_str
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + ".upper()"

    def interpret(self, env):
        return self.str.interpret(env).upper()

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = range(1, size)
        for combination in combinations:
            layer1 = combination
            if layer1 + BustlePCFG.get_instance().get_cost_by_name(StrUpper.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])

            for prog1 in layer1_prog:
                if isinstance(prog1, StrUpper):
                    continue
                yield StrUpper(prog1)


class StrProper(Str):
    ARITY = 1

    def __init__(self, input_str):
        self.str = input_str
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + ".title()"

    def interpret(self, env):
        return self.str.interpret(env).title()

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = range(1, size)
        for combination in combinations:
            layer1 = combination
            if layer1 + BustlePCFG.get_instance().get_cost_by_name(StrProper.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])

            for prog1 in layer1_prog:
                if isinstance(prog1, StrProper):
                    continue
                yield StrProper(prog1)


class StrRepeat(Str):
    ARITY = 2

    def __init__(self, input_str, input_int):
        self.str = input_str
        self.int = input_int
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_int.size + input_str.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + "*" + self.int.toString()

    def interpret(self, env):
        try:
            string_element = self.str.interpret(env)
            integer_element = self.int.interpret(env)
            if len(string_element) > 100 or integer_element > 100 or integer_element <= 0:
                return None
            if len(string_element) * integer_element > 100:
                return None
            return self.str.interpret(env) * self.int.interpret(env)
        except:
            pass
        return None

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)
        self.int.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(StrRepeat.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral) and prog1.toString() == EMPTY_STRING:
                    continue
                for prog2 in layer2_prog:
                    yield StrRepeat(prog1, prog2)


class StrLeftSubstr(Str):
    ARITY = 2

    def __init__(self, input_str, input_int):
        self.str = input_str
        self.int = input_int
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_int.size + input_str.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + "[:" + self.int.toString() + "]"

    def interpret(self, env):
        return self.str.interpret(env)[:self.int.interpret(env)]

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)
        self.int.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(StrLeftSubstr.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral) and prog1.toString() == EMPTY_STRING:
                    continue
                for prog2 in layer2_prog:
                    yield StrLeftSubstr(prog1, prog2)


class StrRightSubstr(Str):
    ARITY = 2

    def __init__(self, input_str, input_int):
        self.str = input_str
        self.int = input_int
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_int.size + input_str.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + "[ length-" + self.int.toString() + ":]"

    def interpret(self, env):
        super_str = self.str.interpret(env)
        number_of_chars = self.int.interpret(env)
        if number_of_chars <= 0:
            return None
        return super_str[max(0, len(super_str) - number_of_chars):]

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)
        self.int.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(StrRightSubstr.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral) and prog1.toString() == EMPTY_STRING:
                    continue
                for prog2 in layer2_prog:
                    yield StrRightSubstr(prog1, prog2)


# end bustle additional classes

class StrCharAt(Str):
    ARITY = 2

    def __init__(self, input_str, pos):
        self.str = input_str
        self.pos = pos
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + pos.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + ".CharAt(" + self.pos.toString() + ")"

    def interpret(self, env):
        index = self.pos.interpret(env)
        string_element = self.str.interpret(env)
        if 0 <= index < len(string_element):
            return string_element[index]
        return None

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)
        self.pos.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(StrCharAt.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral) and prog1.toString() == EMPTY_STRING:
                    continue
                for prog2 in layer2_prog:
                    yield StrCharAt(prog1, prog2)


# String type and classes
STR_TYPES = {'type': 'str', 'classes': (StrLiteral, StrVar, StrConcat, StrReplace,
                                        StrSubstr, StrIte, StrIntToStr, StrCharAt, StrTrim, StrLower, StrUpper,
                                        StrProper, StrLeftSubstr, StrRightSubstr,
                                        StrReplaceOccurence, StrReplaceAdd, StrRepeat)}


# Contains all operations with return type int

class Int:
    def __init__(self):
        self.size = 0

    def getReturnType(self):
        return INT_TYPES['type']

    @classmethod
    def name(cls):
        return cls.__name__


class IntLiteral(Int):
    def __init__(self, value):
        self.value = value
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return str(self.value)

    def interpret(self, env):
        return self.value

    def getProgramIds(self, programIds):
        pass


class IntVar(Int):
    def __init__(self, name):
        self.value = name
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.value

    def interpret(self, env):
        return copy.deepcopy(env[self.value])

    def getProgramIds(self, programIds):
        pass


class IntStrToInt(Int):
    ARITY = 1

    def __init__(self, input_str):
        self.str = input_str
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + ".StrToInt()"

    def interpret(self, env):
        value = self.str.interpret(env)
        if regex_only_digits.search(value) is not None:
            return int(value)
        return None

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.str.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = range(1, size)
        for combination in combinations:
            layer1 = combination
            if layer1 + BustlePCFG.get_instance().get_cost_by_name(IntStrToInt.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])

            for prog1 in layer1_prog:
                yield IntStrToInt(prog1)


class IntPlus(Int):
    ARITY = 2

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = left.size + right.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return "(" + self.left.toString() + " + " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) + self.right.interpret(env)

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.left.getProgramIds(programIds)
        self.right.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(IntPlus.name()) != size:
                continue

            layer1_prog = plist.get_programs(layer1, INT_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])

            for prog1 in layer1_prog:
                for prog2 in layer2_prog:
                    yield IntPlus(prog1, prog2)


class IntMinus(Int):
    ARITY = 2

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = left.size + right.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return "(" + self.left.toString() + " - " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) - self.right.interpret(env)

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.left.getProgramIds(programIds)
        self.right.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(IntMinus.name()) != size:
                continue

            layer1_prog = plist.get_programs(layer1, INT_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])

            for prog1 in layer1_prog:
                for prog2 in layer2_prog:
                    yield IntMinus(prog1, prog2)


class IntLength(Int):
    ARITY = 1

    def __init__(self, input_str):
        self.str = input_str
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + ".Length()"

    def interpret(self, env):
        return len(self.str.interpret(env))

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.str.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = range(1, size)
        for combination in combinations:
            layer1 = combination
            if layer1 + BustlePCFG.get_instance().get_cost_by_name(IntLength.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])

            for prog1 in layer1_prog:
                yield IntLength(prog1)


class IntIteInt(Int):
    ARITY = 3

    def __init__(self, condition, true_case, false_case):
        self.condition = condition
        self.true_case = true_case
        self.false_case = false_case
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = condition.size + true_case.size + false_case.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return "(if" + self.condition.toString() + " then " + self.true_case.toString() + " else " + self.false_case.toString() + ")"

    def interpret(self, env):
        if self.condition.interpret(env):
            return self.true_case.interpret(env)
        else:
            return self.false_case.interpret(env)

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.condition.getProgramIds(programIds)
        self.true_case.getProgramIds(programIds)
        self.false_case.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 2), repeat=3))
        for combination in combinations:
            layer1, layer2, layer3 = combination
            if layer1 + layer2 + layer3 + BustlePCFG.get_instance().get_cost_by_name(IntMinus.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, BOOL_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
            layer3_prog = plist.get_programs(layer3, INT_TYPES['type'])

            for prog1 in layer1_prog:
                for prog2 in layer2_prog:
                    for prog3 in layer3_prog:
                        yield IntIteInt(prog1, prog2, prog3)


class IntIndexOf(Int):
    ARITY = 3

    def __init__(self, input_str, substr, start):
        self.input_str = input_str
        self.substr = substr
        self.start = start
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + substr.size + start.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.input_str.toString() + ".IndexOf(" + self.substr.toString() + "," + self.start.toString() + ")"

    def interpret(self, env):
        index = None
        try:
            start_position = self.start.interpret(env)
            sub_string = self.substr.interpret(env)
            super_string = self.input_str.interpret(env)
            if start_position <= 0 or start_position >= len(super_string):
                return None
            index = super_string.index(sub_string, start_position)
        except ValueError as ve:
            pass
        return index

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.input_str.getProgramIds(programIds)
        self.substr.getProgramIds(programIds)
        self.start.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 2), repeat=3))
        for combination in combinations:
            layer1, layer2, layer3 = combination
            if layer1 + layer2 + layer3 + BustlePCFG.get_instance().get_cost_by_name(IntMinus.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])
            layer3_prog = plist.get_programs(layer3, INT_TYPES['type'])

            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral) and prog1.toString() == EMPTY_STRING:
                    continue
                for prog2 in layer2_prog:
                    if isinstance(prog2, StrLiteral) and prog2.toString() == EMPTY_STRING:
                        continue
                    for prog3 in layer3_prog:
                        yield IntIndexOf(prog1, prog2, prog3)


# bustle additional integer classes
class IntFirstIndexOf(Int):
    ARITY = 2

    def __init__(self, input_str, substr):
        self.input_str = input_str
        self.substr = substr
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + substr.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.input_str.toString() + ".IndexOf(" + self.substr.toString() + ")"

    def interpret(self, env):
        sub_string = self.substr.interpret(env)
        super_string = self.input_str.interpret(env)
        index = None
        try:
            index = super_string.index(sub_string)
        except ValueError as ve:
            pass
        return index

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.input_str.getProgramIds(programIds)
        self.substr.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(IntFirstIndexOf.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])

            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral) and prog1.toString() == EMPTY_STRING:
                    continue
                for prog2 in layer2_prog:
                    if isinstance(prog2, StrLiteral) and prog2.toString() == EMPTY_STRING:
                        continue
                    yield IntFirstIndexOf(prog1, prog2)


# Integer type and classes
INT_TYPES = {'type': 'integer', 'classes': (IntLiteral, IntVar, IntStrToInt, IntPlus,
                                            IntMinus, IntLength, IntIteInt, IntIndexOf, IntFirstIndexOf)}


# Contains all operations with return type bool

class Bool:
    def __init__(self):
        self.size = 0

    def getReturnType(self):
        return BOOL_TYPES['type']

    @classmethod
    def name(cls):
        return cls.__name__


class BoolLiteral(Bool):
    def __init__(self, boolean):
        self.bool = True if boolean is True else False
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return str(self.bool)

    def interpret(self, env):
        return self.bool

    def getProgramIds(self, programIds):
        pass


class BoolEqual(Bool):
    ARITY = 2

    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = left.size + right.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return "Equal(" + self.left.toString() + "," + self.right.toString() + ")"

    def interpret(self, env):
        return True if self.left.interpret(env) == self.right.interpret(env) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.left.getProgramIds(programIds)
        self.right.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(BoolEqual.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])
            for prog1 in layer1_prog:
                if isinstance(prog1, StrLiteral) and prog1.toString() == EMPTY_STRING:
                    continue
                for prog2 in layer2_prog:
                    if isinstance(prog2, StrLiteral) and prog2.toString() == EMPTY_STRING:
                        continue
                    yield BoolEqual(prog1, prog2)


class BoolContain(Bool):
    ARITY = 2

    def __init__(self, input_str, substr):
        self.str = input_str
        self.substr = substr
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + substr.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.str.toString() + ".Contain(" + self.substr.toString() + ")"

    def interpret(self, env):
        return True if self.substr.interpret(env) in self.str.interpret(env) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.str.getProgramIds(programIds)
        self.substr.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(IntMinus.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])
            for prog1 in layer1_prog:
                for prog2 in layer2_prog:
                    yield BoolContain(prog1, prog2)


class BoolSuffixof(Bool):
    ARITY = 2

    def __init__(self, input_str, suffix):
        self.str = input_str
        self.suffix = suffix
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + suffix.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.suffix.toString() + ".SuffixOf(" + self.str.toString() + ")"

    def interpret(self, env):
        return True if self.str.interpret(env).endswith(self.suffix.interpret(env)) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.str.getProgramIds(programIds)
        self.suffix.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(IntMinus.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])
            for prog1 in layer1_prog:
                for prog2 in layer2_prog:
                    yield BoolSuffixof(prog1, prog2)


class BoolPrefixof(Bool):
    ARITY = 2

    def __init__(self, input_str, prefix):
        self.str = input_str
        self.prefix = prefix
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = input_str.size + prefix.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.prefix.toString() + ".Prefixof(" + self.str.toString() + ")"

    def interpret(self, env):
        return True if self.str.interpret(env).startswith(self.prefix.interpret(env)) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.str.getProgramIds(programIds)
        self.prefix.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(IntMinus.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])
            for prog1 in layer1_prog:
                for prog2 in layer2_prog:
                    yield BoolPrefixof(prog1, prog2)


# bustle additional bool classes

class BoolGreaterThan(Bool):
    ARITY = 2

    def __init__(self, first_int, second_int):
        self.first_int = first_int
        self.second_int = second_int
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = first_int.size + second_int.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.first_int.toString() + " > " + self.second_int.toString()

    def interpret(self, env):
        return True if self.first_int.interpret(env) > self.second_int.interpret(env) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.first_int.getProgramIds(programIds)
        self.second_int.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(BoolGreaterThan.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, INT_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
            for prog1 in layer1_prog:
                for prog2 in layer2_prog:
                    if prog1.toString() == prog2.toString():
                        continue
                    yield BoolGreaterThan(prog1, prog2)


class BoolGreaterThanEqual(Bool):
    ARITY = 2

    def __init__(self, first_int, second_int):
        self.first_int = first_int
        self.second_int = second_int
        self.id = BustlePCFG.get_instance().get_program_id()
        self.size = first_int.size + second_int.size + BustlePCFG.get_instance().get_cost(self)

    def toString(self):
        return self.first_int.toString() + " >= " + self.second_int.toString()

    def interpret(self, env):
        return True if self.first_int.interpret(env) >= self.second_int.interpret(env) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.first_int.getProgramIds(programIds)
        self.second_int.getProgramIds(programIds)

    @staticmethod
    def grow(plist, size):
        combinations = list(itertools.product(range(1, size - 1), repeat=2))
        for combination in combinations:
            layer1, layer2 = combination
            if layer1 + layer2 + BustlePCFG.get_instance().get_cost_by_name(BoolGreaterThanEqual.name()) != size:
                continue
            layer1_prog = plist.get_programs(layer1, INT_TYPES['type'])
            layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
            for prog1 in layer1_prog:
                for prog2 in layer2_prog:
                    yield BoolGreaterThan(prog1, prog2)


# Boolean classes and terminals

BOOL_TYPES = {'type': 'boolean', 'classes': (BoolLiteral, BoolEqual, BoolContain,
                                             BoolSuffixof, BoolPrefixof, BoolGreaterThan, BoolGreaterThanEqual)}
TERMINALS = [StrLiteral, StrVar, IntLiteral, IntVar, BoolLiteral]
