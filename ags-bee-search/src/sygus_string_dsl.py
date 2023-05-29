import copy
import itertools

from cfg import BustlePCFG
from utils import *

# Try and except are added in each class to make sure that the code can run without the cost model with BUS.

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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return '\"' + self.value + '\"'

    def interpret(self, env):
        return self.value

    def getProgramIds(self, program_ids):
        pass


class StrVar(Str):
    def __init__(self, name):
        self.value = name
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = BustlePCFG.get_instance().get_cost(self)
        except:
            pass

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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = x.size + y.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return 'concat(' + self.x.toString() + ", " + self.y.toString() + ")"

    def interpret(self, env):
        return self.x.interpret(env) + self.y.interpret(env)

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.x.getProgramIds(program_ids)
        self.y.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, combination):
        # skip if the cost combination exceeds the limit
        layer1, layer2 = combination
        # retrive bank of programs with costs c[0] and c[1]
        layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
        layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])

        for prog1 in layer1_prog:
            if isinstance(prog1, StrLiteral) and prog1.toString() == EMPTY_STRING:
                continue
            for prog2 in layer2_prog:
                if isinstance(prog2, StrLiteral) and prog2.toString() == EMPTY_STRING:
                    continue
                program = StrConcat(prog1, prog2)
                yield program


class StrReplace(Str):
    ARITY = 3

    def __init__(self, input_str, old, new):
        self.str = input_str
        self.old = old
        self.new = new
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + old.size + new.size + \
                BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.str.toString() + '.replace(' + self.old.toString() + ", " + self.new.toString() + ")"

    def interpret(self, env):
        return self.str.interpret(env).replace(self.old.interpret(env), self.new.interpret(env), 1)

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)
        self.old.getProgramIds(program_ids)
        self.new.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2, layer3 = combination
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

    def __init__(self, input_str, start, end):
        self.str = input_str
        self.start = start
        self.end = end
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + start.size + end.size + \
                BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.str.toString() + ".Substr(" + self.start.toString() + "," + self.end.toString() + ")"

    def interpret(self, env):
        return self.str.interpret(env)[self.start.interpret(env): self.end.interpret(env)]

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)
        self.start.getProgramIds(program_ids)
        self.end.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2, layer3 = combination
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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = condition.size + true_case.size + \
                false_case.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

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
    def grow(plist, combination):
        layer1, layer2, layer3 = combination
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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_int.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.int.toString() + ".IntToStr()"

    def interpret(self, env):
        return str(self.int.interpret(env))

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.int.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, combination):
        layer1 = combination[0]

        layer1_prog = plist.get_programs(layer1, INT_TYPES['type'])

        for prog1 in layer1_prog:
            yield StrIntToStr(prog1)


class StrLower(Str):
    ARITY = 1

    def __init__(self, input_str):
        self.str = input_str
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.str.toString() + ".lower()"

    def interpret(self, env):
        return self.str.interpret(env).lower()

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, combination):
        layer1 = combination[0]
        layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])

        for prog1 in layer1_prog:
            if isinstance(prog1, StrLower):
                continue
            yield StrLower(prog1)


class StrUpper(Str):
    ARITY = 1

    def __init__(self, input_str):
        self.str = input_str
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.str.toString() + ".upper()"

    def interpret(self, env):
        return self.str.interpret(env).upper()

    def getProgramIds(self, program_ids):
        program_ids.add(self)
        self.str.getProgramIds(program_ids)

    @staticmethod
    def grow(plist, combination):
        layer1 = combination[0]
        layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])

        for prog1 in layer1_prog:
            if isinstance(prog1, StrUpper):
                continue
            yield StrUpper(prog1)


class StrCharAt(Str):
    ARITY = 2

    def __init__(self, input_str, pos):
        self.str = input_str
        self.pos = pos
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + pos.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

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
    def grow(plist, combination):
        layer1, layer2 = combination
        layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
        layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
        for prog1 in layer1_prog:
            if isinstance(prog1, StrLiteral) and prog1.toString() == EMPTY_STRING:
                continue
            for prog2 in layer2_prog:
                yield StrCharAt(prog1, prog2)


# String type and classes
STR_TYPES = {'type': 'str', 'classes': (StrLiteral, StrVar, StrConcat, StrReplace,
                                        StrSubstr, StrIte, StrIntToStr, StrCharAt, StrLower, StrUpper)}


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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return str(self.value)

    def interpret(self, env):
        return self.value

    def getProgramIds(self, programIds):
        pass


class IntVar(Int):
    def __init__(self, name):
        self.value = name
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = BustlePCFG.get_instance().get_cost(self)
        except:
            pass

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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

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
    def grow(plist, combination):
        layer1 = combination[0]

        layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])

        for prog1 in layer1_prog:
            yield IntStrToInt(prog1)


class IntPlus(Int):
    ARITY = 2

    def __init__(self, left, right):
        self.left = left
        self.right = right
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = left.size + right.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return "(" + self.left.toString() + " + " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) + self.right.interpret(env)

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.left.getProgramIds(programIds)
        self.right.getProgramIds(programIds)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2 = combination

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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = left.size + right.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return "(" + self.left.toString() + " - " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) - self.right.interpret(env)

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.left.getProgramIds(programIds)
        self.right.getProgramIds(programIds)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2 = combination

        layer1_prog = plist.get_programs(layer1, INT_TYPES['type'])
        layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])

        for prog1 in layer1_prog:
            for prog2 in layer2_prog:
                yield IntMinus(prog1, prog2)


class IntMultiply(Int):
    ARITY = 2

    def __init__(self, left, right):
        self.left = left
        self.right = right
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = left.size + right.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return "(" + self.left.toString() + " * " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) * self.right.interpret(env)

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.left.getProgramIds(programIds)
        self.right.getProgramIds(programIds)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2 = combination

        layer1_prog = plist.get_programs(layer1, INT_TYPES['type'])
        layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])

        for prog1 in layer1_prog:
            for prog2 in layer2_prog:
                yield IntMultiply(prog1, prog2)


class IntModulo(Int):
    ARITY = 2

    def __init__(self, left, right):
        self.left = left
        self.right = right
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = left.size + right.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return "(" + self.left.toString() + " % " + self.right.toString() + ")"

    def interpret(self, env):
        try:
            return self.left.interpret(env) % self.right.interpret(env)
        except ZeroDivisionError:
            return None

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.left.getProgramIds(programIds)
        self.right.getProgramIds(programIds)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2 = combination

        layer1_prog = plist.get_programs(layer1, INT_TYPES['type'])
        layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])

        for prog1 in layer1_prog:
            for prog2 in layer2_prog:
                yield IntModulo(prog1, prog2)


class IntLength(Int):
    ARITY = 1

    def __init__(self, input_str):
        self.str = input_str
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.str.toString() + ".Length()"

    def interpret(self, env):
        return len(self.str.interpret(env))

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.str.getProgramIds(programIds)

    @staticmethod
    def grow(plist, combination):
        layer1 = combination[0]
        layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])

        for prog1 in layer1_prog:
            yield IntLength(prog1)


class IntIteInt(Int):
    ARITY = 3

    def __init__(self, condition, true_case, false_case):
        self.condition = condition
        self.true_case = true_case
        self.false_case = false_case
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = condition.size + true_case.size + \
                false_case.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

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
    def grow(plist, combination):
        layer1, layer2, layer3 = combination
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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + substr.size + \
                start.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.input_str.toString() + ".IndexOf(" + self.substr.toString() + "," + self.start.toString() + ")"

    def interpret(self, env):
        start_position = self.start.interpret(env)
        sub_string = self.substr.interpret(env)
        super_string = self.input_str.interpret(env)
        index = None
        try:
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
    def grow(plist, combination):
        layer1, layer2, layer3 = combination
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


# bustle additional integer classes (equivalent of intfind)
class IntFirstIndexOf(Int):
    ARITY = 2

    def __init__(self, input_str, substr):
        self.input_str = input_str
        self.substr = substr
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + substr.size + \
                BustlePCFG.get_instance().get_cost(self)
        except:
            pass

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
    def grow(plist, combination):
        layer1, layer2 = combination
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
INT_TYPES = {'type': 'integer', 'classes': (IntLiteral, IntVar, IntStrToInt, IntPlus, IntMultiply, IntModulo,
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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = BustlePCFG.get_instance().get_cost(self)
        except:
            pass

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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = left.size + right.size + BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return "Equal(" + self.left.toString() + "," + self.right.toString() + ")"

    def interpret(self, env):
        return True if self.left.interpret(env) == self.right.interpret(env) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.left.getProgramIds(programIds)
        self.right.getProgramIds(programIds)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2 = combination
        layer1_prog = plist.get_programs_all(layer1)
        layer2_prog = plist.get_programs_all(layer2)
        for prog1 in layer1_prog:
            for prog2 in layer2_prog:
                if (
                        (isinstance(prog1, STR_TYPES['classes']) and isinstance(prog2, INT_TYPES['classes'])) or
                        (isinstance(prog1, INT_TYPES['classes']) and isinstance(prog2, STR_TYPES['classes'])) or
                        (isinstance(prog1, BOOL_TYPES['classes']) and isinstance(prog2, STR_TYPES['classes'])) or
                        (isinstance(prog1, STR_TYPES['classes']) and isinstance(
                            prog2, BOOL_TYPES['classes']))
                ):
                    yield BoolEqual(IntLiteral(1), IntLiteral(2))
                else:
                    yield BoolEqual(prog1, prog2)


class BoolContain(Bool):
    ARITY = 2

    def __init__(self, input_str, substr):
        self.str = input_str
        self.substr = substr
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + substr.size + \
                BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.str.toString() + ".Contain(" + self.substr.toString() + ")"

    def interpret(self, env):
        return True if self.substr.interpret(env) in self.str.interpret(env) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.str.getProgramIds(programIds)
        self.substr.getProgramIds(programIds)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2 = combination
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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + suffix.size + \
                BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.suffix.toString() + ".SuffixOf(" + self.str.toString() + ")"

    def interpret(self, env):
        return True if self.str.interpret(env).endswith(self.suffix.interpret(env)) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.str.getProgramIds(programIds)
        self.suffix.getProgramIds(programIds)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2 = combination
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
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = input_str.size + prefix.size + \
                BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.prefix.toString() + ".Prefixof(" + self.str.toString() + ")"

    def interpret(self, env):
        return True if self.str.interpret(env).startswith(self.prefix.interpret(env)) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.str.getProgramIds(programIds)
        self.prefix.getProgramIds(programIds)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2 = combination
        layer1_prog = plist.get_programs(layer1, STR_TYPES['type'])
        layer2_prog = plist.get_programs(layer2, STR_TYPES['type'])
        for prog1 in layer1_prog:
            for prog2 in layer2_prog:
                yield BoolPrefixof(prog1, prog2)


class BoolGreaterThan(Bool):
    ARITY = 2

    def __init__(self, first_int, second_int):
        self.first_int = first_int
        self.second_int = second_int
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = first_int.size + second_int.size + \
                BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.first_int.toString() + " > " + self.second_int.toString()

    def interpret(self, env):
        return True if self.first_int.interpret(env) > self.second_int.interpret(env) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.first_int.getProgramIds(programIds)
        self.second_int.getProgramIds(programIds)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2 = combination
        layer1_prog = plist.get_programs(layer1, INT_TYPES['type'])
        layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
        for prog1 in layer1_prog:
            for prog2 in layer2_prog:
                if prog1.toString() == prog2.toString():
                    continue
                yield BoolGreaterThan(prog1, prog2)


class BoolLessThan(Bool):
    ARITY = 2

    def __init__(self, first_int, second_int):
        self.first_int = first_int
        self.second_int = second_int
        try:
            self.id = BustlePCFG.get_instance().get_program_id()
            self.size = first_int.size + second_int.size + \
                BustlePCFG.get_instance().get_cost(self)
        except:
            pass

    def toString(self):
        return self.first_int.toString() + " < " + self.second_int.toString()

    def interpret(self, env):
        return True if self.first_int.interpret(env) < self.second_int.interpret(env) else False

    def getProgramIds(self, programIds):
        programIds.add(self)
        self.first_int.getProgramIds(programIds)
        self.second_int.getProgramIds(programIds)

    @staticmethod
    def grow(plist, combination):
        layer1, layer2 = combination
        layer1_prog = plist.get_programs(layer1, INT_TYPES['type'])
        layer2_prog = plist.get_programs(layer2, INT_TYPES['type'])
        for prog1 in layer1_prog:
            for prog2 in layer2_prog:
                if prog1.toString() == prog2.toString():
                    continue
                yield BoolGreaterThan(prog1, prog2)


# Boolean classes and terminals

BOOL_TYPES = {'type': 'boolean', 'classes': (BoolLiteral, BoolEqual, BoolContain,
                                             BoolSuffixof, BoolPrefixof, BoolGreaterThan, BoolGreaterThan)}
TERMINALS = [StrLiteral, StrVar, IntLiteral, IntVar, BoolLiteral]


NON_TERMINALS = [StrConcat, StrReplace, StrSubstr, StrIte, StrIntToStr, StrCharAt, StrLower, StrUpper, IntStrToInt,
                 IntPlus, IntMinus, IntLength, IntIteInt, IntIndexOf, IntFirstIndexOf, IntMultiply, IntModulo,
                 BoolEqual, BoolContain, BoolSuffixof, BoolPrefixof, BoolGreaterThan, BoolLessThan]
