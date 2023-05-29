# Copyright (C) 2021-2022, Ken Tjhia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations
from enum import Enum, IntEnum

from aenum import Enum as AEnum


class TwoDir(IntEnum):
    FORWARD = 0
    BACKWARD = 1


class FourDir(IntEnum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3


class Color(AEnum):

    _init_ = "value string"

    NEUTRAL = 0, "neutral"
    BLUE = 1, "b"
    RED = 2, "r"
    GREEN = 3, "g"
    CYAN = 4, "c"
    YELLOW = 5, "y"
    MAGENTA = 6, "m"
    BLACK = 7, "k"

    def __str__(self):
        return self.string

    @classmethod
    def str_values(cls):
        return [member.string for member in cls]
