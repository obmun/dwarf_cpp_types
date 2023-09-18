"""
TYpe definitions for the STL types
"""

from .defs import _GenericPointer, StructType, TypeDef


class Container(TypeDef):
    """Base type def for any STL container

    https://en.cppreference.com/w/cpp/named_req/Container
    """
    _value_type: TypeDef

    @property
    def value_type(self):
        return self._value_type

    @value_type.setter
    def value_type(self, value: TypeDef) -> None:
        self._value_type = value


class SequenceSize:
    value: None | int
    """
    Size can be None. For example: for a `DynamicSize`, it is value can only be discovered at RUN TIME
    """

    def __init__(self): pass


class ConstantSize(SequenceSize):
    def __init__(self, v: int):
        super().__init__()
        self.value = v


class RuntimeSize(SequenceSize):
    """A size with a value that can only be found at runtime"""
    def __init__(self): pass


class RangedSize(RuntimeSize):
    """A size that is bound between min and max values

    TODO: ATM there are no supported types with this type of size!
    """
    min: int
    max: int

    def __init__(self):
        super().__init__()
        self.min = self.max = None


class DynamicSize(RuntimeSize):
    def __init__(self):
        super().__init__()


class SequenceContainer(Container):
    # TODO: should we make this type inherit from a more general "sequence" type? The normal array is like this ALSO!
    _size: None | SequenceSize

    def __init__(self):
        self._size = None

    @property
    def size(self) -> None | SequenceSize:
        return self._size


class Vector(SequenceContainer, StructType):
    """Type definition representing a std::vector
    """
    def __init__(self, name: str):
        StructType.__init__(self, name)

    @SequenceContainer.size.setter
    def size(self):
        self._size = DynamicSize()


class Array(SequenceContainer, StructType):
    """Type definition representing a std::array
    """
    def __init__(self, name: str):
        StructType.__init__(self, name)

    @SequenceContainer.size.setter
    def size(self, value: int) -> None:
        self._size = ConstantSize(value)


class SmartPtr(_GenericPointer, StructType):
    """A pointer that is an STL smart_ptr

    We inherit from StructType because all C++ smart ptrs are also classes.
    """
    def __init__(self, name: str):
        _GenericPointer.__init__(self)
        StructType.__init__(self, name)


class UniquePtr(SmartPtr):
    def __init__(self, name: str):
        super().__init__(name)


class SharedPtr(SmartPtr):
    def __init__(self, name: str):
        super().__init__(name)


class String(StructType):
    """Type corresponding to a std::string

    """

    def __init__(self, name: str):
        super().__init__(name)
