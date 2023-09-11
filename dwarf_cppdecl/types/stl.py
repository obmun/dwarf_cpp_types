"""
TYpe definitions for the STL types
"""

from .defs import GenericPointer, StructType, TypeDef


class Container(TypeDef):
    _value_type: TypeDef

    @property
    def value_type(self):
        return self._value_type

    @value_type.setter
    def value_type(self, value: TypeDef) -> None:
        self._value_type = value


class SequenceSize:
    @property
    def value(self) -> int:
        pass


class ConstantSize(SequenceSize):
    pass


class FixedVarSize(SequenceSize):
    @property
    def min(self) -> None|int:
        pass

    @property
    def max(self) -> None|int:
        pass


class DynamicSize(SequenceSize):
    pass


class SequenceContainer(Container):
    # TODO: should we make this type inherit from a more general "sequence" type? The normal array is like this ALSO!
    @property
    def size(self) -> SequenceSize:
        pass


class Vector(SequenceContainer, StructType):
    """Type definition representing a std::vector
    """
    def __init__(self, name: str):
        StructType.__init__(self, name)


class Array(SequenceContainer, StructType):
    """Type definition representing a std::array
    """
    def __init__(self, name: str):
        StructType.__init__(self, name)


class SmartPtr(GenericPointer, StructType):
    """A pointer that is an STL smart_ptr

    We inherit from StructType because all C++ smart ptrs are also classes.
    """
    def __init__(self, name: str):
        GenericPointer.__init__(self)
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
