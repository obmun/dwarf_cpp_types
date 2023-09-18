"""
Definitions for the base C++ types

Enables the user of the module to create an instance of a class describing a concrete C++ type.
Definitions for the STL types are found in its dedicated module.
"""

from enum import Enum
import numpy as np
import typing as t
from .._helpers import _Base
from ..scope import Scope


class Qualifier:
    # TODO: complete the concept of a qualifier
    pass


class Const(Qualifier):
    pass


class TypeDef(_Base):
    """Base class for any type definition in C++

    `name` and `decl` properties
    ----------------------------

    In C++, "type definitions" might be NAMED or UNNAMED.
    Example of (possibly) NAMED type definitions:

    - struct, classes
    - enums and class enums
    - type aliases (`using` keyword)
    - All the STL templates

    Example of UNNAMED type definitions:

    - A primitive type, that is assigned directly to an element (a member attribute, a variable)
    - A C pointer, directly assigned as type of an identifier
    - A C array, identical to a C pointer

    TODO: Q: are `name` and `decl` properties mutually exclusive?
    """
    _name: None | str

    def __init__(self, name: (None | str) = None):
        # Needed for MRO chain in multi-parents
        super().__init__(name)
        self._name = name
        if (name is not None) and not name:
            raise ValueError("Name can be None, but not an empty")
        pass

    @property
    def decl(self) -> t.Any:
        """The CPP declaration of this type.

        The format of this declaration depends on the concrete type. See derived classes.
        """
        raise NotImplementedError('Property not implemented')

    @property
    def name(self) -> None | str:
        return self._name

    def is_named(self) -> bool:
        return self.name is not None


class Void(TypeDef):
    """The "none" type in C/C++
    """

    def __init__(self):
        # This is an "unnamed" type
        super().__init__()

    @TypeDef.decl.getter
    def decl(self) -> str:
        return 'void'

    def __repr__(self) -> str:
        return self.decl

    def __str__(self) -> str:
        return self.__repr__()


class PrimitiveType(TypeDef):
    _decl: t.Any

    def __init__(self, decl: np.generic):
        super().__init__()
        self._decl = decl

    def __repr__(self):
        return f'PrimitiveType({self._decl})'

    def __str__(self):
        return self.__repr__()

    @TypeDef.decl.getter
    def decl(self):
        return self._decl


class Enumeration(TypeDef):
    """
    This is named type
    """
    _VALUES_T = dict[str, t.Any]
    underlying_type: None | PrimitiveType
    _values: _VALUES_T

    def __init__(self, name: str):
        super().__init__(name)
        self._values = {}
        self._underlying_type = None

    @TypeDef.decl.getter
    def decl(self) -> _VALUES_T:
        return self._values

    def add_value(self, identifier: 'str', value) -> None:
        self._values[identifier] = value

    def __repr__(self):
        return f'Enumeration({self.decl})'

    def __str__(self):
        return self.__repr__()


class CEnum(Enumeration):
    pass


class ClassEnum(Enumeration):
    pass


class TypeReference(TypeDef):
    """A type definition that is a reference to another type definition

    A C++ alias or a typedef are represented with this type. This is a named type.
    """
    RefType = Enum('RefType', ['ALIAS', 'TYPEDEF', 'UNK'])
    _dest: TypeDef
    _type: RefType

    def __init__(self, name: str, dest: TypeDef):
        """
        :param name: An alias or typedef is a named type in C++. We cannot use directly `dest.cpp_decl` as name.
        """
        super().__init__(name)
        self._type = self.RefType.UNK
        self._dest = dest

    @property
    def dest(self):
        return self._dest

    @TypeDef.decl.getter
    def decl(self) -> TypeDef:
        return self._dest

    @property
    def ref_type(self) -> RefType:
        """The type of reference: alias or typedef
        """
        return self._type

    @ref_type.setter
    def ref_type(self, v) -> None:
        self._type = v

    def __repr__(self):
        ref_type_str = None
        match self.ref_type:
            case self.RefType.ALIAS:
                ref_type_str = 'alias'
            case self.RefType.TYPEDEF:
                ref_type_str = 'typedef'
            case self.RefType.UNK:
                ref_type_str = '~UNK_ref~'

        return f'{ref_type_str} {self.name} (-> {self._dest.name})'

    def __str__(self):
        return self.__repr__()


# https://stackoverflow.com/questions/71598358/extend-a-python-class-functionality-from-a-different-class
# https://stackoverflow.com/questions/12099237/dynamically-derive-a-class-in-python
class QualifiedType(TypeDef):
    """A type with qualifiers

    A qualifier always applies to an EXISTING type. You cannot do this through multiple inheritance, as the "TypeDef"
    instance defining the un-qualified (original) type MUST exist as well. I.e.: you ALWAYS need a reference to an
    existing original type.
    """
    _dest: TypeDef
    qualifiers: list[Qualifier]

    def __init__(self, dest: TypeDef):
        super().__init__()
        self._dest = dest
        self.qualifiers = []

    @property
    def dest(self):
        return self._dest


class CompoundType(TypeDef):
    """As per C++ std, Compound types [basic.compound]
    """

    def __init__(self, name):
        super().__init__(name)


class StructDeclaration:
    """A declaration of a struct, containing various fields (AKA members)

    In this package, we are NOT interested in methods of the struct. Hence, they are ignored.
    """
    fields: dict[str, TypeDef]

    def __init__(self):
        self.fields = {}

    def __repr__(self):
        return str(list(self.fields.keys()))

    def __str__(self):
        return self.__repr__()


class StructType(CompoundType, Scope):
    """A class or struct type in C++

    Note that a `StructType` is also a `Scope`. It can then also contain other "scopes". The nested scopes are
    handled through the "scope" graph.
    """

    _declaration: StructDeclaration

    def __init__(self, name: str):
        """
        :param name: The name of this struct (also scope name)
        """
        super().__init__(name)

        self._declaration = StructDeclaration()

    @property
    def decl(self) -> StructDeclaration:
        return self._declaration

    @decl.setter
    def decl(self, decl: StructDeclaration) -> None:
        self._declaration = decl

    def __repr__(self):
        return f'Struct({self.decl})'

    def __str__(self):
        return self.__repr__()


class _GenericPointer(TypeDef):
    """A type representing any pointer (C or C++ smart_ptr)
    """
    # We use the same nomenclature as the STL. Check the shared_ptr
    _element_type: None | TypeDef

    def __init__(self):
        TypeDef.__init__(self)
        # With super() this fails for some of our hierarchies :(. Problem came from the UniquePtr class
        # TODO: review the type hierachy
        self._element_type = None

    @property
    def element_type(self) -> None | TypeDef:
        """The type of the pointer (i.e, the type pointed to)

        :return: the type definition, or None if it has not been set yet
        """
        return self._element_type

    @element_type.setter
    def element_type(self, element_type: TypeDef):
        self._element_type = element_type


class CPointer(_GenericPointer):
    def __init__(self):
        super().__init__()

    @TypeDef.decl.getter
    def decl(self) -> str:
        elem_type_decl = self.element_type
        if not elem_type_decl:
            elem_type_decl = "~UNK~"
        return f"({str(elem_type_decl)})*"

    def __repr__(self) -> str:
        return self.decl

    def __str__(self):
        return self.__repr__()
