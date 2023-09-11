"""
Definitions for the base C++ types

Enables the user of the module to create an instance of a class describing a concrete C++ type.
Definitions for the STL types are found in its dedicated module.
"""

from ..scope import Scope
import typing as t


class Qualifier:
    # TODO: complete the concept of a qualifier
    pass


class Const(Qualifier):
    pass


class TypeDef:
    """Base class for any type definition in C++
    """
    _cpp_decl: str

    def __init__(self, cpp_decl: str):
        self._cpp_decl = cpp_decl

    @property
    def cpp_decl(self) -> str:
        return self._cpp_decl


# In C++, "type definitions" might be NAMED or UNNAMED.
# Example of (possibly) NAMED type definitions:
# * struct, classes
# * enums and class enums
# * type aliases (`using` keyword)
# * All the STL templates
# Example of UNNAMED type definitions:
# * A primitive type, that is ASSIGNED directly to an element (a member attribute, a variable)
# * A C pointer, directly assigned as type of an identifier
# * A C array, identical to a C pointer
# ATM we are mixing in this type hierarchy the concepts of cpp_decl and name. This is WRONG.
# TODO: decide on a proper design for this
# NOTE: after a bit of thought, the best solution is obvious: add a name and a cpp_decl
#    Q: are they mutually exclusive?

class Void(TypeDef):
    """The none type in C/C++
    """

    def __init__(self):
        super().__init__('void')


class PrimitiveType(TypeDef):
    value: t.Any

    def __init__(self, value, cpp_decl: str):
        super().__init__(cpp_decl)
        self.value = value

    def __str__(self):
        return f'PrimitiveType({self.value})'


class Enumeration(TypeDef):
    _underlying_type: None | PrimitiveType
    _values: dict[str, t.Any]

    def __init__(self, cpp_decl: str):
        super().__init__(cpp_decl)
        self._values = {}
        self._underlying_type = None


class Reference(TypeDef):
    """A type definition that is a reference to another type definition

    A C++ alias or a typedef are represented with this type
    """
    _dest: TypeDef

    def __init__(self, name: str, dest: TypeDef):
        """

        :param name: We cannot use directly `dest.cpp_decl` as name, as the typedef has its own name (the name of the
        alias to the original type) that we want to keep :param dest:
        """
        super().__init__(name)
        self._dest = dest

    @property
    def dest(self):
        return self._dest


# https://stackoverflow.com/questions/71598358/extend-a-python-class-functionality-from-a-different-class
# https://stackoverflow.com/questions/12099237/dynamically-derive-a-class-in-python
class QualifiedType(Reference):
    """A type  with qualifiers

    A qualifier always applies to an EXISTING type. You cannot do this through multiple inheritance, as the "TypeDef"
    instance defining the un-qualified (original) type MUST exist as well. I.e.: you ALWYAS need a reference to an
    existing original type.
    """

    qualifiers: list[Qualifier]

    def __init__(self, dest: TypeDef):
        super().__init__(dest)
        self.qualifiers = []


class CompoundType(TypeDef):
    """As per C++ std, Compound types [basic.compound]
    """
    pass


class StructDeclaration:
    """A declaration of a struct, containing various fields (AKA members)

    In this code, we are NOT interested in methods of the struct. Hence, they are ignored
    """
    fields: dict[str, 'TypeDef']

    def __init__(self):
        self.fields = {}


class StructType(CompoundType, Scope):
    """A class or struct type in C++

    Note that a `StructType` is also a `Scope`. It can then also contain other "scopes". The nested scopes are
    handled through the "scope" graph.
    """

    _declaration: StructDeclaration

    def __init__(self, name: str):
        """
        .. todo::
           A struct type does NOT have a "cpp_decl". Re-evaluate this `name` mixup garbage. Structs in C++
           have a name (or not), but the full decl is the `StructDefinition` that we include as a member.

        :param name: The scope name
        """
        CompoundType.__init__(self, name)
        # ^^^ TODO: add the ClassType, and replace this hardcoded struct with class keyword
        Scope.__init__(self, name)
        self._declaration = StructDeclaration()

    @property
    def declaration(self) -> StructDeclaration:
        return self._declaration

    @declaration.setter
    def declaration(self, decl: StructDeclaration) -> None:
        self._declaration = decl


class GenericPointer(TypeDef):
    """A type representing any pointer (C or C++ smart_ptr)

    """
    # We use the same nomenclature as the STL. Check the shared_ptr
    _element_type: None | TypeDef

    def __init__(self):
        super().__init__("PTR...")
        # ^^ FOR THE MOMENT. When setting the pointed type, we will correct this wrong name.
        # TODO: should we add the concept that the "name" of the TypeDef instance might be unset?
        self._element_type = None

    @property
    def element_type(self) -> TypeDef:
        if not self._element_type:
            raise RuntimeError('element_type MUST be set before trying to get it')
        return self._element_type

    @element_type.setter
    def element_type(self, element_type: TypeDef):
        self._element_type = element_type
        self._cpp_decl = element_type.cpp_decl + '*'


class CPointer(GenericPointer):
    def __init__(self, *args):
        """
        We need the arbitrary arguments (non kw) because the generic struct processor expects a struct ctor that needs a name
        TODO: revisit this design :(
        """
        super().__init__()

    @GenericPointer.element_type.setter
    def element_type(self, element_type: TypeDef):
        self._element_type = element_type
        self._cpp_decl = element_type.cpp_decl + '*'