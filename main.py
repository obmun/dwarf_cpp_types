import elftools.dwarf.constants as dc
import logging
import numpy as np
import sys
import typing as t
from bidict import bidict
from pathlib import Path
from typing import Annotated
from annotated_types import Ge

from elftools.elf.elffile import ELFFile
from elftools.dwarf.die import DIE
from elftools.dwarf.die import AttributeValue

BASIC_INDENT = '    '
CLASS_OR_STRUCT_TYPE_TAGS = ['DW_TAG_structure_type', 'DW_TAG_class_type']

UInt = Annotated[int, Ge(0)]


class Scope:
    """Anything that can contain other elements
    """
    _name: str
    structs: dict[str, 'StructType']

    def __init__(self, name: str):
        self._name = name
        self.structs = {}

    @property
    def name(self):
        return self._name


class Namespace(Scope):
    """A scope that can only contain other struct definitions

    (and other nested scoped through the `ScopeGraph`)
    """

    def __init__(self, name: str):
        super().__init__(name)


class RootNs(Namespace):
    def __init__(self):
        super().__init__('')


class ScopeGraphNode:
    """The node of a `ScopeGraph`
    """
    _parent: 'ScopeGraphNode'
    _children: list['ScopeGraphNode']
    _depth: UInt
    scope: Scope

    def __init__(self, scope: Scope, parent: t.Optional['ScopeGraphNode'], depth: UInt):
        self._depth = depth
        self.scope = scope
        self._children = []
        self._parent = parent

    def add_child(self, scope: Scope) -> 'ScopeGraphNode':
        """

        :param scope:
        :return:
        """
        new_child = ScopeGraphNode(scope, self, self._depth + 1)
        self._children.append(new_child)
        return new_child

    @property
    def depth(self) -> UInt:
        return self._depth

    def get_path(self) -> list[str]:
        if not self._parent:
            return []
        return self._parent.get_path() + [self.scope.name]


class StructScopeGraphNode(ScopeGraphNode):
    scope: 'StructType'


class ScopeGraph:
    _root: ScopeGraphNode

    def __init__(self):
        self._root = ScopeGraphNode(RootNs(), None, 0)

    @property
    def root(self) -> ScopeGraphNode:
        return self._root


scope_to_die_map: bidict['ScopeGraphNode', DIE] = bidict()
scope_graph = ScopeGraph()


# ^^ TODO: move these 2 to a class


class StructDefinition():
    """A definition of a struct, containing various fields
    """
    fields: dict[str, 'TypeDef']

    def __init__(self):
        self.fields = {}


class DwarfRef:
    pass


class CURef(DwarfRef):
    """A reference within the same Containing Unit

    Identifies any debugging information entry within the containing unit. It is an offset from the first byte of the
    compilation header for the compilation unit containing the reference"""

    offset: int

    def __init__(self, offset: int):
        self.offset = offset


def extract_types(filename, types: t.Optional[list[str]] = None):
    """
    Description

    :param types: a list of types to recover. If not provided, all types will be extracted
    """

    print('Processing file:', filename)
    with open(filename, 'rb') as f:
        elffile = ELFFile(f)

        if not elffile.has_dwarf_info():
            print('  file has no DWARF info')
            return

        dwarf_info = elffile.get_dwarf_info()

        for cu in dwarf_info.iter_CUs():
            # Start with the top DIE, the root for this CU's DIE tree
            top_die = cu.get_top_DIE()
            print('    Top DIE with tag=%s' % top_die.tag)

            print('    name=%s' % Path(top_die.get_full_path()).as_posix())

            process_die_recursive(top_die, scope_graph.root)

    pass


class TypeDef:
    """Base class for any type definition
    """
    _cpp_decl: str

    def __init__(self, cpp_decl: str):
        self._cpp_decl = cpp_decl

    @property
    def cpp_decl(self) -> str:
        return self._cpp_decl


# TODO: complete the concept of a qualifier
class Qualifier:
    pass


class QualifiedType(TypeDef):
    """A TypeDef with qualifiers

    """

    qualifiers: list[Qualifier]


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

    """
    _dest: TypeDef

    def __init__(self, dest: TypeDef):
        super().__init__(dest.cpp_decl)
        self._dest = dest

    @property
    def dest(self):
        return self._dest


class CompoundType(TypeDef):
    """As per C++ std, Compound types [basic.compound]
    """
    pass


class StructType(CompoundType, Scope):
    """A class or struct type in C++

    Note that a `StructDefinition` is also a `Scope`. It can then also contain other "scopes". The nested scopes are handled through the "scope" graph.
    """

    definition: StructDefinition

    def __init__(self, definition: StructDefinition, name: str):
        """
        A struct type does NOT have a "cpp_decl". It's declaration is just simply the C++ struct declaration syntax

        :param definition:
        :param name: The scope name
        """
        CompoundType.__init__(self, f'struct {name}')
        # ^^^ TODO: add the ClassType, and replace this hardcoded struct with class keyword
        Scope.__init__(self, name)
        self.definition = definition


class Container(TypeDef):
    _value_type: TypeDef

    @property
    def value_type(self):
        return self._value_type


class SequenceContainer(Container):
    pass


class Vector(SequenceContainer, StructType):
    """Type definition representing a std::vector
    """
    pass


class Array(SequenceContainer):
    """Type definition representing a std::array
    """
    pass


class SmartPtr(TypeDef):
    _element_type: t.Any


class UniquePtr(SmartPtr):
    pass


class SharedPtr(TypeDef):
    pass


def dwarf_base_type_to_py_type(die: DIE) -> PrimitiveType:
    assert die.tag == 'DW_TAG_base_type'

    mapping = {
        (dc.DW_ATE_boolean, 1): np.bool_,
        (dc.DW_ATE_unsigned_char, 1): np.ubyte,  # According to NumPy doc (https://numpy.org/doc/stable/user/basics
        # .types.html), ubyte is equivalent to unsigned char. Review if this is the correct assingment
        (dc.DW_ATE_signed, 1): np.int8,
        (dc.DW_ATE_signed, 2): np.int16,
        (dc.DW_ATE_signed, 4): np.int32,
        (dc.DW_ATE_signed, 8): np.int64,
        (dc.DW_ATE_unsigned, 1): np.uint8,
        (dc.DW_ATE_unsigned, 2): np.uint16,
        (dc.DW_ATE_unsigned, 4): np.uint32,
        (dc.DW_ATE_unsigned, 8): np.uint64,
        (dc.DW_ATE_float, 4): np.float32,
        (dc.DW_ATE_float, 8): np.float64,
        (dc.DW_ATE_signed_char, 1): np.byte
    }

    # die.attributes["DW_AT_encoding"].value
    # Gives us the "type" of the base type. E.g.: bool, float, complex, signed int, ... DWARF5 Table 7.11
    # die.attributes['DW_AT_name']
    # Equivalent to the encoding + byte_size, but in textual form.

    return PrimitiveType(mapping[(die.attributes["DW_AT_encoding"].value, die.attributes['DW_AT_byte_size'].value)],
                         die.attributes['DW_AT_name'].value.decode('utf-8'))


def process_type_die(die: DIE, scope: t.Optional[ScopeGraphNode], dependency: bool = False) -> t.Optional[TypeDef]:
    """

    :param die:
    :param scope: If None, we will use the PARENT die to find the right SCOPE

    :param dependency: Indicates whether this type is a dependency of other type (e.g., the type of a member of a
    struct). If True, this type will be imported even if it is not in the whitelist, as some type in its dependees
    have been whitelisted.

    :return:
    """
    parent_die = None
    if not scope:
        parent_die = die.get_parent()

    # TODO: here we will always be doing FIRST TIME processing of a type, right? We are processing the DIE type ...
    # How are we gonna handle references to existing types? ATM we only handle Refs to structs... But I guess we will have refs to EVERYTHING

    match die.tag:
        case 'DW_TAG_base_type':
            return dwarf_base_type_to_py_type(die)
        case 'DW_TAG_pointer_type':
            logging.error('cannot handle ptr types yet');
            return None

        case 'DW_TAG_typedef':
            logging.error('typedef (alias) support not implemented yet');
            return None

        case 'DW_TAG_array_type':
            # !!! Array DIEs normally have children!
            # raise NotImplementedError("No array support ...")
            return None

        case 'DW_TAG_enumeration_type':
            return process_enum_type(die, scope, dependency)

        case 'DW_TAG_const_type':
            # A type qualifier is handled as a const_type tag with a NESTED DW_AT_type attribute! (normally a reference)
            # TODO: implement handling and qualifiers if our Type structures that require it!
            return None

        case tag if tag in CLASS_OR_STRUCT_TYPE_TAGS:
            struct_graph_node = process_struct_or_class_type(die, scope, dependency)
            struct_type = struct_graph_node.scope
            return Reference(struct_type)
        case other:
            raise RuntimeError('Do not know how to handle type die of type {}'.format(die.tag))


def process_type_ref(attr: AttributeValue, die: DIE, scope: ScopeGraphNode):
    """

    :param attr:
    :param die: The parent die
    :return:
    """
    if attr.form != 'DW_FORM_ref4':
        raise RuntimeError(f'Do not know now how to handle ref of form {attr.form}')

    ref_addr = attr.value + die.cu.cu_offset
    type_die = die.dwarfinfo.get_DIE_from_refaddr(ref_addr, die.cu)

    parent_die = type_die.get_parent()
    assert scope in scope_to_die_map
    given_scope_die = scope_to_die_map[scope]
    if given_scope_die != parent_die:
        # This is a reference to a type in a different scope.
        # 1) We need to parent the referenced type to its CORRECT scope
        # 2) It might be the case that we have NOT processed yet the parent scope

        if parent_die.tag == 'DW_TAG_compile_unit':
            # If the real parent is just the CU, this is a non-namespaced type (root) or a base type
            # Just use the ROOT ns as parent
            scope = scope_graph.root
        else:
            if parent_die not in scope_to_die_map.inverse:
                # Not processed before! Time to do it now!
                raise NotImplementedError()
            else:
                scope = scope_to_die_map.inverse[parent_die]

    return process_type_die(type_die, scope, True)


def process_type_attr(die: DIE, scope: ScopeGraphNode):
    """Process the attribute that defines the type of this debug info entity

    :param die:
    :param scope: The scope (parent struct) of the member field whose type we are processing. If type is a structure, we need to nest it
    :return:
    """
    type_attr = die.attributes['DW_AT_type']
    match type_attr.form:
        case 'DW_FORM_ref4':
            return process_type_ref(type_attr, die, scope)
        case other:
            raise RuntimeError('Do not know now how to handle type attr of form {}'.format(type_attr.form))


def process_member_die(die: DIE, parent_struct_node: StructScopeGraphNode) -> None:
    assert die.tag == 'DW_TAG_member'

    parent_struct_type = parent_struct_node.scope
    parent_struct_def = parent_struct_type.definition
    indent = BASIC_INDENT * parent_struct_node.depth

    field_name = die.attributes['DW_AT_name'].value.decode('utf-8')
    type_def = process_type_attr(die, parent_struct_node)
    if not type_def:
        logging.error(
            "Could not process type for member '{}::{}'. Definition will be incomplete".format(parent_struct_type.name,
                                                                                               field_name))

    print(indent + 'DIE tag=%s, attrs=' % die.tag)
    attr_indent = indent + BASIC_INDENT

    assert field_name not in parent_struct_def.fields
    parent_struct_def.fields[field_name] = type_def


def process_enum_type(die: DIE, scope_node: ScopeGraphNode, dependency: bool = False) -> None | Enumeration:
    """

    :param die: This type DIE must NOT have been processed before
    :param scope_node:
    :param dependency:
    :return:
    """
    assert die.tag == 'DW_TAG_enumeration_type'

    if 'DW_AT_name' not in die.attributes:
        logging.info("Ignoring unnamed DIE @{}".format(die.offset))
        return None

    name = die.attributes['DW_AT_name'].value.decode("utf-8")

    return Enumeration(name)


def vector_matcher(type_name: str) -> bool:
    return type_name.startswith('vector')


def vector_processor(die: DIE, parent_struct_node: StructScopeGraphNode):
    # TODO: extract this. Many container will have a "nested" type. This code is shared
    nested_type_attr = die.attributes['DW_AT_sibling']
    assert nested_type_attr.form == 'DW_FORM_ref4'
    nested_type = process_type_ref(nested_type_attr, die, parent_struct_node)
    pass


# A list  containing the processors for special STL struct/class types
stl_classes = [
    (vector_matcher, vector_processor)
]


def process_struct_or_class_type(die: DIE, scope_node: ScopeGraphNode,
                                 dependency: bool = False) -> None | ScopeGraphNode:
    """
    A struct or class declaration CANNOT be processed twice. Make sure to detect this situation outside this function

    :param die:
    :param scope_node:

    :param dependency: Indicates whether this type is a dependency of other type (e.g., the type of a member of a
    struct). If True, this type will be imported even if it is not in the whitelist, as some type in its dependees
    have been whitelisted.

    :return: The scope graph node that corresponds to this new structure or class.
    """
    global types_whitelist

    assert die.tag in CLASS_OR_STRUCT_TYPE_TAGS
    indent = BASIC_INDENT * scope_node.depth

    if 'DW_AT_name' not in die.attributes:
        logging.info("Ignoring unnamed DIE @{}".format(die.offset))
        return
    name = die.attributes['DW_AT_name'].value.decode("utf-8")

    full_path = scope_node.get_path() + [name]
    if not dependency and (full_path not in types_whitelist):
        return

    # Generic struct processing
    struct_def = StructDefinition()
    struct_type = StructType(struct_def, name)
    struct_type_node = scope_node.add_child(struct_type)
    assert struct_type_node not in scope_to_die_map
    scope_to_die_map[struct_type_node] = die

    # The STL types are also structs (classes). We need to apply a matcher on the name
    for matcher, processor in stl_classes:
        if matcher(name):
            processor(die, struct_type_node)
            break

    # The member fields of a struct are provided as child DIEs
    for child_die in die.iter_children():
        if child_die.tag == 'DW_TAG_member':
            process_member_die(child_die, struct_type_node)

    scope_node.scope.structs[name] = struct_type

    return struct_type_node


def process_die_recursive(die, scope: ScopeGraphNode):
    """ A recursive function for showing information about a DIE and its
        children.
    """
    curr_scope_full_path = scope.get_path()

    indent = BASIC_INDENT * scope.depth
    print(indent + 'DIE tag=%s, attrs=' % die.tag)

    name = die.attributes.get('DW_AT_name', None)
    if name:
        name = name.value.decode('utf-8')

    ns_name = None
    match die.tag:
        case 'DW_TAG_namespace':
            assert len(curr_scope_full_path) == scope.depth

            ns_name = name
            new_ns = Namespace(ns_name)
            scope = scope.add_child(new_ns)
            assert scope not in scope_to_die_map
            scope_to_die_map[scope] = die

        case t if t in CLASS_OR_STRUCT_TYPE_TAGS:
            process_struct_or_class_type(die, scope)
            # for name, val in die.attributes.items():
            #     print(indent_level + '  %s = %s' % (name, val))

        case other:
            logging.debug("Ignoring DIE '{}' of type '{}'".format(name, die.tag))
            pass

    for child in die.iter_children():
        process_die_recursive(child, scope)


types_whitelist: list[list[str]]

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print('Expected usage: {0} <executable>'.format(sys.argv[0]))
        sys.exit(1)

    # Associated types will be automatically brought in
    types_whitelist = [
        ['TopStruct'],
    ]
    extract_types(sys.argv[1])
