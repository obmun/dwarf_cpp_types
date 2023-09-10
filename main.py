from collections import namedtuple
from itertools import zip_longest

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
QUALIFIER_TAGS = ['DW_TAG_const_type']

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

    (and other nested scopes through the `ScopeGraph`)
    """
    blacklisted: bool

    def __init__(self, name: str):
        super().__init__(name)
        self._blacklisted = False


class RootNs(Namespace):
    def __init__(self):
        super().__init__('')


class ScopeTreeNode:
    """The node of a `ScopeGraph`
    """
    _parent: 'ScopeTreeNode'
    _children: list['ScopeTreeNode']
    _depth: UInt
    scope: Scope

    def __init__(self, scope: Scope, parent: t.Optional['ScopeTreeNode'], depth: UInt):
        self._depth = depth
        self.scope = scope
        self._children = []
        self._parent = parent

    def add_child(self, scope: Scope) -> 'ScopeTreeNode':
        """Adds a new child scope

        :param scope:
        :return:
        """
        new_child = ScopeTreeNode(scope, self, self._depth + 1)
        self._children.append(new_child)
        return new_child

    @property
    def depth(self) -> UInt:
        return self._depth

    def get_path(self) -> list[str]:
        if not self._parent:
            # I am the root NS. My path is the empty list
            return []
        return self._parent.get_path() + [self.scope.name]

    def build_child_scope_path(self, scope: Scope) -> list[str]:
        """Given a scope, builds the full path as if the namespace itself was a child of this scope-tree node

        :return:
        """
        return self.get_path() + [scope.name]


class StructScopeTreeNode(ScopeTreeNode):
    """A scope-graph node where the scope is a struct (and not a namespace)

    """
    scope: 'StructType'


class ScopeTree:
    """A type representing a tree of scopes

    """
    _root: ScopeTreeNode

    def __init__(self):
        self._root = ScopeTreeNode(RootNs(), None, 0)

    @property
    def root(self) -> ScopeTreeNode:
        return self._root


# These "globals" should be moved to a class (together with all the associated code)

scope_node_to_die_map: bidict['ScopeTreeNode', DIE] = bidict()
die_to_type_map: bidict[DIE, 'TypeDef'] = bidict()
# ^^ THIS IS FUCKED! I am holding the DIE instances in memory!!!
# TODO: I need a unique ID for each DIE. Maybe the offset?????
scope_graph = ScopeTree()


class StructDefinition:
    """A definition of a struct, containing various fields (AKA members)

    In this code, we are NOT interested in methods of the struct. Hence, they are ignored
    """
    fields: dict[str, 'TypeDef']

    def __init__(self):
        self.fields = {}


# class DwarfRef:
#     pass
#
#
# class CURef(DwarfRef):
#     """A reference within the same Containing Unit
#
#     Identifies any debugging information entry within the containing unit. It is an offset from the first byte of the
#     compilation header for the compilation unit containing the reference"""
#
#     offset: int
#
#     def __init__(self, offset: int):
#         self.offset = offset


def extract_types(filename, types: t.Optional[list[str]] = None):
    """
    Description

    :param types: a list of types to recover. If not provided, all types will be extracted
    """

    logging.info(f'Processing file {filename}')
    with open(filename, 'rb') as f:
        elf_file = ELFFile(f)

        if not elf_file.has_dwarf_info():
            logging.warning('File has no DWARF info')
            return

        dwarf_info = elf_file.get_dwarf_info()

        for cu_index, cu in enumerate(dwarf_info.iter_CUs()):
            if cu_index > 0:
                raise NotImplementedError("We do not support DWARF info containing multiple Compile Units")

            # Start with the top DIE, the root for this CU's DIE tree
            top_die = cu.get_top_DIE()
            logging.debug(f'Top DIE with tag={top_die.tag}, name={Path(top_die.get_full_path()).as_posix()}')

            maybe_recurse_scope_die(top_die, scope_graph.root)

    pass


# TODO: complete the concept of a qualifier
class Qualifier:
    pass


class Const(Qualifier):
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
    A qualifier always applies to an EXISTING type. You cannot do this through multiple inheritance, as the "TypeDef" instance defining the un-qualified (original) type MUST exist as well. I.e.: you ALWYAS need a reference to an existing original type.
    """

    qualifiers: list[Qualifier]

    def __init__(self, dest: TypeDef):
        super().__init__(dest)
        self.qualifiers = []


class CompoundType(TypeDef):
    """As per C++ std, Compound types [basic.compound]
    """
    pass


class StructType(CompoundType, Scope):
    """A class or struct type in C++

    Note that a `StructDefinition` is also a `Scope`. It can then also contain other "scopes". The nested scopes are handled through the "scope" graph.
    """

    _definition: StructDefinition

    def __init__(self, name: str):
        """
        A struct type does NOT have a "cpp_decl". It's declaration is just simply the C++ struct declaration syntax

        :param definition:
        :param name: The scope name
        """
        CompoundType.__init__(self, f'struct {name}')
        # ^^^ TODO: add the ClassType, and replace this hardcoded struct with class keyword
        Scope.__init__(self, name)
        self._definition = StructDefinition()

    @property
    def definition(self) -> StructDefinition:
        return self._definition

    @definition.setter
    def definition(self, defi: StructDefinition) -> None:
        self._definition = defi


class Container(TypeDef):
    _value_type: TypeDef

    @property
    def value_type(self):
        return self._value_type

    @value_type.setter
    def value_type(self, value: TypeDef) -> None:
        self._value_type = value


class SequenceContainer(Container):
    pass


class Vector(SequenceContainer, StructType):
    """Type definition representing a std::vector
    """

    def __init__(self, name: str):
        StructType.__init__(self, name)


class Array(SequenceContainer):
    """Type definition representing a std::array
    """
    pass


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


def process_type_die(die: DIE, scope: t.Optional[ScopeTreeNode], dependency: bool = False) -> t.Optional[TypeDef]:
    """Processes a DIE that contains the declaration of a type

    This is a helper method. Feed it any of the DWARF type declarations DIEs, and it will take care of execute the right processing method.

    :param die: The DIE to process
    :param scope: If None, we will use the PARENT die to find the right SCOPE. Otherwise, the scope that will be the parent of the new type instance

    :param dependency: Indicates whether this type is a dependency of other type (e.g., the type of a member of a
    struct). If True, this type will be imported even if it is not in the whitelist, as some type in its dependees
    have been whitelisted.

    :return: The type-definition from the interpretation of this DIE and children
    """
    parent_die = None
    if not scope:
        parent_die = die.get_parent()

    # TODO: here we will always be doing FIRST TIME processing of a type, right? We are processing the DIE type ...
    # How are we gonna handle references to existing types? ATM we only handle Refs to structs... But I guess we will have refs to EVERYTHING

    ret = None
    match die.tag:
        case 'DW_TAG_base_type':
            ret = dwarf_base_type_to_py_type(die)

        case 'DW_TAG_pointer_type':
            ret = process_c_pointer(die, scope)

        case 'DW_TAG_typedef':
            ret = process_typedef(die, scope)

        case 'DW_TAG_array_type':
            # !!! Array DIEs normally have children!
            # raise NotImplementedError("No array support ...")
            pass

        case 'DW_TAG_enumeration_type':
            ret = process_enum_type(die, scope, dependency)

        case tag if tag in QUALIFIER_TAGS:
            # A type qualifier is a DIE with a relevant tag and a DW_AT_type attr of "ref" form
            # E.g.:
            # - const: DIE with tag==const_type with DW_AT_type-attribute of form==reference
            ret = recurse_extracting_qualifiers(die, scope)

        case tag if tag in CLASS_OR_STRUCT_TYPE_TAGS:
            struct_graph_node = maybe_process_struct_or_class_type(die, scope, dependency)
            struct_type = struct_graph_node.scope
            ret = struct_type
            # ret = Reference(struct_type)
            # TODO: why did I do this? The struct type def in Python is already a reference to the object ...

        case other:
            raise RuntimeError('Do not know how to handle type die of type {}'.format(die.tag))

    die_to_type_map[die] = ret
    return ret


def process_c_pointer(die: DIE, scope: ScopeTreeNode) -> CPointer:
    assert die.tag == 'DW_TAG_pointer_type'

    # A C pointer DIE either:
    # - Contains a DW_AT_type attr, for pointers pointer to a concrete type
    # - Does NOT CONTAIN 'DW_AT_type' attr in case of void* pointer
    if 'DW_AT_type' not in die.attributes:
        # This is a void ptr
        referenced_type = Void()
    else:
        referenced_type = process_type_attr(die, scope)

    ret = CPointer()
    ret.element_type = referenced_type
    return ret


def process_typedef(die: DIE, scope: ScopeTreeNode) -> Reference:
    assert die.tag == 'DW_TAG_typedef'
    # A typedef DIE just has a 'DW_AT_type' attr that we need to process, referencing another typoe.
    referenced_type = process_type_attr(die, scope)
    # However, as we have a bidir map between DIEs and created types, the typedef must be created as well as its own
    # "unique" type, and reference the real underyling type. Just exactly as DWARF does it
    ret = Reference(die.attributes['DW_AT_name'].value.decode(), referenced_type)
    return ret


def process_type_ref(attr: AttributeValue, die: DIE, scope: ScopeTreeNode) -> t.Optional[TypeDef]:
    """Follows a type reference, reaching the underlying type definition and evaluating it

    :param attr: The DW_AT_type attr to process
    :param die: The DIE containing this attribute (needed because the attr does not keep reference to its holding DIE)
    :return:
    """
    if attr.form != 'DW_FORM_ref4':
        raise RuntimeError(f'Do not know now how to handle ref of form {attr.form}')

    ref_addr = attr.value + die.cu.cu_offset
    type_die = die.dwarfinfo.get_DIE_from_refaddr(ref_addr, die.cu)

    # As this is a type reference, we need to check if we processed the DIE already
    existing_type = die_to_type_map.get(type_die)
    if existing_type:
        return existing_type

    # New type. Process it

    parent_die = type_die.get_parent()
    assert scope in scope_node_to_die_map
    caller_provided_scope_die = scope_node_to_die_map[scope]
    if caller_provided_scope_die != parent_die:
        # This is a reference to a type in a different scope.
        # 1) We need to parent the referenced type to its CORRECT scope
        # 2) It might be the case that we have NOT processed yet the parent scope

        if parent_die.tag == 'DW_TAG_compile_unit':
            # If the real parent is just the CU, this is a non-namespaced type (root) or a base type
            # Just use the ROOT ns as parent
            scope = scope_graph.root

            # DWARF notes:
            # * When the compiler (at least GCC) creates DWARF info for C-pointer declarations, these are sometimes directly parented to the CU
            #   - It seems that these basic declarations are NOT parented  inside the containing struct whose member uses this basic pointer type
            #     Probably for dedup?
            #   - E.g.: for a member of a struct of type `int16 *ptr;` this has been the case.
        else:
            if parent_die not in scope_node_to_die_map.inverse:
                # Not processed before! Time to do it now!

                # ADD a filter to blackslit:
                # 1. Types that start with __
                # 2. std:: types by default, except certain accepted types (the ones from the stl_processors)
                raise NotImplementedError()
            else:
                scope = scope_node_to_die_map.inverse[parent_die]

    return process_type_die(type_die, scope, True)


def process_type_attr(die: DIE, scope: ScopeTreeNode):
    """Process the attribute that defines the type of this Debug Info Entry

    :param die:
    :param scope: The scope (parent struct or namespace) of the "declaration" whose type we are processing. If this type ends up being a structure (another scope), we need to nest it inside the parent scope
    :return:
    """
    type_attr = die.attributes['DW_AT_type']
    match type_attr.form:
        case 'DW_FORM_ref4':
            return process_type_ref(type_attr, die, scope)
        case other:
            raise RuntimeError('Do not know now how to handle type attr of form {}'.format(type_attr.form))


def process_member_die(die: DIE, parent_struct_node: StructScopeTreeNode) -> None:
    """Processes a tag==member child DIE to add the member description to the parent struct

    :param die:
    :param parent_struct_node:
    :return:
    """
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

    attr_indent = indent + BASIC_INDENT

    assert field_name not in parent_struct_def.fields
    parent_struct_def.fields[field_name] = type_def


def recurse_extracting_qualifiers(die: DIE, scope_node: ScopeTreeNode) -> None | QualifiedType:
    tag_to_qualifier_type = {
        'DW_TAG_const_type': Const
    }
    qualifier = tag_to_qualifier_type[die.tag]
    qualified_type_attr = die.attributes['DW_AT_type']
    if qualified_type_attr.form != 'DW_FORM_ref4':
        raise NotImplementedError(f'Do not know how to handle DW_AT_type attr of form {qualified_type_attr.form}')
    underlying_type = process_type_ref(qualified_type_attr, die, scope_node)
    if not underlying_type:
        return
    qualified_type = QualifiedType(underlying_type)
    qualified_type.qualifiers.append(qualifier)
    return qualified_type


def process_enum_type(die: DIE, scope_node: ScopeTreeNode, dependency: bool = False) -> None | Enumeration:
    """

    :param die: The DIE containing the enum type definition. This type DIE must NOT have been processed before
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


def array_matcher(type_name: str) -> bool:
    return type_name.startswith('array<')


def string_matcher(type_name: str) -> bool:
    return type_name.startswith('basic_string<')


def unique_ptr_matcher(type_name: str) -> bool:
    return type_name.startswith('unique_ptr<')


def vector_matcher(type_name: str) -> bool:
    return type_name.startswith('vector<')


def get_name(die: DIE, none_val: None | str = None) -> None | str:
    name_attr = die.attributes.get('DW_AT_name', None)
    if name_attr:
        name = name_attr.value.decode('utf-8')
        return name
    if none_val:
        return none_val
    return None


def print_children(die: DIE) -> None:
    for child_die in die.iter_children():
        name = get_name(child_die, '')
        print(f"Child tag={child_die.tag},name={name}")


def print_siblings(die: DIE) -> None:
    for sibling_die in die.iter_siblings():
        name = get_name(sibling_die, '')
        print(f"Sibling tag={sibling_die.tag},name={name}")


def get_child(die: DIE, tag: str, name: str) -> None | DIE:
    """Retrieves a child of the given tag and with the given name

    :return:
    """
    matching_die = None
    for child_die in die.iter_children():
        if child_die.tag != tag:
            continue
        name_attr = child_die.attributes.get('DW_AT_name')
        if not name_attr:
            continue
        if name_attr.value.decode() != name:
            continue
        matching_die = child_die
        break
    return matching_die


STLProcessorResult = namedtuple('STLProcessorResult', 'process_members')


def array_or_vector_processor(type_instance: Array | Vector, die: DIE,
                              struct_node: StructScopeTreeNode) -> STLProcessorResult:
    """Processes a DIE that contains the definition of a std::array or a std::vector insance

    :param type_instance: The array/vector type instance we are populating
    :param die: The DIE containing the definition of this array
    :param struct_node: This array/vector type as a scope, as the vector is itself a struct/class and hence the scope containing all its members
    """
    value_type_typedef_child_die = get_child(die, 'DW_TAG_typedef', 'value_type')
    value_type = process_typedef(value_type_typedef_child_die, struct_node)
    type_instance.value_type = value_type
    # For the moment, stop further processing of this type
    return STLProcessorResult(process_members=False)


# TODO: dedup this function!! IT IS THE SAME ONE as for the VECTOR!
def array_processor(array: Array, die: DIE, array_struct_node: StructScopeTreeNode) -> STLProcessorResult:
    return array_or_vector_processor(array, die, array_struct_node)


def string_processor(string: String, die: DIE, struct_node: StructScopeTreeNode) -> STLProcessorResult:
    logging.error("Not yet implemented")
    return STLProcessorResult(process_members=False)


def unique_ptr_processor(unique_ptr: UniquePtr, die: DIE, struct_node: StructScopeTreeNode) -> STLProcessorResult:
    logging.error("Not yet implemented")
    return STLProcessorResult(process_members=False)


def vector_processor(vector: Vector, die: DIE, vector_struct_node: StructScopeTreeNode) -> STLProcessorResult:
    return array_or_vector_processor(vector, die, vector_struct_node)


STLClassProcessorsEntry = namedtuple('STLClassProcessorsEntry', 'type_class processor')
# A list containing the processors for special STL struct/class types
stl_class_processors = [
    (array_matcher, STLClassProcessorsEntry(Array, array_processor)),
    (string_matcher, STLClassProcessorsEntry(String, string_processor)),
    (unique_ptr_matcher, STLClassProcessorsEntry(UniquePtr, unique_ptr_processor)),
    (vector_matcher, STLClassProcessorsEntry(Vector, vector_processor))
]


def maybe_process_struct_or_class_type(die: DIE, scope_node: ScopeTreeNode,
                                       dependency: bool = False) -> None | ScopeTreeNode:
    """Processes a DIE that is a (tag) structure_type or class_type and its child members.

    Processes as well all its child 'member' DIEs, recursively. Any other type of DIEs are ignored, as they are not
    relevant for the struct/class definition.

    A struct or class declaration CANNOT be processed twice. Make sure to detect this situation outside this function.

    It applies the struct whitelist. If the DIE contains a struct that has not been explicitly whitelisted and
    is NOT marked as dependency of a parent type (`dependency` param)

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
        logging.warning("Ignoring unnamed struct/class DIE @{}".format(die.offset))
        return
    name = die.attributes['DW_AT_name'].value.decode("utf-8")

    full_path = scope_node.get_path() + [name]
    if not dependency and (full_path not in types_whitelist):
        return

    def generic_struct_processing(type_cls: type[StructType]):
        nonlocal die, name, scope_node

        struct_type = type_cls(name)
        struct_type_node = scope_node.add_child(struct_type)
        assert struct_type_node not in scope_node_to_die_map
        scope_node_to_die_map[struct_type_node] = die

        return struct_type, struct_type_node

    # The STL types are also structs (classes). We need to apply a matcher on the name to do additional type-specific
    # processing
    struct_type = struct_type_node = None
    with_dedicated_processor = False
    process_members = True
    if full_path[0] == 'std':
        for matcher, (type_cls, processor) in stl_class_processors:
            if matcher(name):
                with_dedicated_processor = True
                struct_type, struct_type_node = generic_struct_processing(type_cls)
                result = processor(struct_type, die, struct_type_node)
                process_members = result.process_members
                break
        else:
            logging.debug(f"Could not find an STL-processor for {name}. Doing generic struct processing only")
    if not with_dedicated_processor:
        # Do just the basic processing
        struct_type, struct_type_node = generic_struct_processing(StructType)

    # The member fields of a struct are provided as child DIEs
    if process_members:
        for child_die in die.iter_children():
            if child_die.tag == 'DW_TAG_member':
                process_member_die(child_die, struct_type_node)

    scope_node.scope.structs[name] = struct_type

    return struct_type_node


def maybe_recurse_scope_die(die, scope: ScopeTreeNode):
    """Processes and recurses a DIE that represents a scope (namespace, struct or CU)

    Applies namespace blacklist rules

    :param scope: The parent scope where the nested scopes or discovered entries will be placed
    """
    curr_scope_full_path = scope.get_path()

    indent = BASIC_INDENT * scope.depth

    name = die.attributes.get('DW_AT_name', None)
    if name:
        name = name.value.decode('utf-8')

    recurse = True
    match die.tag:
        case 'DW_TAG_compile_unit':
            # Register the CU DIE to the root scope
            scope_node_to_die_map[scope_graph.root] = die

        case 'DW_TAG_namespace':
            assert len(curr_scope_full_path) == scope.depth
            assert name

            new_ns = Namespace(name)
            if is_ns_blacklisted(scope.build_child_scope_path(new_ns)):
                # We will NOT process the children (types or nested NSs) of this scope. But we will still add it to the graph, as blacklisted
                new_ns.blacklisted = True
                recurse = False
            scope = scope.add_child(new_ns)
            assert scope not in scope_node_to_die_map
            scope_node_to_die_map[scope] = die

        case t if t in CLASS_OR_STRUCT_TYPE_TAGS:
            maybe_process_struct_or_class_type(die, scope)
            # for name, val in die.attributes.items():
            #     print(indent_level + '  %s = %s' % (name, val))

        case other:
            logging.info("Ignoring DIE '{}' of unhandled type '{}'".format(name if name else "UNNAMED", die.tag))
            recurse = False

    if recurse:
        for child in die.iter_children():
            # Let's continue recursing non-member DIEs (e.g. other nested namespaces)
            maybe_recurse_scope_die(child, scope)


def is_namespace_glob_expression(val: list[str]) -> bool:
    if not len(val):
        raise ValueError("val cannot be an empty list")

    start_pos = 0
    try:
        star_pos = val.index('*')
    except ValueError as e:
        # Namespace path does not contain *. It is not a NS glob expression
        return False
    if star_pos != len(val) - 1:
        raise ValueError('Malformed glob expression')


def namespace_matches_glob_expression(ns: list[str], ns_glob_expr: list[str]) -> bool:
    assert is_namespace_glob_expression(ns_glob_expr)
    for ns_elem, expr_elem in zip_longest(ns, ns_glob_expr):
        if ns_elem is None:
            if expr_elem == '*':
                # ns finished, but at the same point as the * operator is found in the glob_expr. E.g.:
                # - ns == ['Foo', 'Bar']
                # - expr: == [ 'Foo', 'Bar', '*']
                # The expr is considered as including ALL elements inside Foo::Bar. I.e., the full Foo::Bar.
                # Hence, this is a match
                return True
            else:
                # Given NS is more general than the glob expr. The glob expr specifies a "deeper" scope, and hence does
                # not apply to this one
                return False
        elif expr_elem is None:
            # We cannot reach this point :D. This is covered by the elif below
            assert False
            # The given NS is inside (child) the one in the glob expr
            # As ATM we only support the '*' operator, this is fundamentally a match
            # assert ns_glob_expr[-1] == '*'
            # return True
        elif expr_elem == '*':
            # We reached the end of the glob expression. So far we matched the parent NSs => match
            return True
        elif ns_elem != expr_elem:
            return False

    # We cannot reach this point
    assert False


def match_gnu_namespaces() -> bool:
    """Filters out __gnu_debug, __gnu_cxx, ... and others

    :return:
    """
    return False


types_whitelist: list[list[str]]
# The blacklist can be either:
# * a complete namespace path (sequence of namespaces to fully select a _concrete_ leaf namespace) or
#   Only the namespace direct children are considered. Child namespaces are still evaluated
# * a Simple Namespace Glob Expression (SNGE (TM)).
#   A namespace path + the "include all children" '*' operator, indicating that all the contents of the scope should be avoided
# * a predicate to be applied to a scope node that returns whether the node is blacklisted (return True) or not (False)
namespaces_blacklist = [
    ['std', '*'],  # By default, unless a dependency, do not include any std:: content
    # gnu_namespaces_matcher,
]


def is_ns_blacklisted(ns: list[str]) -> bool:
    for blacklist_entry in namespaces_blacklist:
        match = False
        if is_namespace_glob_expression(blacklist_entry):
            match = namespace_matches_glob_expression(ns, blacklist_entry)
        elif callable(blacklist_entry):
            raise NotImplementedError('NS blacklist predicates still not implemented')
        else:
            match = ns == blacklist_entry
        if match:
            return True
    return False


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    FORMAT = "%(levelname)s:%(name)s:%(funcName)s - %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    if len(sys.argv) < 2:
        print('Expected usage: {0} <executable>'.format(sys.argv[0]))
        sys.exit(1)

    # Associated types will be automatically brought in
    types_whitelist = [
        ['TopStruct'],
        # Example syntax: ['root_namespace', 'nested_ns', 'Type']
    ]
    extract_types(sys.argv[1])
