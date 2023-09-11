from collections import namedtuple

import elftools.dwarf.constants as dc
import logging
import numpy as np
import typing as t
from bidict import bidict
from pathlib import Path

from elftools.dwarf.die import DIE
from elftools.dwarf.die import AttributeValue
from elftools.elf.elffile import ELFFile

from .scope import Namespace, ScopeTree, ScopeTreeNode
from .types.defs import *
from .types.scope_support import is_ns_blacklisted, StructScopeTreeNode
from .types import stl
from .utils import *


class TypesExtractor:
    _BASIC_INDENT = '    '
    _CLASS_OR_STRUCT_TYPE_TAGS = ['DW_TAG_structure_type', 'DW_TAG_class_type']
    _QUALIFIER_TAGS = ['DW_TAG_const_type']

    _scope_node_to_die_map: bidict['ScopeTreeNode', DIE]
    _die_to_type_map: bidict[DIE, 'TypeDef']
    # ^^ THIS IS FUCKED! I am holding the DIE instances in memory!!!
    # TODO: I need a unique ID for each DIE. Maybe the offset?
    _scope_tree: ScopeTree
    types_whitelist: list[list[str]]

    def __init__(self):
        self._scope_node_to_die_map = bidict()
        self._die_to_type_map = bidict()
        self._scope_tree = ScopeTree()
        self.types_whitelist = None

    def process(self, filename, types: t.Optional[list[str]] = None):
        """
        Extracts the C++ types from the given ELF file

        :param types: a list of types to recover. If not provided, all types will be extracted
        """

        self.types_whitelist = types
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

                self._maybe_recurse_scope_die(top_die, self._scope_tree.root)

        self.types_whitelist

    @property
    def scope_tree(self):
        return self._scope_tree

    @staticmethod
    def _dwarf_base_type_to_py_type(die: DIE) -> PrimitiveType:
        assert die.tag == 'DW_TAG_base_type'

        MAPPING: t.Final[dict] = {
            (dc.DW_ATE_boolean, 1): np.bool_,
            (dc.DW_ATE_unsigned_char, 1): np.ubyte,  # According to NumPy doc (https://numpy.org/doc/stable/user/basics
            # .types.html), ubyte is equivalent to unsigned char. Review if this is the correct assignment
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

        return PrimitiveType(
            MAPPING[(die.attributes["DW_AT_encoding"].value, die.attributes['DW_AT_byte_size'].value)],
            die.attributes['DW_AT_name'].value.decode('utf-8'))

    def _process_type_die(self, die: DIE, scope_node: t.Optional[ScopeTreeNode], dependency: bool = False) -> t.Optional[TypeDef]:
        """Processes a DIE that contains the declaration of a type

        This is a helper method. Feed it any of the DWARF type declarations DIEs, and it will take care of execute the right processing method.

        :param die: The DIE to process
        :param scope_node: If None, we will use the PARENT die to find the right SCOPE. Otherwise, the scope that will be the parent of the new type instance

        :param dependency: Indicates whether this type is a dependency of other type (e.g., the type of a member of a
        struct). If True, this type will be imported even if it is not in the whitelist, as some type in its dependees
        have been whitelisted.

        :return: The type-definition from the interpretation of this DIE and children. None if the type-def could not
        be interpreted.
        """
        parent_die = None
        if not scope_node:
            parent_die = die.get_parent()

        # TODO: here we will always be doing FIRST TIME processing of a type, right? We are processing the DIE type ...
        # How are we gonna handle references to existing types? ATM we only handle Refs to structs... But I guess we will have refs to EVERYTHING

        ret = None
        add_type_def_to_scope = False
        match die.tag:
            case 'DW_TAG_base_type':
                ret = self._dwarf_base_type_to_py_type(die)

            case 'DW_TAG_pointer_type':
                ret = self._process_c_pointer(die, scope_node)

            case 'DW_TAG_typedef':
                #add_type_def_to_scope = True
                # TODO: this can be generalized.
                # Option 1: add a property to the type definition:
                #     if hasattr(type_def, `name`):
                #         scope.add_named_type_def()
                # Option 2 (better): make the TypeDef have both `name` and `def` properties :)
                ret = self._process_typedef(die, scope_node)

            case 'DW_TAG_array_type':
                # !Array DIEs normally have children!
                #raise NotImplementedError("No array support ...")
                pass

            case 'DW_TAG_enumeration_type':
                ret = self._process_enum_type(die, scope_node, dependency)

            case tag if tag in self._QUALIFIER_TAGS:
                # A type qualifier is a DIE with a relevant tag and a DW_AT_type attr of "ref" form
                # E.g.:
                # - const: DIE with tag==const_type with DW_AT_type-attribute of form==reference
                ret = self._recurse_extracting_qualifiers(die, scope_node)

            case tag if tag in self._CLASS_OR_STRUCT_TYPE_TAGS:
                add_type_def_to_scope = True
                struct_type, struct_graph_node = (self._maybe_process_struct_or_class_type(die, scope_node, dependency)
                                                  or (None, None))
                ret = struct_type
                # ret = Reference(struct_type)
                # TODO: why did I do this? The struct type def in Python is already a reference to the object ...

            case other:
                # raise RuntimeError('Do not know how to handle type die of type {}'.format(die.tag))
                pass

        if ret:
            self._die_to_type_map[die] = ret
            if add_type_def_to_scope:
                scope_node.scope.structs[struct_type.cpp_decl] = struct_type
        return ret

    def _process_c_pointer(self, die: DIE, scope: ScopeTreeNode) -> CPointer:
        assert die.tag == 'DW_TAG_pointer_type'

        # A C pointer DIE either:
        # - Contains a DW_AT_type attr, for pointers pointer to a concrete type
        # - Does NOT CONTAIN 'DW_AT_type' attr in case of void* pointer
        if 'DW_AT_type' not in die.attributes:
            # This is a void ptr
            referenced_type = Void()
        else:
            referenced_type = self._process_type_attr(die, scope)

        ret = CPointer()
        ret.element_type = referenced_type
        return ret

    def _process_typedef(self, die: DIE, scope: ScopeTreeNode) -> Reference:
        assert die.tag == 'DW_TAG_typedef'
        # A typedef DIE just has a 'DW_AT_type' attr that we need to process, referencing another typoe.
        referenced_type = self._process_type_attr(die, scope)
        # However, as we have a bidir map between DIEs and created types, the typedef must be created as well as its own
        # "unique" type, and reference the real underyling type. Just exactly as DWARF does it
        ret = Reference(die.attributes['DW_AT_name'].value.decode(), referenced_type)
        return ret

    def _process_type_ref(self, attr: AttributeValue, die: DIE, scope: ScopeTreeNode) -> t.Optional[TypeDef]:
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
        existing_type = self._die_to_type_map.get(type_die)
        if existing_type:
            return existing_type

        # New type. Process it

        parent_die = type_die.get_parent()
        assert scope in self._scope_node_to_die_map
        caller_provided_scope_die = self._scope_node_to_die_map[scope]
        if caller_provided_scope_die != parent_die:
            # This is a reference to a type in a different scope.
            # 1) We need to parent the referenced type to its CORRECT scope
            # 2) It might be the case that we have NOT processed yet the parent scope

            if parent_die.tag == 'DW_TAG_compile_unit':
                # If the real parent is just the CU, this is a non-namespaced type (root) or a base type
                # Just use the ROOT ns as parent
                scope = self._scope_tree.root

                # DWARF notes:
                # * When the compiler (at least GCC) creates DWARF info for C-pointer declarations, these are sometimes directly parented to the CU
                #   - It seems that these basic declarations are NOT parented  inside the containing struct whose member uses this basic pointer type
                #     Probably for dedup?
                #   - E.g.: for a member of a struct of type `int16 *ptr;` this has been the case.
            else:
                if parent_die not in self._scope_node_to_die_map.inverse:
                    # Not processed before! Time to do it now!

                    # ADD a filter to blackslit:
                    # 1. Types that start with __
                    # 2. std:: types by default, except certain accepted types (the ones from the stl_processors)
                    raise NotImplementedError()
                else:
                    scope = self._scope_node_to_die_map.inverse[parent_die]

        return self._process_type_die(type_die, scope, True)

    def _process_type_attr(self, die: DIE, scope: ScopeTreeNode):
        """Process the attribute that defines the type of this Debug Info Entry

        :param die:
        :param scope: The scope (parent struct or namespace) of the "declaration" whose type we are processing. If this type ends up being a structure (another scope), we need to nest it inside the parent scope
        :return:
        """
        type_attr = die.attributes['DW_AT_type']
        match type_attr.form:
            case 'DW_FORM_ref4':
                return self._process_type_ref(type_attr, die, scope)
            case other:
                raise RuntimeError('Do not know now how to handle type attr of form {}'.format(type_attr.form))

    def _process_member_die(self, die: DIE, parent_struct_node: StructScopeTreeNode) -> None:
        """Processes a tag==member child DIE to add the member description to the parent struct

        :param die:
        :param parent_struct_node:
        :return:
        """
        assert die.tag == 'DW_TAG_member'

        parent_struct_type = parent_struct_node.scope
        parent_struct_def = parent_struct_type.declaration
        indent = self._BASIC_INDENT * parent_struct_node.depth

        field_name = die.attributes['DW_AT_name'].value.decode('utf-8')
        type_def = self._process_type_attr(die, parent_struct_node)
        if not type_def:
            logging.error(
                "Could not process type for member '{}::{}'. Definition will be incomplete".format(
                    parent_struct_type.name,
                    field_name))

        attr_indent = indent + self._BASIC_INDENT

        assert field_name not in parent_struct_def.fields
        parent_struct_def.fields[field_name] = type_def

    def _recurse_extracting_qualifiers(self, die: DIE, scope_node: ScopeTreeNode) -> None | QualifiedType:
        tag_to_qualifier_type = {
            'DW_TAG_const_type': Const
        }
        qualifier = tag_to_qualifier_type[die.tag]
        qualified_type_attr = die.attributes['DW_AT_type']
        if qualified_type_attr.form != 'DW_FORM_ref4':
            raise NotImplementedError(f'Do not know how to handle DW_AT_type attr of form {qualified_type_attr.form}')
        underlying_type = self._process_type_ref(qualified_type_attr, die, scope_node)
        if not underlying_type:
            return
        qualified_type = QualifiedType(underlying_type)
        qualified_type.qualifiers.append(qualifier)
        return qualified_type

    @staticmethod
    def _process_enum_type(die: DIE, scope_node: ScopeTreeNode, dependency: bool = False) -> None | Enumeration:
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

    @staticmethod
    def array_matcher(type_name: str) -> bool:
        return type_name.startswith('array<')

    @staticmethod
    def string_matcher(type_name: str) -> bool:
        return type_name.startswith('basic_string<')

    @staticmethod
    def unique_ptr_matcher(type_name: str) -> bool:
        return type_name.startswith('unique_ptr<')

    @staticmethod
    def vector_matcher(type_name: str) -> bool:
        return type_name.startswith('vector<')

    _STLProcessorResult = namedtuple('STLProcessorResult', 'process_members')

    def array_or_vector_processor(self, type_instance: stl.Array | stl.Vector, die: DIE,
                                  struct_node: StructScopeTreeNode) -> _STLProcessorResult:
        """Processes a DIE that contains the definition of a std::array or a std::vector insance

        :param type_instance: The array/vector type instance we are populating
        :param die: The DIE containing the definition of this array
        :param struct_node: This array/vector type as a scope, as the vector is itself a struct/class and hence the scope containing all its members
        """
        value_type_typedef_child_die = get_child(die, 'DW_TAG_typedef', 'value_type')
        value_type = self._process_typedef(value_type_typedef_child_die, struct_node)
        type_instance.value_type = value_type
        # For the moment, stop further processing of this type
        return self._STLProcessorResult(process_members=False)

    def array_processor(self, array: stl.Array, die: DIE,
                        array_struct_node: StructScopeTreeNode) -> _STLProcessorResult:
        return self.array_or_vector_processor(array, die, array_struct_node)

    def string_processor(self, string: stl.String, die: DIE, struct_node: StructScopeTreeNode) -> _STLProcessorResult:
        logging.error("Not yet implemented")
        return self._STLProcessorResult(process_members=False)

    def unique_ptr_processor(self, unique_ptr: stl.UniquePtr, die: DIE,
                             struct_node: StructScopeTreeNode) -> _STLProcessorResult:
        logging.error("Not yet implemented")
        return self._STLProcessorResult(process_members=False)

    def vector_processor(self, vector: stl.Vector, die: DIE,
                         vector_struct_node: StructScopeTreeNode) -> _STLProcessorResult:
        return self.array_or_vector_processor(vector, die, vector_struct_node)

    _STLClassProcessorsEntry = namedtuple('STLClassProcessorsEntry', 'type_class processor')
    # A list containing the processors for special STL struct/class types
    _STL_CLASS_PROCESSORS = [
        (array_matcher, _STLClassProcessorsEntry(stl.Array, array_processor)),
        (string_matcher, _STLClassProcessorsEntry(stl.String, string_processor)),
        (unique_ptr_matcher, _STLClassProcessorsEntry(stl.UniquePtr, unique_ptr_processor)),
        (vector_matcher, _STLClassProcessorsEntry(stl.Vector, vector_processor))
    ]

    def _maybe_process_struct_or_class_type(self, die: DIE, scope_node: ScopeTreeNode,
                                            dependency: bool = False) -> None | tuple[StructType, ScopeTreeNode]:
        """Processes a DIE that is a (tag) structure_type or class_type and its child members.

        A complete definition of the struct is created. I.e., it processes as well all its child 'member' DIEs,
        recursively. Any other type of children DIEs are ignored, as they are not relevant for the struct/class
        definition.

        A struct or class declaration CANNOT be processed twice. Make sure to detect this situation outside this
        function.

        It applies the struct whitelist. If the DIE contains a struct that has not been explicitly whitelisted and
        is NOT marked as dependency of a parent type (`dependency` param)

        This method takes care of modifying the scope tree, adding the new scope that the new structure creates to
        the tree.

        .. important::
           This method does NOT add the struct type definition to its scope. It ONLY modifies the scope tree.
           The reason is that ALL type definitions go through the `_process_type_die()` method.

        :param die:
        :param scope_node:

        :param dependency: Indicates whether this type is a dependency of other type (e.g., the type of a member of a
        struct). If True, this type will be imported even if it is not in the whitelist, as some type in its dependees
        have been whitelisted.

        :return: None or the struct type def and the scope tree node that corresponds to this new structure or class
        in the scope tree
        """

        assert die.tag in self._CLASS_OR_STRUCT_TYPE_TAGS

        if 'DW_AT_name' not in die.attributes:
            logging.warning("Ignoring unnamed struct/class DIE @{}".format(die.offset))
            return
        name = die.attributes['DW_AT_name'].value.decode("utf-8")

        full_path = scope_node.get_path() + [name]
        if not dependency and self.types_whitelist:
            if full_path not in self.types_whitelist:
                return

        # Consider using a TypeVar here to be able to define the return type of this helper function
        def generic_struct_processing(type_cls: type[StructType]):
            nonlocal die, name, scope_node

            struct_type = type_cls(name)
            struct_type_node = scope_node.add_child(struct_type)
            assert struct_type_node not in self._scope_node_to_die_map
            self._scope_node_to_die_map[struct_type_node] = die

            return struct_type, struct_type_node

        # The STL types are also structs (classes). We need to apply a matcher on the name to do additional
        # type-specific processing
        struct_type = struct_type_node = None
        with_dedicated_processor = False
        process_members = True
        if full_path[0] == 'std':
            for matcher, (type_cls, processor) in self._STL_CLASS_PROCESSORS:
                if matcher(name):
                    with_dedicated_processor = True
                    struct_type, struct_type_node = generic_struct_processing(type_cls)
                    result = processor(self, struct_type, die, struct_type_node)
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
                    self._process_member_die(child_die, struct_type_node)

        return struct_type, struct_type_node

    def _maybe_recurse_scope_die(self, die, scope: ScopeTreeNode):
        """Processes and recurses a DIE that represents a scope (namespace, struct or CU)

        Applies namespace blacklist rules

        :param scope: The parent scope where the nested scopes or discovered entries will be placed
        """
        curr_scope_full_path = scope.get_path()

        indent = self._BASIC_INDENT * scope.depth

        name = die.attributes.get('DW_AT_name', None)
        if name:
            name = name.value.decode('utf-8')

        recurse = True
        match die.tag:
            case 'DW_TAG_compile_unit':
                # Register the CU DIE to the root scope
                self._scope_node_to_die_map[self._scope_tree.root] = die

            case 'DW_TAG_namespace':
                assert len(curr_scope_full_path) == scope.depth
                assert name

                new_ns = Namespace(name)
                if is_ns_blacklisted(scope.build_child_scope_path(new_ns)):
                    # We will NOT process the children (types or nested NSs) of this scope. But we will still add it to the graph, as blacklisted
                    new_ns.blacklisted = True
                    recurse = False
                scope = scope.add_child(new_ns)
                assert scope not in self._scope_node_to_die_map
                self._scope_node_to_die_map[scope] = die

            case t if t in self._CLASS_OR_STRUCT_TYPE_TAGS:
                # We call _maybe_process_struct_or_class_type always THRU the _process_type_die main entry point
                self._process_type_die(die, scope)
                # The struct_or_class processor already recurses needed members
                recurse = False

            case other:
                logging.info("Ignoring DIE '{}' of unhandled type '{}'".format(name if name else "UNNAMED", die.tag))
                recurse = False

        if recurse:
            for child in die.iter_children():
                # Let's continue recursing non-member DIEs (e.g. other nested namespaces)
                self._maybe_recurse_scope_die(child, scope)
