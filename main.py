import elftools.dwarf.constants as dc
import logging
import numpy as np
import sys
import typing as t
from pathlib import Path
from typing import Annotated
from annotated_types import Ge

from elftools.elf.elffile import ELFFile
from elftools.dwarf.die import DIE
from elftools.dwarf.die import AttributeValue

BASIC_INDENT = '    '


# from elftools.common.utils import bytes2str


class FieldDef:
    name: str

    def __init__(self, name: str, type):
        self.name = name
        self.type = type


class StructDefinition:
    name: str
    fields: dict[str, FieldDef]

    def __init__(self, name: str):
        self.name = name
        self.fields = {}
        pass


UInt = Annotated[int, Ge(0)]


class Namespace:
    name: str
    depth: UInt
    nested_namespaces: list['Namespace']
    structs: dict[str, StructDefinition]

    def __init__(self, name: str, depth: UInt):
        self.name = name
        self.depth = depth
        self.nested_namespaces = []
        self.structs = {}


class RootNs(Namespace):
    def __init__(self):
        super().__init__('root', 0)


class DwarfRef:
    pass


class CURef(DwarfRef):
    """A reference within the same Containing Unit

    Identifies any debugging information entry within the containing unit. It is an offset from the first byte of the
    compilation header for the compilation unit containing the reference"""

    offset: int

    def __init__(self, offset: int):
        self.offset = offset

def process_file_simple(filename):
    print('Processing file:', filename)
    with open(filename, 'rb') as f:
        elffile = ELFFile(f)

        if not elffile.has_dwarf_info():
            print('  file has no DWARF info')
            return

        # get_dwarf_info returns a DWARFInfo context object, which is the
        # starting point for all DWARF-based processing in pyelftools.
        dwarf_info = elffile.get_dwarf_info()

        root_ns = RootNs()

        for cu in dwarf_info.iter_CUs():
            # Start with the top DIE, the root for this CU's DIE tree
            top_DIE = cu.get_top_DIE()
            print('    Top DIE with tag=%s' % top_DIE.tag)

            # We're interested in the filename...
            print('    name=%s' % Path(top_DIE.get_full_path()).as_posix())

            # Display DIEs recursively starting with top_DIE
            die_info_rec(top_DIE, root_ns)

        pass


class TypeDef:
    name: str

    def __init__(self, name:str):
        self.name = name


class PrimitiveType(TypeDef):
    value: t.Any
    name: str

    def __init__(self, value, name:str):
        super().__init__(name)
        self.value = value


def dwarf_base_type_to_py_type(die: DIE):
    assert die.tag == 'DW_TAG_base_type'

    mapping = {
        (dc.DW_ATE_boolean, 1): np.bool_,
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


def process_type_die(die: DIE):
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
            #raise NotImplementedError("No array support ...")
            return None
        case other:
            raise RuntimeError('Do not know how to handle type die of type {}'.format(die.tag))


def process_type_attr(attr: AttributeValue, die: DIE):
    match attr.form:
        case 'DW_FORM_ref4':
            ref_addr = attr.value + die.cu.cu_offset
            type_die = die.dwarfinfo.get_DIE_from_refaddr(ref_addr, die.cu)
            return process_type_die(type_die)
        case other:
            raise RuntimeError('Do not know now how to handle type attr of form {}'.format(attr.form))

    pass


def process_member_die(die: DIE, parent_struct: StructDefinition, indent_level) -> None:
    assert die.tag == 'DW_TAG_member'

    field_name = die.attributes['DW_AT_name'].value.decode('utf-8')
    type_def = process_type_attr(die.attributes['DW_AT_type'], die)
    if not type_def:
        logging.error("Could not process type for member '{}::{}'. Definition will be incomplete".format(parent_struct.name, field_name))
    field = FieldDef(field_name, type_def)

    print(indent_level + 'DIE tag=%s, attrs=' % die.tag)
    attr_indent = indent_level + '  '
    for name, val in die.attributes.items():
        print(attr_indent + '  %s = %s' % (name, val))

    assert field_name not in parent_struct.fields
    parent_struct.fields[field_name] = field


def process_structure_type(die: DIE, curr_ns: Namespace):
    assert die.tag == 'DW_TAG_structure_type'

    indent = BASIC_INDENT * curr_ns.depth

    if 'DW_AT_name' not in die.attributes:
        return
    name = die.attributes['DW_AT_name'].value.decode("utf-8")
    struct_def = StructDefinition(name=name)

    for attr_name, attr_val in die.attributes.items():
        print(indent + '  %s = %s' % (attr_name, attr_val))

    child_indent = indent + BASIC_INDENT
    for child_die in die.iter_children():
        if child_die.tag == 'DW_TAG_member':
            process_member_die(child_die, struct_def, child_indent)

    curr_ns.structs[name] = struct_def


curr_ns_full_path = []


def die_info_rec(die, ns: Namespace):
    """ A recursive function for showing information about a DIE and its
        children.
    """
    global curr_ns_full_path

    indent = BASIC_INDENT * ns.depth
    print(indent + 'DIE tag=%s, attrs=' % die.tag)

    ns_name = None
    match die.tag:
        case 'DW_TAG_namespace':
            assert len(curr_ns_full_path) == ns.depth

            ns_name = die.attributes['DW_AT_name'].value.decode('utf-8')
            curr_ns_full_path += [ns_name]

            new_ns = Namespace(ns_name, ns.depth + 1)
            ns.nested_namespaces.append(new_ns)
            ns = new_ns

        case 'DW_TAG_structure_type' if 'std' not in curr_ns_full_path:
            process_structure_type(die, ns)
            # for name, val in die.attributes.items():
            #     print(indent_level + '  %s = %s' % (name, val))

        case other:
            pass

    for child in die.iter_children():
        die_info_rec(child, ns)

    if ns_name:
        del curr_ns_full_path[-1]


if __name__ == '__main__':
    if sys.argv[1] == '--test':
        process_file_simple(sys.argv[2])
        sys.exit(0)

    if len(sys.argv) < 2:
        print('Expected usage: {0} <executable>'.format(sys.argv[0]))
        sys.exit(1)
    process_file_simple(sys.argv[1])
