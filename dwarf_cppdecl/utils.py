from __future__ import annotations
from bidict import bidict
from elftools.dwarf.compileunit import CompileUnit
from elftools.dwarf.die import DIE


class DieUniqueId:
    """A unique ID for a DIE instance

    Hashable. To be used as a key in dictionaries!
    """
    _value = int

    def __init__(self, die: DIE):
        # The DIE offset is a UNIQUE property of the DIE inside a CU. No 2 DIEs can share the same offset.
        self._value = die.offset

    @property
    def value(self) -> int:
        return self._value

    def __hash__(self):
        return self._value

    def __eq__(self, other: DieUniqueId):
        return self._value == other._value

    def __ne__(self, other: DieUniqueId):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)


class DieMaps:
    """A helper class with a few maps of DIEs to some produced information

    Contains:

    - A `ScopeTreeNode` <-> DIE bidir map
    - A `TypeDef` <-> DIE bidir map

    The DIE instance that most of the methods of this class gets as argument IS NOT DIRECTLY used as a KEY in the
    internal maps. Instead, instances of `DieUniqueId` are produced. This ensure that we do not keep in memory
    references of all the DIEs processed, but rather we allow the GC to dispose of them.

    We use directly the REFERENCES to the provided scope tree nodes and TypeDefs. I.e.: we will keep these objects alive as long as they are inside this class.
    """
    _scope_node_to_die_map: bidict[ScopeTreeNode, DieUniqueId]
    _die_to_type_map: bidict[DieUniqueId, TypeDef]

    def __init__(self, cu: CompileUnit):
        self._scope_node_to_die_map = bidict()
        self._die_to_type_map = bidict()
        self._cu = cu

    def _unique_id_to_die(self, uid: DieUniqueId) -> DIE:
        # The DIE.offset property is already an OFFSET relative to the start of the stream! I.e.: it already contains
        # the CU offset. This is different from the DW_FORM_ref4 value, so do not get confused.
        return self._cu.get_DIE_from_refaddr(uid.value)

    def add_scope_die(self, die: DIE, scope: 'ScopeTreeNode') -> None:
        self._scope_node_to_die_map[scope] = DieUniqueId(die)

    def has_scope(self, scope: 'ScopeTreeNode') -> bool:
        """Checks whether the given scope has been previously added to the scope-DIE map

        :param scope:
        :return:
        """
        return scope in self._scope_node_to_die_map

    def has_scope_die(self, die: DIE) -> bool:
        """Checks if the given DIE, that corresponds to a scope, has been added to the map

        :param die:
        :return:
        """
        return DieUniqueId(die) in self._scope_node_to_die_map.inverse

    def get_scope_from_die(self, die: DIE) -> 'ScopeTreeNode':
        return self._scope_node_to_die_map.inverse[DieUniqueId(die)]

    def get_die_for_scope(self, scope: 'ScopeTreeNode') -> DIE:
        return self._unique_id_to_die(self._scope_node_to_die_map[scope])

    def add_type_die(self, die: DIE, type_def: 'TypeDef') -> None:
        self._die_to_type_map[DieUniqueId(die)] = type_def

    def get_type_from_die(self, type_die: DIE) -> None | 'TypeDef':
        """Tries to retrieve the DIE for a previously registered processed TypeDef 
        
        :param type_die: the DIE possibly associated with an already processed TypeDef
        :return: 
        """
        return self._die_to_type_map.get(DieUniqueId(type_die))


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

    :param name: The name of the child to fetch. If the empty string, the FIRST child matching the tag will be returned.
    :return:
    """
    matching_die = None
    for child_die in die.iter_children():
        if child_die.tag != tag:
            continue

        if name == "":
            return child_die

        name_attr = child_die.attributes.get('DW_AT_name')
        if not name_attr:
            continue
        if name_attr.value.decode() != name:
            continue
        matching_die = child_die
        break
    return matching_die


def count_children(die: DIE, tag: str) -> int:
    """Counts the # of children with the given tag

    :param die:
    :param tag:
    :return:
    """
    count = 0
    for child_die in die.iter_children():
        if child_die.tag == tag:
            count += 1
    return count


from .types.defs import TypeDef
