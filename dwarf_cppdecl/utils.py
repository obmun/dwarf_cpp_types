from elftools.dwarf.die import DIE


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
