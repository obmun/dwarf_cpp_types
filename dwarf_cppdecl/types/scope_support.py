from itertools import zip_longest
from ..scope import ScopeTreeNode
from .defs import StructType


class StructScopeTreeNode(ScopeTreeNode):
    """A scope-graph node where the scope is a struct (and not a namespace)

    """
    scope: 'StructType'


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


# The blacklist can be either:
# * a complete namespace path (sequence of namespaces to fully select a _concrete_ leaf namespace) or
#   Only the namespace direct children are considered. Child namespaces are still evaluated
# * a Simple Namespace Glob Expression (SNGE (TM)).
#   A namespace path + the "include all children" '*' operator, indicating that all the contents of the scope should be avoided
# * a predicate to be applied to a scope node that returns whether the node is blacklisted (return True) or not (False)
_NAMESPACES_BLACKLIST = [
    ['std', '*'],  # By default, unless a dependency, do not include any std:: content
    # gnu_namespaces_matcher,
]


def is_ns_blacklisted(ns: list[str]) -> bool:
    for blacklist_entry in _NAMESPACES_BLACKLIST:
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

