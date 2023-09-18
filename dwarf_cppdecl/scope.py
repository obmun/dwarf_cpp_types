"""
Support for the creation of trees of scopes containing type definitions
"""

from __future__ import annotations
from annotated_types import Ge
from ._helpers import _Base
import typing as t
from typing import Annotated

UInt = Annotated[int, Ge(0)]


class Scope(_Base):
    """Anything that can contain other elements
    """
    _name: str

    named_types: dict[str, t.Any]
    """Named type-defs of this cope
    
    Contains any named type-definition whose parent is this scope. This of course includes struct declarations.
    """
    # We would like to use type annotations in this dict. It would look like
    #   structs: dict[str, StructType]
    # The problem is that this would create a circular dep. Instead, we make our life easier.
    # See at the end of this file.

    def __init__(self, name: str):
        super().__init__(name)
        self._name = name
        self.named_types = {}

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
        if not parent and depth != 0:
            raise ValueError("No parent provided (=> root), but depth is not 0")

    def add_child(self, scope: Scope) -> 'ScopeTreeNode':
        """Adds a new child scope

        :param scope:
        :return:
        """
        new_child = ScopeTreeNode(scope, self, self._depth + 1)
        self._children.append(new_child)
        return new_child

    @property
    def children(self) -> list['ScopeTreeNode']:
        return self._children

    @property
    def depth(self) -> UInt:
        return self._depth

    @property
    def is_root(self) -> bool:
        return not self.depth

    def get_path(self) -> list[str]:
        if not self._parent:
            # I am the root NS. My path is the empty list
            return []
        return self._parent.get_path() + [self.scope.name]
        # TODO: if we have here a vector (which is underneath a struct), we have a _PROBLEM_

    def build_child_scope_path(self, scope: Scope) -> list[str]:
        """Given a scope, builds the full path as if the namespace itself was a child of this scope-tree node

        :return:
        """
        return self.get_path() + [scope.name]

    def __repr__(self):
        if self.is_root:
            path = "~ROOT~"
        else:
            path = "::".join(self.get_path())
        return f'{self.__class__.__name__}({path})'

    def __str__(self):
        return self.__repr()


class ScopeTree:
    """A type representing a tree of scopes

    """
    _SCOPE_SEPARATOR = "::"
    _root: ScopeTreeNode

    def __init__(self):
        self._root = ScopeTreeNode(RootNs(), None, 0)

    @property
    def root(self) -> ScopeTreeNode:
        return self._root

    @classmethod
    def _depth_first_visit(cls, node: ScopeTreeNode, struct_visitor):
        for child in node.children:
            ScopeTree._depth_first_visit(child, struct_visitor)
        scope_path = cls._SCOPE_SEPARATOR.join(node.get_path())
        for name, s in node.scope.named_types.items():
            struct_visitor(scope_path, name, s)

    def flatten(self):
        ret = {}

        def add_structs_to_dict(scope_path: str, name: str, s):
            nonlocal ret
            ret[self._SCOPE_SEPARATOR.join((scope_path, name))] = s

        self._depth_first_visit(self.root, add_structs_to_dict)
        return ret



# We are using a forward declaration of this class. In order to avoid the circular import, just move the full
# definition import until the end of the module, so the module is fully populated before we bring the evaluation of
# types.defs
#from .types.defs import StructType
# ^^ This does not always work. Try to import types.defs :D
