from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Type
from importlib import import_module

from marker.schema import BlockTypes
from marker.schema.blocks import (
    Bibliography,
    Block,
    Caption,
    ChemicalBlock,
    Diagram,
    Code,
    Equation,
    Figure,
    Footnote,
    Form,
    Handwriting,
    InlineMath,
    ListItem,
    PageFooter,
    PageHeader,
    Picture,
    SectionHeader,
    Table,
    TableOfContents,
    Text,
    ComplexRegion,
    TableCell,
    Reference,
)
from marker.schema.document import Document
from marker.schema.groups import (
    FigureGroup,
    ListGroup,
    PageGroup,
    PictureGroup,
    TableGroup,
)
from marker.schema.text import Line, Span
from marker.schema.text.char import Char

BLOCK_REGISTRY: Dict[BlockTypes, str] = {}
_block_overrides: ContextVar[Mapping[BlockTypes, Type[Block]]] = ContextVar(
    "block_overrides", default={}
)


@contextmanager
def use_block_overrides(overrides: Mapping[BlockTypes, Type[Block]]):
    """Use one converter's block classes for this operation."""
    token = _block_overrides.set(overrides)
    try:
        yield
    finally:
        _block_overrides.reset(token)


def register_block_class(block_type: BlockTypes, block_cls: Type[Block]):
    BLOCK_REGISTRY[block_type] = f"{block_cls.__module__}.{block_cls.__name__}"


def get_block_class(block_type: BlockTypes) -> Type[Block]:
    override = _block_overrides.get().get(block_type)
    if override is not None:
        return override
    class_path = BLOCK_REGISTRY[block_type]
    module_name, class_name = class_path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


register_block_class(BlockTypes.Line, Line)
register_block_class(BlockTypes.Span, Span)
register_block_class(BlockTypes.Char, Char)
register_block_class(BlockTypes.FigureGroup, FigureGroup)
register_block_class(BlockTypes.TableGroup, TableGroup)
register_block_class(BlockTypes.ListGroup, ListGroup)
register_block_class(BlockTypes.PictureGroup, PictureGroup)
register_block_class(BlockTypes.Page, PageGroup)
register_block_class(BlockTypes.Caption, Caption)
register_block_class(BlockTypes.Code, Code)
register_block_class(BlockTypes.Figure, Figure)
register_block_class(BlockTypes.Footnote, Footnote)
register_block_class(BlockTypes.Form, Form)
register_block_class(BlockTypes.Equation, Equation)
register_block_class(BlockTypes.Handwriting, Handwriting)
register_block_class(BlockTypes.TextInlineMath, InlineMath)
register_block_class(BlockTypes.ListItem, ListItem)
register_block_class(BlockTypes.PageFooter, PageFooter)
register_block_class(BlockTypes.PageHeader, PageHeader)
register_block_class(BlockTypes.Picture, Picture)
register_block_class(BlockTypes.SectionHeader, SectionHeader)
register_block_class(BlockTypes.Table, Table)
register_block_class(BlockTypes.Text, Text)
register_block_class(BlockTypes.TableOfContents, TableOfContents)
register_block_class(BlockTypes.ComplexRegion, ComplexRegion)
register_block_class(BlockTypes.TableCell, TableCell)
register_block_class(BlockTypes.Reference, Reference)
register_block_class(BlockTypes.Bibliography, Bibliography)
register_block_class(BlockTypes.ChemicalBlock, ChemicalBlock)
register_block_class(BlockTypes.Diagram, Diagram)
register_block_class(BlockTypes.Document, Document)

assert len(BLOCK_REGISTRY) == len(BlockTypes)
assert all(
    [
        get_block_class(k).model_fields["block_type"].default == k
        for k, _ in BLOCK_REGISTRY.items()
    ]
)
