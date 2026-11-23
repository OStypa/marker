import pytest
from bs4 import BeautifulSoup

from marker.renderers.markdown import MarkdownRenderer
from marker.schema import BlockTypes
from marker.processors.table import TableProcessor
from marker.processors.table_recon import (
    match_table_cells,
    reconstruct_source_table,
    table_lines_from_pdftext,
)
from marker.schema.blocks.table import Table
from marker.schema.blocks.text import Text
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox


def test_native_tokens_separate_columns_and_keep_decimal_points():
    chars = [
        {"char": char, "bbox": [left, top, right, 20]}
        for char, left, right, top in (
            ("A", 10, 15, 10),
            (" ", 90, 90, 20),
            ("-", 90, 92, 18),
            ("0", 93, 98, 10),
            (".", 99, 100, 19),
            ("1", 101, 106, 10),
        )
    ]
    line = {"bbox": [10, 10, 106, 20], "spans": [{"chars": chars}]}
    assert table_lines_from_pdftext(
        {"blocks": [{"lines": [line]}]}, [0, 0, 120, 30]
    ) == [([("A", 10, 15), ("-0.1", 90, 106)], 10, 20)]


def test_numbered_grid_preserves_heading_and_separate_rows():
    group = ([("Earnings", 95, 155)], 10, 20)
    header = ([("Age", 10, 35), ("(1)", 95, 105), ("(2)", 145, 155)], 25, 35)
    data = [
        ([("12", 10, 20), ("-0.035", 90, 112), ("0.002", 142, 158)], 45, 55),
        ([("(0.047)", 90, 112), ("(0.054)", 142, 158)], 65, 75),
    ]
    html, cells = reconstruct_source_table([group, header, *data])
    assert html.startswith("<p>Earnings</p>")
    assert [entry[2] for entry in cells] == [
        "Age",
        "(1)",
        "(2)",
        "12",
        "-0.035",
        "0.002",
        "",
        "(0.047)",
        "(0.054)",
    ]

    earlier_data = (
        [("Participants", 10, 40), ("25", 96, 104), ("30", 146, 154)],
        10,
        20,
    )
    assert reconstruct_source_table([earlier_data, header, *data]) is None
    misplaced = ([("13", 10, 20), ("0.013", 128, 139)], 85, 95)
    assert reconstruct_source_table([group, header, *data, misplaced]) is None


def test_unresolved_table_keeps_evidence_and_uses_a_text_block():
    polygon = PolygonBox.from_bbox([0, 0, 200, 150])
    table = Table(page_id=0, block_id=0, polygon=polygon)
    table_id = str(table.id)
    page = PageGroup(
        page_id=0, block_id=0, polygon=polygon, children=[table], structure=[table.id]
    )
    document = Document(filepath="test-table.pdf", pages=[page])
    TableProcessor(None, config={"disable_ocr": True})(document)

    assert table.removed
    assert table.native_table_source["source_block_id"] == table_id
    assert table.native_table_source["reconstruction_disposition"] == "not_accepted"
    assert isinstance(page.get_block(page.structure[0]), Text)


def test_cells_match_one_source_row_and_keep_signed_values():
    html = (
        "<table><tr><th>Outcome</th><th>(1)</th><th>(2)</th></tr>"
        "<tr><td>Wages</td><td>-0.18**</td><td>-0.18**</td></tr>"
        "<tr><td></td><td>(0.03)</td><td>(0.05)</td></tr></table>"
    )
    lines = [
        ([("Outcome", 10, 35), ("(1)", 103, 112), ("(2)", 156, 165)], 10, 20),
        ([("Wages", 10, 32), ("-0.18**", 100, 118), ("-0.18**", 150, 168)], 30, 40),
        ([("(0.03)", 102, 115), ("(0.05)", 152, 166)], 50, 60),
    ]
    cells = match_table_cells(html, lines)
    assert len(cells) == 9
    assert cells[4] == (1, 1, "-0.18**", (100, 30, 118, 40))
    assert cells[5] == (1, 2, "-0.18**", (150, 30, 168, 40))
    assert cells[6] == (2, 0, "", (10, 50, 35, 60))
    swapped = html.replace("(0.03)</td><td>(0.05)", "(0.05)</td><td>(0.03)")
    assert match_table_cells(swapped, lines) == []
    merged = html.replace("<th>(1)</th><th>(2)</th>", '<th colspan="2">(1) (2)</th>')
    assert match_table_cells(merged, lines) == []
    headerless = "<table>" + html[html.index("<tr><td>") :].replace(
        "<tr><td>Wages</td><td>-0.18**</td><td>-0.18**</td>",
        "<tr><th>Wages</th><th>-0.18**</th><th>-0.18**</th>",
    )
    assert match_table_cells(headerless, lines[1:]) == []


@pytest.mark.config({"page_range": [5]})
def test_table_processor(pdf_document, recognition_model):
    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    tables = pdf_document.contained_blocks((BlockTypes.Table,))
    assert len(tables) == 2
    for block in tables:
        assert block.html
        assert "<table" in block.html

    renderer = MarkdownRenderer()
    table_output = renderer(pdf_document)
    assert "Schedule" in table_output.markdown


@pytest.mark.filename("table_ex.pdf")
@pytest.mark.config({"page_range": [0], "force_ocr": True})
def test_avoid_double_ocr(pdf_document, recognition_model):
    tables = pdf_document.contained_blocks((BlockTypes.Table,))
    lines = tables[0].contained_blocks(pdf_document, (BlockTypes.Line,))
    assert len(lines) == 0

    processor = TableProcessor(recognition_model, config={"force_ocr": True})
    processor(pdf_document)

    renderer = MarkdownRenderer()
    table_output = renderer(pdf_document)
    assert "Participants" in table_output.markdown


@pytest.mark.filename("multicol-blocks.pdf")
@pytest.mark.config({"page_range": [3]})
def test_overlap_blocks(pdf_document, recognition_model):
    page = pdf_document.pages[0]
    assert "Cascading, and the Auxiliary Problem Principle" in page.raw_text(
        pdf_document
    )

    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    assert "Cascading, and the Auxiliary Problem Principle" in page.raw_text(
        pdf_document
    )


@pytest.mark.filename("pres.pdf")
@pytest.mark.config({"page_range": [4]})
def test_ocr_table(pdf_document, recognition_model):
    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    renderer = MarkdownRenderer()
    table_output = renderer(pdf_document)
    assert "1.2E-38" in table_output.markdown


@pytest.mark.config({"page_range": [11]})
def test_split_rows(pdf_document, recognition_model):
    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    table = pdf_document.contained_blocks((BlockTypes.Table,))[-1]
    assert table.html
    rows = BeautifulSoup(table.html, "html.parser").find_all("tr")
    assert len(rows) >= 5
