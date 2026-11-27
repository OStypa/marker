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


@pytest.mark.parametrize(
    "rotation,partial_rule", [(90, False), (270, False), (90, True)]
)
def test_rotated_ruled_table_retains_native_groups_and_surrounding_text(
    tmp_path, rotation, partial_rule
):
    import ctypes
    import pypdfium2 as pdfium
    from marker.providers.pdf import PdfProvider

    path = tmp_path / "rotated.pdf"
    with pdfium.PdfDocument.new() as pdf:
        sheet = pdf.new_page(190, 300)
        sheet.set_rotation(rotation)

        def place(obj, x, y):
            matrix = (
                (0, 1, -1, 0, y, x)
                if rotation == 90
                else (0, -1, 1, 0, 190 - y, 300 - x)
            )
            item = pdfium.PdfObject(obj)
            item.set_matrix(pdfium.PdfMatrix(*matrix))
            sheet.insert_obj(item)

        def text(value, x, y):
            obj = pdfium.raw.FPDFPageObj_NewTextObj(pdf, b"Helvetica", 8)
            utf16 = ctypes.create_string_buffer((value + "\0").encode("utf-16le"))
            assert pdfium.raw.FPDFText_SetText(
                obj, ctypes.cast(utf16, pdfium.raw.FPDF_WIDESTRING)
            )
            place(obj, x, y)

        def rule(left, right, y):
            obj = pdfium.raw.FPDFPageObj_CreateNewPath(0, 0)
            assert pdfium.raw.FPDFPath_LineTo(obj, right - left, 0)
            assert pdfium.raw.FPDFPath_SetDrawMode(obj, 0, 1)
            assert pdfium.raw.FPDFPageObj_SetStrokeWidth(obj, 0.6)
            place(obj, left, y)

        text("Appendix", 25, 16)
        text("Table B.2: Sample", 105, 46)
        rule(18, 282, 52)
        text("Measures", 150, 66)
        rule(131 if partial_rule else 112, 279, 72)
        for y, row in (
            (86, ("Period", "Rate", "Level")),
            (104, ("First", "-0.8", "2.10")),
            (122, ("Second", "0.11", "1.20")),
            (140, ("Third", "3.14", "2.30")),
        ):
            for x, word in zip((25, 120, 200), row):
                text(word, x, y)
        rule(18, 282, 153)
        text("Note:", 25, 174)
        text("sample.", 47, 174)
        sheet.gen_content()
        pdf.save(path)

    with PdfProvider(str(path), {"cache_pdftext_pages": True}) as provider:
        page = PageGroup(
            page_id=0,
            polygon=provider.get_page_bbox(0),
            children=[],
            structure=[],
            pdftext_page=provider.raw_pdftext_pages[0],
            lowres_image=provider.get_images([0], 96)[0],
        )
        table = page.add_block(Table, PolygonBox.from_bbox([12, 4, 288, 180]))
        page.add_structure(table)
        native_lines = provider.get_page_lines(0)
        assigned = [line for line in native_lines if line.raw_text.strip() != "sample."]
        assert len(assigned) == len(native_lines) - 1
        page.merge_blocks(assigned, "pdftext")
    original_polygon = table.polygon.polygon
    original_page = page.pdftext_page
    document = Document(filepath=str(path), pages=[page])
    TableProcessor(None, config={"disable_ocr": True})(document)
    cells = table.contained_blocks(document, (BlockTypes.TableCell,))
    assert table.native_table_source["source_table_polygon"] == original_polygon
    assert (
        table.native_table_source["pdftext_page"]["blocks"] == original_page["blocks"]
    )
    assert table.native_table_source["ownership_status"] == "unresolved"
    if partial_rule:
        assert table.polygon.polygon == original_polygon
        assert not any(cell.colspan > 1 for cell in cells)
        return
    assert [page.get_block(ref).block_type for ref in page.structure] == [
        BlockTypes.Text,
        BlockTypes.Caption,
        BlockTypes.Table,
        BlockTypes.Text,
    ]
    assert [
        page.get_block(ref).raw_text(document)
        for ref in (page.structure[0], page.structure[1], page.structure[-1])
    ] == ["Appendix", "Table B.2: Sample", "Note: sample."]
    assert 20 < table.polygon.bbox[1] < 46 and 153 <= table.polygon.bbox[3] < 170
    assert [
        (cell.row_id, cell.col_id, cell.colspan)
        for cell in cells
        if cell.text == "Measures"
    ] == [(0, 1, 2)]
    owner = [
        cell
        for cell in cells
        if cell.polygon.bbox[0] < 128 < cell.polygon.bbox[2]
        and cell.polygon.bbox[1] < 101 < cell.polygon.bbox[3]
    ]
    assert [(cell.row_id, cell.col_id, cell.text) for cell in owner] == [(2, 1, "-0.8")]


@pytest.mark.parametrize("ruled, counts", [(True, False), (True, True), (False, False)])
def test_captioned_native_crop_preserves_the_existing_text(ruled, counts):
    from PIL import Image, ImageDraw

    def native_line(text, x, y):
        bbox = [x, y, x + 4 * len(text), y + 7]
        return {
            "bbox": bbox,
            "spans": [
                {
                    "text": text,
                    "chars": [
                        {"char": char, "bbox": [x + 4 * i, y, x + 4 * i + 4, y + 7]}
                        for i, char in enumerate(text)
                    ],
                }
            ],
        }

    image = Image.new("RGB", (280, 200), "white")
    if ruled:
        draw = ImageDraw.Draw(image)
        for y in (40, 59, 120):
            draw.line((30, y, 250, y), fill="black")
    lines = [
        native_line("Table 2: Estimates", 102, 16),
        native_line("A long description of the outcome by demographic group", 30, 27),
        native_line("Outcome", 36, 46),
        native_line("First", 140, 46),
        native_line("Second", 198, 46),
    ]
    for number, y in enumerate((66, 82, 98)):
        lines.extend(
            [
                native_line("Variable", 36, y),
                native_line(str(31 + number) if counts else f"0.{number + 1}", 144, y),
                native_line(str(41 + number) if counts else f"0.{number + 2}", 200, y),
            ]
        )
    lines.append(native_line("Notes: Evidence from example data.", 35, 126))
    native = {"rotation": 0, "blocks": [{"lines": lines}]}
    page_box = PolygonBox.from_bbox([0, 0, 280, 200])
    original = Text(
        page_id=0, block_id=0, polygon=page_box, html="<p>Existing text.</p>"
    )
    page = PageGroup(
        page_id=0,
        block_id=0,
        polygon=page_box,
        children=[original],
        structure=[original.id],
        lowres_image=image,
        pdftext_page=native,
    )
    document = Document(filepath="fixture.pdf", pages=[page])
    processor = TableProcessor(None, {"disable_ocr": True})
    processor(document)

    tables = document.contained_blocks((BlockTypes.Table,))
    assert len(tables) == int(ruled)
    assert "Existing text." in MarkdownRenderer()(document).markdown
    if ruled:
        crop = tables[0]
        assert crop.polygon.y_start < 16 and 120 < crop.polygon.y_end < 126
        assert crop.native_table_source["printed_caption"] == "Table 2: Estimates"
        assert crop.native_table_source["reconstruction_disposition"] == "not_attempted"
        assert crop.native_table_source["pdftext_page"] == native
        assert page.pdftext_page is None
        next_native = {"rotation": 0, "blocks": []}
        next_page = PageGroup(page_id=0, polygon=page_box, pdftext_page=next_native)
        processor(Document(filepath="next.pdf", pages=[next_page]))
        assert next_page.pdftext_page == next_native
    else:
        assert page.pdftext_page == native
