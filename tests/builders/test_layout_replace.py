from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from surya.layout.schema import LayoutBox, LayoutResult

from marker.builders.document import DocumentBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.providers import ProviderOutput
from marker.renderers.markdown import MarkdownRenderer
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class
from marker.schema.text.line import Line
from marker.schema.text.span import Span


@pytest.mark.filename("thinkpython.pdf")
@pytest.mark.config({"page_range": [0]})
def test_layout_replace(
    request, config, doc_provider, layout_model, fast_layout_model, ocr_error_model
):
    # The llm layout builder replaces blocks - this makes sure text is still merged properly
    layout_builder = LayoutBuilder(layout_model, fast_layout_model, config)
    line_builder = LineBuilder(ocr_error_model, config)
    builder = DocumentBuilder(config)
    document = builder.build_document(doc_provider)
    layout_builder(document, doc_provider)
    page = document.pages[0]
    new_blocks = []
    for block in page.contained_blocks(document, (BlockTypes.Text,)):
        generated_block_class = get_block_class(BlockTypes.TextInlineMath)
        generated_block = generated_block_class(
            polygon=block.polygon,
            page_id=block.page_id,
            structure=block.structure,
        )
        page.replace_block(block, generated_block)
        new_blocks.append(generated_block)
    line_builder(document, doc_provider)

    for block in new_blocks:
        assert block.raw_text(document).strip()

    renderer = MarkdownRenderer(config)
    rendered = renderer(document)

    assert "Think Python" in rendered.markdown


@pytest.mark.parametrize("table_evidence", [None, "rule", "caption"])
def test_proof_text_keeps_native_lines_when_layout_calls_it_a_table(
    monkeypatch, table_evidence
):
    def line(text, y, font="Times-Roman"):
        box = PolygonBox.from_bbox([24, y, 165, y + 8])
        return ProviderOutput(
            line=Line(page_id=0, polygon=box),
            spans=[
                Span(
                    text=text,
                    font=font,
                    font_size=10,
                    font_weight=400,
                    minimum_position=y,
                    maximum_position=y + len(text),
                    formats=["plain"],
                    page_id=0,
                    polygon=box,
                )
            ],
        )

    image = Image.new("RGB", (220, 150), "white")
    lines = [
        line("Proof of Proposition 1: Add the terms.", 42),
        line("∑ x = ∑ y", 66, "CMSY10"),
        line("This gives the claimed result.", 92),
    ]
    if table_evidence == "rule":
        ImageDraw.Draw(image).line((20, 57, 190, 57), fill="black")
    if table_evidence == "caption":
        lines.append(line("Table 2: Example values", 15))
    page = PageGroup(
        page_id=0,
        polygon=PolygonBox.from_bbox([0, 0, 220, 150]),
        lowres_image=image,
        pdftext_page={"rotation": 0, "blocks": []},
    )
    result = LayoutResult(
        image_bbox=page.polygon.bbox,
        bboxes=[
            LayoutBox(
                polygon=PolygonBox.from_bbox([20, 40, 192, 104]).polygon,
                label="Table",
                raw_label="Table",
                position=0,
            )
        ],
    )
    document = Document(filepath="fixture.pdf", pages=[page])
    provider = SimpleNamespace(page_lines={0: lines})
    builder = LayoutBuilder(config={"disable_ocr": True})
    monkeypatch.setattr(builder, "surya_layout", lambda *_: [result])
    builder(document, provider)
    if table_evidence:
        assert page.structure[0].block_type == BlockTypes.Table
    else:
        assert page.structure[0].block_type == BlockTypes.Text
        page.merge_blocks(lines, "pdftext")
        rendered = MarkdownRenderer({"extract_images": False})(document).markdown
        assert "Add the terms." in rendered
        assert "∑ x = ∑ y" in rendered
        assert "This gives the claimed result." in rendered
