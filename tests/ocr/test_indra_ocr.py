import tempfile
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.cpu

from marker.builders.ocr import OcrBuilder
from marker.converters.pdf import PdfConverter
import marker.models  # noqa: F401 - ensures Surya cache dir redirect in read-only HOME
from marker.models import create_model_dict
from marker.ocr.indra import (
    IndraRecognitionPredictor,
    _recover_captioned_tables,
    _recover_embedded_figures,
    _recover_lettered_chart,
    select_recognition_model,
)
from marker.output import text_from_rendered
from marker.processors.equation import EquationProcessor
from marker.processors.table import TableProcessor
from surya.layout.schema import LayoutBox, LayoutResult

DATA_DIR = Path(__file__).parent.parent / "data" / "olmocr_bench" / "pdfs"


def test_only_an_isolated_chart_with_one_printed_title_becomes_a_figure(tmp_path):
    import pypdfium2 as pdfium
    from marker.providers.pdf import PdfProvider

    path = tmp_path / "charts.pdf"
    with pdfium.PdfDocument.new() as pdf:
        page = pdf.new_page(250, 400)
        bitmap = pdfium.PdfBitmap.from_pil(Image.new("RGB", (40, 40), "black"))
        for left, bottom, width, height in (
            (20, 310, 80, 60), (140, 310, 80, 60),  # Two panels share the first title.
            (150, 320, 10, 10),  # The right panel still competes when an object covers it.
            (20, 175, 200, 60),  # The second title has one image.
            (20, 40, 200, 60), (30, 50, 25, 30),  # A separate object covers the image.
        ):
            picture = pdfium.PdfImage.new(pdf)
            picture.set_bitmap(bitmap)
            picture.set_matrix(pdfium.PdfMatrix(width, 0, 0, height, left, bottom))
            page.insert_obj(picture)
        page.gen_content()
        pdf.save(path)

    blocks = [
        {"label": "Text", "text": f"Figure A{index}: Earnings", "bbox": [20, y, 220, y + 10]}
        for index, y in enumerate((10, 145, 280), start=1)
    ]
    with PdfProvider(str(path)) as provider:
        _recover_embedded_figures(blocks, provider, 0, 10)
    assert [(block["label"], block.get("text")) for block in blocks] == [
        ("Text", "Figure A1: Earnings"), ("Caption", "Figure A2: Earnings"),
        ("Figure", None), ("Text", "Figure A3: Earnings"),
    ]
    assert blocks[2]["bbox"] == [20, 165, 220, 225]
    detected = {"label": "Figure", "bbox": blocks[2]["bbox"]}
    blocks[2] = detected
    with PdfProvider(str(path)) as provider:
        _recover_embedded_figures(blocks, provider, 0, 10)
    assert [block for block in blocks if block["label"] == "Figure"] == [detected]


@pytest.mark.parametrize(
    "case", ["chart", "ruled_table", "rotated_rule", "missing_panel", "other_column"]
)
def test_lettered_vector_panels_need_one_source_figure(tmp_path, case):
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    from marker.providers.pdf import PdfProvider

    path = tmp_path / "vector-panels.pdf"
    title = "Figure A.2: Sample outcomes"
    with pdfium.PdfDocument.new() as pdf, pdfium.PdfDocument.new() as panels:
        page = pdf.new_page(280, 400)

        def add_text(value, x, y):
            obj = pdfium.PdfObject(raw.FPDFPageObj_NewTextObj(pdf, b"Helvetica", 9), pdf=pdf)
            chars = (raw.FPDF_WCHAR * (len(value) + 1))(*map(ord, value), 0)
            assert raw.FPDFText_SetText(obj, chars)
            obj.set_matrix(pdfium.PdfMatrix().translate(x, y))
            page.insert_obj(obj)

        add_text(title, 55, 337)
        labels = ("(a) Sample", "(c) Sample" if case == "missing_panel" else "(b) Sample")
        for i, y in enumerate((240, 140)):
            panel = panels.new_page(180, 90)
            backdrop = pdfium.PdfObject(raw.FPDFPageObj_CreateNewRect(0, 0, 180, 90), pdf=panels)
            raw.FPDFPageObj_SetFillColor(backdrop, 255, 255, 255, 255)
            raw.FPDFPath_SetDrawMode(backdrop, raw.FPDF_FILLMODE_WINDING, 0)
            panel.insert_obj(backdrop)
            line = pdfium.PdfObject(raw.FPDFPageObj_CreateNewPath(8, 15), pdf=panels)
            points = [(50, 60), (100, 25), (170, 75)]
            if case in {"ruled_table", "rotated_rule"} and i == 1:
                points = [(170, 15), (170, 75), (8, 75)]
            for x1, y1 in points:
                raw.FPDFPath_LineTo(line, x1, y1)
            raw.FPDFPageObj_SetStrokeColor(line, 0, 0, 0, 255)
            raw.FPDFPath_SetDrawMode(line, raw.FPDF_FILLMODE_NONE, 1)
            panel.insert_obj(line)
            panel.gen_content()
            add_text(labels[i], 75, y + 80)
            form = panels.page_as_xobject(i, pdf).as_pageobject()
            if case == "rotated_rule" and i == 1:
                form.set_matrix(pdfium.PdfMatrix(1, 0.035, -0.035, 1, 30, y))
            else:
                form.set_matrix(pdfium.PdfMatrix().translate(30, y))
            page.insert_obj(form)
        add_text("Table 1: Prices", 45, 70)
        if case == "other_column":
            add_text("Table 2: Sales", 215, 260)
        page.gen_content()
        pdf.save(path)

    caption = {"label": "SectionHeader", "text": title + "\n(a) Sample", "bbox": [55, 56, 170, 82]}
    outside_table = {"label": "Table", "bbox": [45, 310, 170, 350], "text": "Table 1: Prices"}
    blocks = [
        caption,
        {"label": "Table", "text": "1 2", "bbox": [31, 90, 260, 145]},
        {"label": "Text", "text": "(b) Sample", "bbox": [75, 171, 120, 183]},
        {"label": "Table", "text": "2 4", "bbox": [31, 189, 260, 250]},
        outside_table,
    ]
    with PdfProvider(str(path), {"pdftext_workers": 1}) as provider:
        if case == "rotated_rule":
            with provider.get_doc() as source:
                source_page = source[0]
                forms = list(source_page.get_objects(max_depth=1, filter=[raw.FPDF_PAGEOBJ_FORM]))
                assert not provider._has_chart_stroke(source_page, forms[-1])
        _recover_lettered_chart(blocks, provider, 0, 9)
        assert blocks[-1] is outside_table
        if case != "chart":
            assert [b["label"] for b in blocks] == [
                "SectionHeader",
                "Table",
                "Text",
                "Table",
                "Table",
            ]
            return
        assert [b["label"] for b in blocks] == ["Caption", "Figure", "Table"]
        assert caption["text"] == title
        assert blocks[1]["bbox"] == pytest.approx([30, 70, 210, 260])

        from marker.builders.document import DocumentBuilder
        from marker.builders.layout import LayoutBuilder
        from marker.schema.polygon import PolygonBox

        document = DocumentBuilder({"disable_ocr": True}).build_document(provider)
        page = document.pages[0]
        result = LayoutResult(
            image_bbox=page.polygon.bbox,
            bboxes=[
                LayoutBox(
                    polygon=PolygonBox.from_bbox(blocks[1]["bbox"]).polygon,
                    label=blocks[1]["label"],
                    raw_label=blocks[1]["raw_label"],
                    position=0,
                )
            ],
        )
        layout = LayoutBuilder()
        layout.add_blocks_to_pages([page], [result])
        layout.expand_layout_blocks(document)
        line = next(line for line in provider.get_page_lines(0) if line.raw_text.startswith("(a)"))
        line.line.polygon = PolygonBox.from_bbox([45, 80, 265, 90])
        page.merge_blocks([line], "pdftext")
        figure = page.get_block(page.structure[0])
        assert figure.polygon.bbox == pytest.approx(blocks[1]["bbox"])
        assert figure.raw_text(document).strip() == "(a) Sample"


@pytest.mark.parametrize(
    "case",
    [
        "table",
        "page_numbers",
        "contents_title",
        "other_column",
        "two_titles",
        "combined_titles",
        "two_blocks",
        "intervening_text",
    ],
)
def test_numbered_caption_needs_one_table_with_measured_values(case):
    title = {
        "label": "SectionHeader",
        "raw_label": "Section-Header",
        "text": "TABLE 1 -- Sample means",
        "bbox": [40, 10, 135, 20],
    }
    body = {
        "label": "TableOfContents",
        "raw_label": "Table-Of-Contents",
        "text": "",
        "bbox": [10, 45, 180, 115],
    }
    blocks = [title, body]
    if case == "contents_title":
        blocks.append(
            {
                "label": "SectionHeader",
                "text": "List of Tables",
                "bbox": [10, 0, 100, 8],
            }
        )
    elif case == "other_column":
        title["bbox"] = [210, 10, 305, 20]
    elif case == "two_titles":
        blocks.append(
            {**title, "text": "Table 2: Other means", "bbox": [35, 22, 130, 31]}
        )
    elif case == "combined_titles":
        title["text"] += "\nTable 2: Other means"
    elif case == "two_blocks":
        blocks.append({**body, "label": "Table", "bbox": [10, 35, 180, 112]})
    elif case == "intervening_text":
        blocks.append(
            {
                "label": "Text",
                "text": "A separate paragraph.",
                "bbox": [10, 23, 164, 33],
            }
        )
    values = ["1", "12", "23"] if case == "page_numbers" else ["1.35", "12.7", "23.8"]
    lines = []
    for index, value in enumerate(values):
        y = 49 + index * 23
        lines.append(
            {
                "bbox": [12, y, 170, y + 10],
                "spans": [
                    {"text": "Group", "bbox": [12, y, 45, y + 10]},
                    {"text": value, "bbox": [149, y, 170, y + 10]},
                ],
            }
        )
    _recover_captioned_tables(blocks, {"blocks": [{"lines": lines}]}, line_height=10)
    assert body["label"] == ("Table" if case == "table" else "TableOfContents")
    assert title["label"] == ("Caption" if case == "table" else "SectionHeader")
    assert body["raw_label"] == "Table-Of-Contents"


def _make_text_image(text: str = "Indra OCR High Performance Engine", size=(800, 200)) -> Image.Image:
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 60), text, fill="black", font_size=32)
    return img


def test_select_recognition_model():
    surya_sentinel = object()
    indra_sentinel = object()

    # Default (balanced mode) uses surya recognition_model for both full_page=True and False
    assert select_recognition_model(surya_sentinel, indra_sentinel, mode="balanced") is surya_sentinel
    assert select_recognition_model(surya_sentinel, indra_sentinel, mode="balanced", full_page=False) is surya_sentinel

    # Fast and indra modes select indra_model
    assert select_recognition_model(surya_sentinel, indra_sentinel, mode="fast") is indra_sentinel
    assert select_recognition_model(surya_sentinel, indra_sentinel, mode="indra") is indra_sentinel

    # Explicit ocr_backend overrides mode
    assert (
        select_recognition_model(
            surya_sentinel, indra_sentinel, mode="balanced", ocr_backend="indra"
        )
        is indra_sentinel
    )
    assert (
        select_recognition_model(
            surya_sentinel, indra_sentinel, mode="fast", ocr_backend="surya"
        )
        is surya_sentinel
    )


def test_indra_predictor_full_page():
    predictor = IndraRecognitionPredictor()
    img = _make_text_image("Marker with Indra OCR full page test")
    results = predictor(images=[img], full_page=True)
    assert len(results) == 1
    assert len(results[0].blocks) >= 1
    combined_html = " ".join(b.html for b in results[0].blocks)
    assert "Indra" in combined_html or "Marker" in combined_html


def test_indra_predictor_block_modes():
    predictor = IndraRecognitionPredictor()
    img = _make_text_image("Section 1: Introduction to Indra", size=(800, 300))

    layout = LayoutResult(
        image_bbox=[0, 0, 800, 300],
        bboxes=[
            LayoutBox(
                polygon=[[20, 40], [780, 40], [780, 140], [20, 140]],
                label="SectionHeader",
                raw_label="SectionHeader",
                position=0,
            ),
        ],
    )
    results = predictor(images=[img], layout_results=[layout], full_page=False)
    assert len(results) == 1
    assert len(results[0].blocks) == 1
    block = results[0].blocks[0]
    assert block.label == "SectionHeader"
    assert block.html.startswith("<h")


def test_builder_and_processors_prefer_indra_in_fast_mode():
    surya_sentinel = object()
    indra_predictor = IndraRecognitionPredictor()

    ocr_builder = OcrBuilder(
        surya_sentinel, config={"mode": "fast"}, indra_model=indra_predictor
    )
    assert ocr_builder.get_recognition_model() is indra_predictor

    table_proc = TableProcessor(
        surya_sentinel, config={"mode": "indra"}, indra_model=indra_predictor
    )
    assert table_proc.get_recognition_model() is indra_predictor

    eq_proc = EquationProcessor(
        surya_sentinel, config={"ocr_backend": "indra"}, indra_model=indra_predictor
    )
    assert eq_proc.get_recognition_model() is indra_predictor


def test_pdf_converter_indra_mode_digital_and_forced_ocr():
    pdf_path = DATA_DIR / "headers_footers_page1.pdf"
    artifact_dict = create_model_dict()

    # 1. Mode 1: no_ocr mode (pure text layer + lightweight Indra layout, zero OCR)
    converter_no_ocr = PdfConverter(
        artifact_dict=artifact_dict,
        config={
            "mode": "no_ocr",
            "disable_image_extraction": True,
            "disable_tqdm": True,
            "pdftext_workers": 1,
        },
    )
    assert converter_no_ocr.config["disable_ocr"] is True
    md_no_ocr, _, _ = text_from_rendered(converter_no_ocr(str(pdf_path)))
    assert len(md_no_ocr.strip()) > 100

    # 2. Mode 2: indra / fast mode (line-level Indra ONNX OCR)
    converter = PdfConverter(
        artifact_dict=artifact_dict,
        config={
            "mode": "indra",
            "disable_image_extraction": True,
            "disable_tqdm": True,
            "pdftext_workers": 1,
        },
    )
    md, _, _ = text_from_rendered(converter(str(pdf_path)))
    assert len(md.strip()) > 100

    # 3. Full-page Indra OCR on image input
    img = _make_text_image("Standalone image converted via Indra OCR")
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        img.save(tf.name)
        tf.flush()
        md_img, _, _ = text_from_rendered(converter(tf.name))
        assert "Indra" in md_img or "Standalone" in md_img
