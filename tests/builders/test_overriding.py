import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from surya.layout.schema import LayoutBox, LayoutResult

from marker.converters.pdf import PdfConverter
from marker.providers import BaseProvider, ProviderOutput
from marker.renderers.markdown import MarkdownRenderer

from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.blocks import SectionHeader
from marker.schema.document import Document
from marker.schema.polygon import PolygonBox
from marker.schema.registry import BLOCK_REGISTRY, get_block_class, register_block_class
from marker.schema.text import Line, Span


class NewSectionHeader(SectionHeader):
    pass


class NewLine(Line):
    pass


@pytest.mark.config({
    "page_range": [0],
    "override_map": {BlockTypes.SectionHeader: NewSectionHeader}
})
def test_overriding(pdf_document: Document):
    assert pdf_document.pages[0]\
        .get_block(pdf_document.pages[0].structure[0]).__class__ == NewSectionHeader


def get_lines(pdf: str, config=None):
    from tests.utils import setup_pdf_provider

    for block_type, block_cls in config["override_map"].items():
        register_block_class(block_type, block_cls)

    provider: PdfProvider = setup_pdf_provider(pdf, config)
    return provider.get_page_lines(0)


def test_overriding_mp(pdf_dataset):
    config = {
        "page_range": [0],
        "override_map": {BlockTypes.Line: NewLine}
    }

    pdf_list = ["adversarial.pdf", "adversarial_rot.pdf"]

    with mp.Pool(processes=2) as pool:
        results = pool.starmap(get_lines, [(pdf, config) for pdf in pdf_list])
        assert all([r[0].line.__class__ == NewLine for r in results])


class PublicLine(Line):
    pass


def test_converter_scopes_overrides_during_build_and_render(monkeypatch):
    page_box = PolygonBox.from_bbox([0, 0, 100, 100])
    line_box = PolygonBox.from_bbox([10, 10, 80, 20])
    overlap = Barrier(2)
    prose = "The baseline calculation preserves the complete source sentence."

    class Provider(BaseProvider):
        page_range = [0]

        def __init__(self, filepath, config):
            super().__init__(filepath, config)
            if filepath.startswith("parallel"):
                overlap.wait(timeout=5)
            self.page_lines = {
                0: [
                    ProviderOutput(
                        line=get_block_class(BlockTypes.Line)(page_id=0, polygon=line_box),
                        spans=[
                            Span(
                                page_id=0,
                                polygon=line_box,
                                text=prose,
                                font="Test",
                                font_weight=400,
                                font_size=10,
                                minimum_position=0,
                                maximum_position=len(prose),
                                formats=["plain"],
                            )
                        ],
                    )
                ]
            }

        def get_images(self, idxs, dpi):
            images = []
            for _ in idxs:
                image = Image.new("RGB", (100, 100), "white")
                ImageDraw.Draw(image).text((12, 10), "Native text", fill="black")
                images.append(image)
            return images

        def get_page_bbox(self, idx):
            return page_box

        def get_page_refs(self, idx):
            return []

    class Layout:
        def __call__(self, images):
            box = LayoutBox(
                polygon=page_box.polygon,
                label="Text",
                raw_label="Text",
                position=0,
                count=0,
            )
            return [
                LayoutResult(bboxes=[box], image_bbox=page_box.bbox) for _ in images
            ]

    class ErrorPredictor:
        calls = 0

        def __call__(self, texts, *, batch_size):
            self.calls += 1
            assert texts == [prose]
            return SimpleNamespace(labels=["good"], scores=[0.4])

    class Recognition:
        def __call__(self, *args, **kwargs):
            pytest.fail("Clean text reached OCR.")

    class Renderer(MarkdownRenderer):
        def __call__(self, document):
            assert get_block_class(BlockTypes.Line) is NewLine
            nested = plain.build_document("nested.memory")
            assert actual_line_type(nested) is PublicLine
            assert get_block_class(BlockTypes.Line) is NewLine
            return super().__call__(document)

    def actual_line_type(document):
        (line,) = document.contained_blocks((BlockTypes.Line,))
        return type(line)

    monkeypatch.setattr("marker.converters.download_font", lambda: None)
    monkeypatch.setattr("marker.converters.pdf.provider_from_filepath", lambda _: Provider)
    monkeypatch.setitem(BLOCK_REGISTRY, BlockTypes.Line, BLOCK_REGISTRY[BlockTypes.Line])
    register_block_class(BlockTypes.Line, PublicLine)

    def converter(**config):
        return PdfConverter(
            artifact_dict={
                "layout_model": Layout(),
                "indra_model": object(),
                "ocr_error_model": ErrorPredictor(),
                "recognition_model": Recognition(),
            },
            processor_list=[],
            config={"mode": "balanced", "ocr_full_page": False, **config},
        )

    custom = converter(override_map={BlockTypes.Line: NewLine})
    plain = converter()
    assert get_block_class(BlockTypes.Line) is PublicLine

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(custom.build_document, "parallel-custom.memory")
        second = pool.submit(plain.build_document, "parallel-plain.memory")
        assert actual_line_type(first.result(timeout=10)) is NewLine
        assert actual_line_type(second.result(timeout=10)) is PublicLine

    assert custom.artifact_dict["ocr_error_model"].calls == 2
    assert plain.artifact_dict["ocr_error_model"].calls == 2

    custom.renderer = Renderer
    assert prose in custom("render.memory").markdown
    assert get_block_class(BlockTypes.Line) is PublicLine
