import contextlib
import ctypes
import logging
import re
from typing import Annotated, Dict, List, Optional, Set

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from ftfy import fix_text
from pdftext.extraction import dictionary_output
from pdftext.schema import Reference
from pdftext.pdf.utils import flatten as flatten_pdf_page

from PIL import Image
from pypdfium2 import PdfiumError, PdfDocument

from marker.providers import BaseProvider, ProviderOutput, Char, ProviderPageLines
from marker.providers.utils import alphanum_ratio
from marker.schema import BlockTypes
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class
from marker.schema.text.line import Line
from marker.schema.text.span import Span

# Ignore pypdfium2 warning about form flattening
logging.getLogger("pypdfium2").setLevel(logging.ERROR)


class PdfProvider(BaseProvider):
    """
    A provider for PDF files.
    """

    page_range: Annotated[
        List[int],
        "The range of pages to process.",
        "Default is None, which will process all pages.",
    ] = None
    pdftext_workers: Annotated[
        int,
        "The number of workers to use for pdftext.",
    ] = 4
    flatten_pdf: Annotated[
        bool,
        "Whether to flatten the PDF structure.",
    ] = True
    force_ocr: Annotated[
        bool,
        "Whether to force OCR on the whole document.",
    ] = False
    ocr_invalid_chars: Annotated[
        tuple,
        "The characters to consider invalid for OCR.",
    ] = (chr(0xFFFD), "�")
    ocr_space_threshold: Annotated[
        float,
        "The minimum ratio of spaces to non-spaces to detect bad text.",
    ] = 0.7
    ocr_newline_threshold: Annotated[
        float,
        "The minimum ratio of newlines to non-newlines to detect bad text.",
    ] = 0.6
    ocr_alphanum_threshold: Annotated[
        float,
        "The minimum ratio of alphanumeric characters to non-alphanumeric characters to consider an alphanumeric character.",
    ] = 0.3
    image_threshold: Annotated[
        float,
        "The minimum coverage ratio of the image to the page to consider skipping the page.",
    ] = 0.65
    strip_existing_ocr: Annotated[
        bool,
        "Whether to strip existing OCR text from the PDF.",
    ] = False
    disable_ocr: Annotated[
        bool,
        "Whether OCR is disabled.",
    ] = False
    disable_links: Annotated[
        bool,
        "Whether to disable links.",
    ] = False
    keep_chars: Annotated[
        bool,
        "Whether to keep character-level information in the output.",
        "Only affects pdftext pages - OCR'd pages have no character data.",
    ] = False
    cache_pdftext_pages: Annotated[
        bool,
        "Cache the raw pdftext pages so table cell text can be assigned",
        "without re-extracting the PDF. Increases memory usage.",
    ] = True

    def __init__(self, filepath: str, config=None):
        super().__init__(filepath, config)

        self.filepath = filepath
        self.raw_pdftext_pages: Dict[int, dict] = {}

        with self.get_doc() as doc:
            self.page_count = len(doc)

            if self.page_range is None:
                self.page_range = range(len(doc))

            assert max(self.page_range) < len(doc) and min(self.page_range) >= 0, (
                f"Invalid page range, values must be between 0 and {len(doc) - 1}.  Min of provided page range is {min(self.page_range)} and max is {max(self.page_range)}."
            )

            self.page_lines: ProviderPageLines = {i: [] for i in self.page_range}
            self.page_refs: Dict[int, List[Reference]] = {
                i: [] for i in self.page_range
            }

            if self.force_ocr:
                # Manually assign page bboxes, since we can't get them from pdftext
                self.page_bboxes = {i: doc[i].get_bbox() for i in self.page_range}
            else:
                self.page_lines = self.pdftext_extraction(doc)

    @contextlib.contextmanager
    def get_doc(self):
        doc = None
        try:
            doc = pdfium.PdfDocument(self.filepath)

            # Must be called on the parent pdf, before retrieving pages to render correctly
            if self.flatten_pdf:
                doc.init_forms()

            yield doc
        finally:
            if doc:
                doc.close()

    def __len__(self) -> int:
        return self.page_count

    def font_flags_to_format(self, flags: Optional[int]) -> Set[str]:
        if flags is None:
            return {"plain"}

        flag_map = {
            1: "FixedPitch",
            2: "Serif",
            3: "Symbolic",
            4: "Script",
            6: "Nonsymbolic",
            7: "Italic",
            17: "AllCap",
            18: "SmallCap",
            19: "ForceBold",
            20: "UseExternAttr",
        }
        set_flags = set()
        for bit_position, flag_name in flag_map.items():
            if flags & (1 << (bit_position - 1)):
                set_flags.add(flag_name)
        if not set_flags:
            set_flags.add("Plain")

        formats = set()
        if set_flags == {"Symbolic", "Italic"} or set_flags == {
            "Symbolic",
            "Italic",
            "UseExternAttr",
        }:
            formats.add("plain")
        elif set_flags == {"UseExternAttr"}:
            formats.add("plain")
        elif set_flags == {"Plain"}:
            formats.add("plain")
        else:
            if set_flags & {"Italic"}:
                formats.add("italic")
            if set_flags & {"ForceBold"}:
                formats.add("bold")
            if set_flags & {
                "FixedPitch",
                "Serif",
                "Script",
                "Nonsymbolic",
                "AllCap",
                "SmallCap",
                "UseExternAttr",
            }:
                formats.add("plain")
        return formats

    def font_names_to_format(self, font_name: str | None) -> Set[str]:
        formats = set()
        if font_name is None:
            return formats

        if "bold" in font_name.lower():
            formats.add("bold")
        if "ital" in font_name.lower():
            formats.add("italic")
        return formats

    @staticmethod
    def normalize_spaces(text):
        space_chars = [
            "\u2003",  # em space
            "\u2002",  # en space
            "\u00a0",  # non-breaking space
            "\u200b",  # zero-width space
            "\u3000",  # ideographic space
        ]
        for space in space_chars:
            text = text.replace(space, " ")
        return text

    def pdftext_extraction(self, doc: PdfDocument) -> ProviderPageLines:
        page_lines: ProviderPageLines = {}
        page_char_blocks = dictionary_output(
            self.filepath,
            page_range=self.page_range,
            # Chars are needed for table cell text assignment when caching
            keep_chars=self.keep_chars or self.cache_pdftext_pages,
            workers=self.pdftext_workers,
            flatten_pdf=self.flatten_pdf,
            quote_loosebox=False,
            disable_links=self.disable_links,
        )
        if self.cache_pdftext_pages:
            self.raw_pdftext_pages = {page["page"]: page for page in page_char_blocks}
        self.page_bboxes = {
            i: [0, 0, page["width"], page["height"]]
            for i, page in zip(self.page_range, page_char_blocks)
        }

        SpanClass: Span = get_block_class(BlockTypes.Span)
        LineClass: Line = get_block_class(BlockTypes.Line)
        CharClass: Char = get_block_class(BlockTypes.Char)

        for page in page_char_blocks:
            page_id = page["page"]
            lines: List[ProviderOutput] = []
            if not self.check_page(page_id, doc):
                continue

            for block in page["blocks"]:
                for line in block["lines"]:
                    spans: List[Span] = []
                    chars: List[List[Char]] = []
                    for span in line["spans"]:
                        if not span["text"]:
                            continue
                        font_formats = self.font_flags_to_format(
                            span["font"]["flags"]
                        ).union(self.font_names_to_format(span["font"]["name"]))
                        font_name = span["font"]["name"] or "Unknown"
                        font_weight = span["font"]["weight"] or 0
                        font_size = span["font"]["size"] or 0
                        polygon = PolygonBox.from_bbox(
                            span["bbox"], ensure_nonzero_area=True
                        )
                        superscript = span.get("superscript", False)
                        subscript = span.get("subscript", False)
                        raw_span_text = span["text"]
                        text = self.normalize_spaces(
                            raw_span_text
                            if raw_span_text.isascii()
                            else fix_text(raw_span_text)
                        )
                        if superscript or subscript:
                            text = text.strip()

                        spans.append(
                            SpanClass(
                                polygon=polygon,
                                text=text,
                                font=font_name,
                                font_weight=font_weight,
                                font_size=font_size,
                                minimum_position=span["char_start_idx"],
                                maximum_position=span["char_end_idx"],
                                formats=list(font_formats),
                                page_id=page_id,
                                text_extraction_method="pdftext",
                                url=span.get("url"),
                                has_superscript=superscript,
                                has_subscript=subscript,
                            )
                        )

                        if self.keep_chars:
                            span_chars = [
                                CharClass(
                                    text=c["char"],
                                    polygon=PolygonBox.from_bbox(
                                        c["bbox"], ensure_nonzero_area=True
                                    ),
                                    idx=c["char_idx"],
                                )
                                for c in span["chars"]
                            ]
                            chars.append(span_chars)
                        else:
                            chars.append([])

                    polygon = PolygonBox.from_bbox(
                        line["bbox"], ensure_nonzero_area=True
                    )

                    assert len(spans) == len(chars), (
                        f"Spans and chars length mismatch on page {page_id}: {len(spans)} spans, {len(chars)} chars"
                    )
                    lines.append(
                        ProviderOutput(
                            line=LineClass(polygon=polygon, page_id=page_id),
                            spans=spans,
                            chars=chars,
                        )
                    )
            if self.check_line_spans(lines):
                page_lines[page_id] = lines

            self.page_refs[page_id] = []
            if page_refs := page.get("refs", None):
                self.page_refs[page_id] = page_refs

        return page_lines

    def check_line_spans(self, page_lines: List[ProviderOutput]) -> bool:
        page_spans = [span for line in page_lines for span in line.spans]
        if len(page_spans) == 0:
            return False

        text = "\n".join(f" {span.text}" for span in page_spans) + "\n"
        if len(text.strip()) == 0:
            return False
        if self.detect_bad_ocr(text):
            return False
        return True

    def check_page(self, page_id: int, doc: PdfDocument) -> bool:
        page = doc.get_page(page_id)
        page_bbox = PolygonBox.from_bbox(page.get_bbox())
        try:
            page_objs = list(
                page.get_objects(
                    filter=[pdfium_c.FPDF_PAGEOBJ_TEXT, pdfium_c.FPDF_PAGEOBJ_IMAGE]
                )
            )
        except PdfiumError:
            # Happens when pdfium fails to get the number of page objects
            return False

        text_objs = [
            obj for obj in page_objs if obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT
        ]
        # if we do not see any text objects in the pdf, we can skip this page
        if not text_objs:
            return False

        if self.strip_existing_ocr:
            non_embedded_fonts = []
            empty_fonts = []
            font_cache = {}
            for text_obj in text_objs:
                if pdfium_c.FPDFTextObj_GetTextRenderMode(text_obj) in [
                    pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE,
                    pdfium_c.FPDF_TEXTRENDERMODE_UNKNOWN,
                ]:
                    return False

                font = pdfium_c.FPDFTextObj_GetFont(text_obj)
                font_key = int(ctypes.cast(font, ctypes.c_void_p).value or 0)
                if font_key not in font_cache:
                    font_name = self._get_fontname(font)
                    is_non_embedded = pdfium_c.FPDFFont_GetIsEmbedded(font) == 0
                    is_empty = "glyphless" in font_name.lower()
                    font_cache[font_key] = (is_non_embedded, is_empty)
                else:
                    is_non_embedded, is_empty = font_cache[font_key]

                non_embedded_fonts.append(is_non_embedded)
                empty_fonts.append(is_empty)

            if all(non_embedded_fonts) or all(empty_fonts):
                return False

            # if we see very large images covering most of the page, we can skip this page
            for img_obj in page_objs:
                if img_obj.type != pdfium_c.FPDF_PAGEOBJ_IMAGE:
                    continue
                img_bbox = PolygonBox.from_bbox(img_obj.get_pos())
                if page_bbox.intersection_pct(img_bbox) >= self.image_threshold:
                    return False

        return True

    def detect_bad_ocr(self, text):
        if len(text) == 0:
            # Assume OCR failed if we have no text
            return True

        # Collapse leader runs (TOC dot leaders, form underscores/dashes) before the
        # ratio checks. They are legitimate typography, not garbled OCR, but their low
        # alphanumeric density otherwise trips the heuristic and forces clean pages
        # (e.g. tables of contents) into needless full-page OCR.
        text = re.sub(r"(?:[.·•…_\-]\s*){3,}", " ", text)

        spaces = len(re.findall(r"\s+", text))
        alpha_chars = len(re.sub(r"\s+", "", text))
        if spaces / (alpha_chars + spaces) > self.ocr_space_threshold:
            return True

        newlines = len(re.findall(r"\n+", text))
        non_newlines = len(re.sub(r"\n+", "", text))
        if newlines / (newlines + non_newlines) > self.ocr_newline_threshold:
            return True

        if alphanum_ratio(text) < self.ocr_alphanum_threshold:  # Garbled text
            return True

        invalid_chars = len([c for c in text if c in self.ocr_invalid_chars])
        if invalid_chars > max(6.0, len(text) * 0.03):
            return True

        return False

    @staticmethod
    def _render_image(
        pdf: pdfium.PdfDocument, idx: int, dpi: int, flatten_page: bool
    ) -> Image.Image:
        page = pdf[idx]
        if flatten_page:
            flatten_pdf_page(page)
            page = pdf[idx]
        image = page.render(scale=dpi / 72, draw_annots=False).to_pil()
        image = image.convert("RGB")
        return image

    def get_images(self, idxs: List[int], dpi: int) -> List[Image.Image]:
        with self.get_doc() as doc:
            images = [
                self._render_image(doc, idx, dpi, self.flatten_pdf) for idx in idxs
            ]
        return images

    def image_placements(self, idx: int) -> list[tuple[list[float], bool]]:
        """Return competing image bounds. Only isolated, visible top-level images qualify."""
        with self.get_doc() as doc:
            page = doc[idx]
            left, bottom, width, height = page.get_bbox()
            if page.get_rotation() or left != 0 or bottom != 0:
                return []
            try:
                objects = list(page.get_objects(max_depth=1))
                image_type = pdfium_c.FPDF_PAGEOBJ_IMAGE
                if not any(obj.type == image_type for obj in objects):
                    return []
                bounds = [(obj, obj.get_bounds()) for obj in objects]
                placements = []
                for obj, (x0, y0, x1, y1) in bounds:
                    box = [max(0, x0), max(0, height - y1), min(width, x1), min(height, height - y0)]
                    if box[0] >= box[2] or box[1] >= box[3]:
                        continue
                    if obj.type != image_type:
                        if obj.type == pdfium_c.FPDF_PAGEOBJ_FORM and next(
                            page.get_objects(form=obj, filter=[image_type]), None
                        ) is not None:
                            placements.append((box, False))
                        continue
                    matrix = obj.get_matrix()
                    clip = pdfium_c.FPDFPageObj_GetClipPath(obj)
                    active = pdfium_c.FPDF_BOOL()
                    isolated = (
                        0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
                        and matrix.b == matrix.c == 0 and matrix.a > 0 and matrix.d > 0
                        and not pdfium_c.FPDFPageObj_HasTransparency(obj)
                        and pdfium_c.FPDFClipPath_CountPaths(clip) <= 0
                        and pdfium_c.FPDFPageObj_GetIsActive(obj, ctypes.byref(active)) and active.value
                        and not any(
                            other is not obj and x0 < r and l < x1 and y0 < t and b < y1
                            for other, (l, b, r, t) in bounds
                        )
                    )
                    placements.append((box, bool(isolated)))
                return placements
            except PdfiumError:
                return []

    @staticmethod
    def _has_chart_stroke(page, form):
        """Find a visible open polyline that spans a substantial part of the panel."""
        matrix = form.get_matrix()
        if matrix.b != 0 or matrix.c != 0 or matrix.a <= 0 or matrix.d <= 0:
            return False
        left, bottom, right, top = form.get_bounds()
        for path in page.get_objects(form=form, max_depth=1, filter=[pdfium_c.FPDF_PAGEOBJ_PATH]):
            count = pdfium_c.FPDFPath_CountSegments(path)
            if count < 3:
                continue
            x0, y0, x1, y1 = matrix.on_rect(*path.get_bounds())
            if x1 - x0 < (right - left) / 2 or y1 - y0 < (top - bottom) / 20:
                continue
            fill, stroke, active = ctypes.c_int(), pdfium_c.FPDF_BOOL(), pdfium_c.FPDF_BOOL()
            if not (
                pdfium_c.FPDFPath_GetDrawMode(path, ctypes.byref(fill), ctypes.byref(stroke))
                and stroke.value
                and not fill.value
                and pdfium_c.FPDFPageObj_GetIsActive(path, ctypes.byref(active))
                and active.value
                and not pdfium_c.FPDFPageObj_HasTransparency(path)
                and pdfium_c.FPDFClipPath_CountPaths(pdfium_c.FPDFPageObj_GetClipPath(path)) <= 0
            ):
                continue
            last = pdfium_c.FPDFPath_GetPathSegment(path, count - 1)
            if pdfium_c.FPDFPathSegment_GetClose(last):
                continue
            transform = path.get_matrix()
            if transform.b != 0 or transform.c != 0:
                continue
            previous = None
            slopes = 0
            for i in range(count):
                segment = pdfium_c.FPDFPath_GetPathSegment(path, i)
                expected = pdfium_c.FPDF_SEGMENT_MOVETO if i == 0 else pdfium_c.FPDF_SEGMENT_LINETO
                x, y = ctypes.c_float(), ctypes.c_float()
                if pdfium_c.FPDFPathSegment_GetType(
                    segment
                ) != expected or not pdfium_c.FPDFPathSegment_GetPoint(
                    segment, ctypes.byref(x), ctypes.byref(y)
                ):
                    break
                point = matrix.on_point(*transform.on_point(x.value, y.value))
                if previous:
                    slopes += abs(point[0] - previous[0]) >= 1 and abs(point[1] - previous[1]) >= 1
                previous = point
            else:
                if slopes >= 2:
                    return True
        return False

    def lettered_chart_bounds(self, idx: int, title: str):
        """Check a native title followed by consecutive lettered panels in stacked PDF forms."""
        with self.get_doc() as doc:
            page = doc[idx]
            left, bottom, width, height = page.get_bbox()
            if page.get_rotation() or left != 0 or bottom != 0:
                return None
            try:
                textpage = page.get_textpage()
                objects = list(page.get_objects(max_depth=1))
                forms = [
                    i for i, obj in enumerate(objects) if obj.type == pdfium_c.FPDF_PAGEOBJ_FORM
                ]
                if not 2 <= len(forms) <= 26 or forms[0] < 2:
                    return None
                first = forms[0] - 2
                if forms != list(range(first + 2, forms[-1] + 1, 2)):
                    return None
                boxes = []
                labels = []
                for i, obj in enumerate(objects[first : forms[-1] + 1]):
                    x0, y0, x1, y1 = obj.get_bounds()
                    active = pdfium_c.FPDF_BOOL()
                    if not (
                        0 <= x0 < x1 <= width
                        and 0 <= y0 < y1 <= height
                        and pdfium_c.FPDFPageObj_GetIsActive(obj, ctypes.byref(active))
                        and active.value
                        and not pdfium_c.FPDFPageObj_HasTransparency(obj)
                        and pdfium_c.FPDFClipPath_CountPaths(pdfium_c.FPDFPageObj_GetClipPath(obj))
                        <= 0
                    ):
                        return None
                    box = [x0, height - y1, x1, height - y0]
                    if i == 0 or i % 2:
                        if (
                            obj.type != pdfium_c.FPDF_PAGEOBJ_TEXT
                            or pdfium_c.FPDFTextObj_GetTextRenderMode(obj) not in (0, 1, 2)
                        ):
                            return None
                        obj.textpage = textpage
                        printed = " ".join(obj.extract().split())
                        if i == 0:
                            if printed != " ".join(title.split()):
                                return None
                            caption = box
                            continue
                        letter = chr(ord("a") + len(labels))
                        if not re.match(rf"^\({letter}\)(?:\s|$)", printed, re.I):
                            return None
                        labels.append(box)
                        continue
                    label = labels[-1]
                    line_height = label[3] - label[1]
                    previous = boxes[-1] if boxes else caption
                    if not (
                        box[0] <= label[0] < label[2] <= box[2]
                        and abs(label[1] - box[1]) <= 2 * line_height
                        and 0 <= box[1] - previous[3] <= 3 * line_height
                        and box[0] < (previous[0] + previous[2]) / 2 < box[2]
                        and self._has_chart_stroke(page, obj)
                    ):
                        return None
                    boxes.append(box)
                panels = boxes + labels
                figure = [
                    min(b[0] for b in panels),
                    min(b[1] for b in panels),
                    max(b[2] for b in panels),
                    max(b[3] for b in panels),
                ]
                # Text geometry in layout can overshoot a form. Require exclusive use of this page band.
                for obj in objects[:first] + objects[forms[-1] + 1 :]:
                    _, y0, _, y1 = obj.get_bounds()
                    if height - y1 < figure[3] and height - y0 > caption[1]:
                        return None
                return caption, figure
            except PdfiumError:
                return None

    def get_page_bbox(self, idx: int) -> PolygonBox | None:
        bbox = self.page_bboxes.get(idx)
        if bbox:
            return PolygonBox.from_bbox(bbox)

    def get_page_lines(self, idx: int) -> List[ProviderOutput]:
        return self.page_lines[idx]

    def get_page_refs(self, idx: int) -> List[Reference]:
        return self.page_refs[idx]

    @staticmethod
    def _get_fontname(font) -> str:
        font_name = ""
        buffer_size = 256

        try:
            font_name_buffer = ctypes.create_string_buffer(buffer_size)
            length = pdfium_c.FPDFFont_GetBaseFontName(
                font, font_name_buffer, buffer_size
            )
            if length < buffer_size:
                font_name = font_name_buffer.value.decode("utf-8")
            else:
                font_name_buffer = ctypes.create_string_buffer(length)
                pdfium_c.FPDFFont_GetBaseFontName(font, font_name_buffer, length)
                font_name = font_name_buffer.value.decode("utf-8")
        except Exception:
            pass

        return font_name
