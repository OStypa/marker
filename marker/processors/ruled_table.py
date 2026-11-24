"""Read a quarter-turned table with native text and visible horizontal rules."""

import re
from html import escape
from itertools import pairwise
from statistics import median

import numpy as np

from marker.ocr.indra import _TABLE_CAPTION_RE
from marker.processors.table_recon import (
    _TABLE_VALUE,
    _garble_ok,
    _table_row_bands,
    table_lines_from_pdftext,
)
from marker.schema import BlockTypes
from marker.schema.blocks.tablecell import TableCell
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class


def _horizontal_rules(page, region):
    """Find continuous horizontal strokes on the displayed page."""
    image = page.get_image(highres=False)
    if image is None:
        return []
    sx, sy = (n / size for n, size in zip(image.size, page.polygon.size))
    left, top, right, bottom = region
    width = right - left
    start = max(0, int(top * sy))
    stop = min(image.height, int(bottom * sy) + 1)
    if start >= stop or width <= 0:
        return []
    pixels = np.asarray(image.crop((0, start, image.width, stop)).convert("L"))
    rules = []
    for y, row in enumerate(pixels, start):
        edges = np.flatnonzero(np.diff(row < 200, prepend=False, append=False))
        runs = edges.reshape(-1, 2)
        for x0, x1 in runs[runs[:, 1] - runs[:, 0] >= width * sx / 10]:
            x0, x1 = x0 / sx, x1 / sx
            if x0 < left - width / 20 or x1 > right + width / 20:
                continue
            for rule in reversed(rules):
                if (
                    abs(rule[3] - y / sy) <= 0.001
                    and abs(rule[0] - x0) <= 1 / sx
                    and abs(rule[2] - x1) <= 1 / sx
                ):
                    rule[3] = (y + 1) / sy
                    break
            else:
                rules.append([x0, y / sy, x1, (y + 1) / sy])
    return [rule for rule in rules if rule[3] - rule[1] <= 2]


def _join_words(words, height):
    """Join adjacent native words. Column gaps must exceed ordinary word spaces."""
    groups = []
    for text, left, right in words:
        if groups and left - groups[-1][2] <= height * 0.65:
            previous, start, end = groups.pop()
            groups.append((previous + " " + text, start, max(end, right)))
        else:
            groups.append((text, left, right))
    return groups


def _ruled_rows(bands, rules, bounds):
    rows = [
        (top, bottom, _join_words(words, bottom - top)) for top, bottom, words in bands
    ]
    data = [
        i
        for i, (_, _, row) in enumerate(rows)
        if len(row) >= 3
        and any(c.isalpha() for c in row[0][0])
        and all(_TABLE_VALUE.fullmatch(word) for word, _, _ in row[1:])
    ]
    if len(data) < 3:
        return []
    count = len(rows[data[0]][2])
    if any(len(rows[i][2]) != count for i in data):
        return []
    anchors = [
        median((rows[i][2][col][1] + rows[i][2][col][2]) / 2 for i in data)
        for col in range(1, count)
    ]
    headers = [
        i
        for i, (_, _, row) in enumerate(rows[: data[0]])
        if len(row) == count
        and all(any(c.isalpha() for c in word) for word, _, _ in row)
    ]
    if len(headers) != 1:
        return []
    header = headers[0]
    if len({word for word, _, _ in rows[header][2][1:]}) != count - 1:
        return []
    regular = {header, *data}
    for i in regular:
        if not all(
            left <= anchor <= right
            for (_, left, right), anchor in zip(rows[i][2][1:], anchors)
        ):
            return []
    extents = [
        (min(rows[i][2][j][1] for i in regular), max(rows[i][2][j][2] for i in regular))
        for j in range(count)
    ]
    if any(left[1] >= right[0] for left, right in pairwise(extents)):
        return []
    cuts = [bounds[0], *((a[1] + b[0]) / 2 for a, b in pairwise(extents)), bounds[2]]
    result = []
    for i, (top, bottom, row) in enumerate(rows):
        if i in regular:
            groups = [(j, j + 1, word) for j, (word, _, _) in enumerate(row)]
        else:
            next_top = rows[i + 1][0] if i + 1 < len(rows) else bounds[3]
            underlines = [r for r in rules if bottom < (r[1] + r[3]) / 2 < next_top]
            if not underlines or len(underlines) != len(row):
                return []
            groups = []
            for (text, left, right), rule in zip(row, sorted(underlines)):
                covered = [
                    j
                    for j, (lo, hi) in enumerate(extents)
                    if rule[0] <= lo and hi <= rule[2]
                ]
                if (
                    not covered
                    or not rule[0] <= left < right <= rule[2]
                    or (groups and groups[-1][1] > covered[0])
                ):
                    return []
                if any(
                    rule[0] < hi and lo < rule[2] and j not in covered
                    for j, (lo, hi) in enumerate(extents)
                ):
                    return []
                groups.append((covered[0], covered[-1] + 1, text))
        cells = []
        col = 0
        for first, last, text in groups + [(count, count, "")]:
            while col < first:
                cells.append((col, 1, "", [cuts[col], top, cuts[col + 1], bottom]))
                col += 1
            if first < last:
                cells.append(
                    (first, last - first, text, [cuts[first], top, cuts[last], bottom])
                )
            col = last
        result.append((i not in data, cells))
    return result


def recover_rotated_table(document, page, block):
    """Use a ruled grid only when every native row has an unambiguous layout."""
    if (
        page.pdftext_page.get("rotation") not in (90, 270)
        or block.text_extraction_method != "pdftext"
        or block.id not in (page.structure or [])
        or not block.structure
        or any(ref.block_type != BlockTypes.Line for ref in block.structure)
    ):
        return None
    rules = _horizontal_rules(page, block.polygon.bbox)
    if len(rules) < 3:
        return None
    longest = max(rules, key=lambda rule: rule[2] - rule[0])
    outer = [
        rule
        for rule in rules
        if abs(rule[0] - longest[0]) <= 2 and abs(rule[2] - longest[2]) <= 2
    ]
    if len(outer) != 2:
        return None
    left = min(rule[0] for rule in outer)
    right = max(rule[2] for rule in outer)
    top, bottom = outer[0][1], outer[1][3]
    bounds = [left, top, right, bottom]
    lines = table_lines_from_pdftext(page.pdftext_page, bounds)
    if not _garble_ok(lines):
        return None
    rows = _ruled_rows(_table_row_bands(lines), rules, bounds)
    if not rows:
        return None

    # The layout block can lack words that the complete native page retains.
    detected = block.polygon.bbox
    outside = []
    for native_block in page.pdftext_page["blocks"]:
        for line in native_block["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            if (
                x1 <= detected[0]
                or x0 >= detected[2]
                or y1 <= detected[1]
                or y0 >= detected[3]
            ):
                continue
            if left <= x0 and x1 <= right and top <= y0 and y1 <= bottom:
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            if (
                x0 < right
                and left < x1
                and y0 < bottom
                and top < y1
                or x0 < detected[0] - 0.1
                or x1 > detected[2] + 0.1
                or y0 < detected[1] - 0.1
                or y1 > detected[3] + 0.1
            ):
                return None
            outside.append(([(text, x0, x1)], y0, y1))
    exterior = _table_row_bands(outside)
    captions = [
        i
        for i, (y0, y1, parts) in enumerate(exterior)
        if 0 < top - y1 <= 3 * (y1 - y0)
        and left <= parts[0][1] < parts[-1][2] <= right
        and _TABLE_CAPTION_RE.match(" ".join(text for text, _, _ in parts))
    ]
    if len(captions) != 1:
        return None
    caption = captions[0]
    if any(y0 >= exterior[caption][1] and y1 < top for y0, y1, _ in exterior):
        return None
    crop = PolygonBox.from_bbox(
        [left - 1, exterior[caption][0] - 1, right + 1, bottom + 0.5]
    ).fit_to_bounds(page.polygon.bbox)
    siblings = [document.get_block(ref) for ref in page.structure if ref != block.id]
    recovered_region = crop.merge([block.polygon])
    if any(recovered_region.intersection_area(other.polygon) > 0 for other in siblings):
        return None

    before, after = [], []
    for i, (y0, y1, parts) in enumerate(exterior):
        kind = BlockTypes.Caption if i == caption else BlockTypes.Text
        rect = [parts[0][1], y0, max(end for _, _, end in parts), y1]
        text = " ".join(part for part, _, _ in parts)
        other = page.add_full_block(
            get_block_class(kind)(
                page_id=page.page_id,
                polygon=PolygonBox.from_bbox(rect),
                html=f"<p>{escape(text)}</p>",
                text_extraction_method="pdftext",
            )
        )
        (before if y1 < top else after).append(other.id)
    index = page.structure.index(block.id)
    page.structure[index : index + 1] = [*before, block.id, *after]
    block.polygon = crop
    cells = []
    html = ["<table>"]
    for row_id, (is_header, row) in enumerate(rows):
        html.append("<tr>")
        for column, span, text, box in row:
            cell = TableCell(
                page_id=page.page_id,
                polygon=PolygonBox.from_bbox(box),
                row_id=row_id,
                col_id=column,
                rowspan=1,
                colspan=span,
                is_header=is_header,
                text_lines=[text],
            )
            cells.append(cell)
            tag = "th" if is_header else "td"
            attr = f' colspan="{span}"' if span > 1 else ""
            html.append(f"<{tag}{attr}>{escape(text)}</{tag}>")
        html.append("</tr>")
    html.append("</table>")
    return "".join(html), cells


_LABEL_ONLY = re.compile(
    r"Table\s+(?:[A-Z]?[.-]?\d+(?:\.\d+)*|[IVXLCDM]+)", re.IGNORECASE
)
_NOTES = re.compile(r"(?:Notes?|Sources?)\s*[:.]", re.IGNORECASE)


def _native_lines(pdftext_page):
    for block in pdftext_page.get("blocks", ()):
        for line in block.get("lines", ()):
            box = line.get("bbox")
            text = " ".join(
                "".join(span.get("text", "") for span in line.get("spans", ())).split()
            )
            if box is not None and text:
                yield box, text


def _value_columns(row):
    values = []
    for text, left, right in reversed(row):
        if not _TABLE_VALUE.fullmatch(text.removeprefix("(").removesuffix(")")):
            break
        values.append(((left + right) / 2, "." in text))
    return values


def _aligned_value_columns(bands, allow_counts):
    rows = [_join_words(words, bottom - top) for top, bottom, words in bands]
    records = [_value_columns(row) for row in rows]
    if not any(decimal for row in records for _, decimal in row):
        if not allow_counts:
            return []
        records = [
            values
            for row, values in zip(rows, records)
            if len(row) > len(values) >= 2 and any(char.isalpha() for char in row[0][0])
        ]
    anchors = max(records, key=len, default=[])
    if len(anchors) < 2:
        return []
    aligned = [
        [
            next((i for i, (x, _) in enumerate(anchors) if abs(x - value) <= 3), -1)
            for value, _ in row
        ]
        for row in records
    ]
    for first in range(len(anchors)):
        for second in range(first + 1, len(anchors)):
            if sum(first in row and second in row for row in aligned) >= 3:
                return [x for x, _ in anchors]
    return []


def captioned_ruled_regions(page):
    """Locate printed tables independently of their text reconstruction."""
    native = page.pdftext_page
    if not native or native.get("rotation") not in (None, 0):
        return []
    lines = sorted(_native_lines(native), key=lambda item: item[0][1])
    captions = [
        i
        for i, (_, text) in enumerate(lines)
        if _TABLE_CAPTION_RE.match(text) or _LABEL_ONLY.fullmatch(text)
    ]
    if not captions:
        return []
    rules = _horizontal_rules(page, page.polygon.bbox)
    regions = []
    for index, next_caption in pairwise([*captions, len(lines)]):
        caption, title = lines[index]
        cx0, cy0, cx1, cy1 = caption
        height = cy1 - cy0
        center = (cx0 + cx1) / 2
        top = next(
            (
                rule
                for rule in rules
                if cy1 < rule[1] <= cy1 + 5 * height
                and rule[2] - rule[0] >= 8 * height
                and rule[0] - 1 <= cx0 <= cx1 <= rule[2] + 1
                and abs((rule[0] + rule[2]) / 2 - center) <= height
            ),
            None,
        )
        if top is None:
            continue
        subtitle = [
            (box, text)
            for box, text in lines[index + 1 : next_caption]
            if box[1] < top[1]
        ]
        if len(subtitle) > 2 or any(
            not (
                cy1 <= box[1] < box[3] <= top[1]
                and abs((box[0] + box[2]) / 2 - center) <= height
            )
            for box, _ in subtitle
        ):
            continue

        x0, y0, x1, _ = top
        end = (
            lines[next_caption][0][1]
            if next_caption < len(lines)
            else page.polygon.y_end
        )
        end = next(
            (
                box[1]
                for box, text in lines[index + 1 : next_caption]
                if box[1] > y0 and _NOTES.match(text)
            ),
            end,
        )
        full = [
            rule
            for rule in rules
            if y0 < rule[1] < end and abs(rule[0] - x0) <= 2 and abs(rule[2] - x1) <= 2
        ]
        if not full:
            continue

        bottom = full[-1][3]
        closed = bottom - y0 >= 5 * height
        data_end = bottom if closed else end
        if not closed:
            # An open part must run to the page footer with complete data rows.
            if end != page.polygon.y_end or len(full) < 2:
                continue
            footer = [
                (box, text)
                for box, text in lines[index + 1 : next_caption]
                if box[1] > end - 0.08 * page.polygon.height
            ]
            if len(footer) == 1 and footer[0][1].isdecimal():
                data_end = footer[0][0][1]
        left, right = x0 - 1.5, x1 + 1.5
        bands = _table_row_bands(
            table_lines_from_pdftext(native, [left, y0, right, data_end])
        )
        count_header = (
            closed
            and len(full) >= 2
            and any(
                b < full[0][1]
                and len(row := _join_words(words, b - a)) >= 3
                and not _value_columns(row)
                for a, b, words in bands
            )
        )
        anchors = _aligned_value_columns(bands, count_header)
        if not anchors:
            continue
        if not closed:
            body = [(a, b, words) for a, b, words in bands if a > bottom]
            if len(body) < 6 or not body or body[-1][1] < end * 0.7:
                continue
            if any(
                not (values := _value_columns(_join_words(words, b - a)))
                or any(
                    min(abs(x - anchor) for anchor in anchors) > 3 for x, _ in values
                )
                for a, b, words in body
            ):
                continue
            bottom = body[-1][1] + 0.5

        # Reject any native line clipped across the side bounds or a paragraph
        # that spans the data area.
        if any(
            box[1] < bottom
            and box[3] > y0
            and box[0] < right
            and box[2] > left
            and (
                box[0] < left
                or box[2] > right
                or box[2] - box[0] > 0.75 * (right - left)
                and len(text.split()) > 12
            )
            for box, text in lines
        ):
            continue
        caption_boxes = [caption, *(box for box, _ in subtitle)]
        crop_left = min(left, *(box[0] - 0.5 for box in caption_boxes))
        crop_right = max(right, *(box[2] + 0.5 for box in caption_boxes))
        polygon = PolygonBox.from_bbox(
            [crop_left, cy0 - 0.5, crop_right, bottom + 0.25]
        ).fit_to_bounds(page.polygon.bbox)
        body_polygon = PolygonBox.from_bbox([left, y0, right, bottom])
        regions.append((polygon, body_polygon, title))
    return [
        region
        for region in regions
        if not any(
            region is not other and region[0].intersection_area(other[0]) > 0
            for other in regions
        )
    ]
