import re
from math import ceil, floor, pi

import numpy as np
from PIL import Image

from marker.ocr.indra import _TABLE_CAPTION_RE


_LABEL_ONLY = re.compile(
    r"Table\s+(?:[A-Z]?[.-]?\d+(?:\.\d+)*|(?:[A-Z][.-])?[IVXLCDM]+)", re.IGNORECASE
)


_CONTINUED_LABEL = re.compile(
    rf"(?P<label>{_LABEL_ONLY.pattern})\s*\(\s*continued\s*\)", re.IGNORECASE
)


def _reading_box(box, turn):
    """Express a rectangle in the printed text direction. Translation cancels in comparisons."""
    x0, y0, x1, y1 = box
    if turn == 1:
        return [y0, -x1, y1, -x0]
    if turn == 3:
        return [-y1, x0, -y0, x1]
    return box


def _title_precedes_grid(page, table_box, title_box, lines, turn):
    """Check the printed title against the first full-width rule in reading order."""
    image = page.get_image(highres=False)
    if image is None:
        return False
    left, table_top, right, _ = table_box
    _, top, _, bottom = title_box
    height = bottom - top
    width = right - left
    start = min(table_top, top) - 2 * height
    detached = table_top - top > height
    if detached:
        # A detached heading must open the first ruled table in its column.
        start = _reading_box(page.polygon.bbox, turn)[1]
        if any(
            box[3] < top
            and left < (box[0] + box[2]) / 2 < right
            and re.match(r"^(?:table|tab\.)\s", text, re.I)
            for box, text in lines
        ):
            return False
    stop = bottom + 5 * height
    region = _reading_box(
        [left + 0.12 * width, start, right - 0.12 * width, stop], -turn % 4
    )
    sx, sy = (n / size for n, size in zip(image.size, page.polygon.size))
    x0, y0, x1, y1 = region
    pixels = image.crop((floor(x0 * sx), floor(y0 * sy), ceil(x1 * sx), ceil(y1 * sy)))
    if turn:
        pixels = pixels.transpose(
            Image.Transpose.ROTATE_90 if turn == 1 else Image.Transpose.ROTATE_270
        )
    ink = np.asarray(pixels.convert("L")) < 200
    rules = ink.all(axis=1)
    rows = np.flatnonzero(np.diff(rules, prepend=False, append=False))
    scale = sx if turn else sy
    if not rows.size or (rows[1] - rows[0]) / scale > 2:
        return False
    rule = start + rows[0] / scale
    if rule < bottom - 1 / scale:
        return False
    if not detached:
        return True

    # A grid can start at its first printed column heading, below a separate subtitle.
    if not 0 <= table_top - rule <= 1.5 * height:
        return False
    between = [
        (box, text)
        for box, text in lines
        if bottom <= box[1] < table_top - 0.5 and box[0] < right and box[2] > left
    ]
    return 1 <= len(between) <= 2 and all(
        box[3] <= rule
        and left <= box[0] < box[2] <= right
        and abs(box[0] + box[2] - left - right) <= 2 * height
        and not re.match(r"^(?:table|figure|fig\.)\s", text, re.I)
        for box, text in between
    )


def printed_table_labels(page, tables):
    """Associate an existing table with a standalone printed identifier at its upper edge."""
    native = page.pdftext_page
    if not tables or not native or native.get("rotation", 0):
        return []
    by_turn = {0: [], 1: [], 3: []}
    for block in native.get("blocks", ()):
        for line in block.get("lines", ()):
            spans = [s for s in line.get("spans", ()) if s.get("text", "").strip()]
            if not spans or not line.get("bbox"):
                continue
            angle = spans[0].get("rotation", 0)
            turn = round(angle / (pi / 2))
            if turn % 4 not in by_turn or any(
                abs(s.get("rotation", 0) - turn * pi / 2) > 0.001 for s in spans
            ):
                continue
            text = " ".join("".join(s.get("text", "") for s in line["spans"]).split())
            by_turn[turn % 4].append((_reading_box(line["bbox"], turn % 4), text))

    pairs = []
    for turn, lines in by_turn.items():
        for (left, top, right, bottom), text in lines:
            if not re.match(r"^table\b", text, re.IGNORECASE):
                continue
            height = bottom - top
            if height <= 0:
                continue
            # Sideways type often puts the word Table and its number in separate native lines.
            row = [
                (box, word)
                for box, word in lines
                if min(bottom, box[3]) - max(top, box[1]) >= height / 2
                and box[0] <= right + 1.5 * height
                and box[2] >= left - 1.5 * height
            ]
            row.sort(key=lambda item: item[0][0])
            title = " ".join(word for _, word in row)
            label = _LABEL_ONLY.fullmatch(title) or _CONTINUED_LABEL.fullmatch(title)
            if not label and not _TABLE_CAPTION_RE.match(text):
                continue
            x0, x1 = row[0][0][0], max(box[2] for box, _ in row)
            for table in tables:
                table_box = _reading_box(table.polygon.bbox, turn)
                xmin, ymin, xmax, ymax = table_box
                if xmin - 1 <= x0 < x1 <= xmax + 1 and ymin - 5 * height <= top < ymax:
                    # A separate printed title can sit above the crop.
                    name = None
                    if (
                        label
                        and top <= ymin + height
                        and abs(x0 + x1 - xmin - xmax) <= 4 * height
                        and _title_precedes_grid(
                            page, table_box, (x0, top, x1, bottom), lines, turn
                        )
                    ):
                        name = label.groupdict().get("label") or label.group()
                    pairs.append((table, (turn, x0, top), name))
    return [
        (table, name)
        for table, position, name in pairs
        if name and sum(t is table or p == position for t, p, _ in pairs) == 1
    ]
