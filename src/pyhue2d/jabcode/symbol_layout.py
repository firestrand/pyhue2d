"""Shared standard module coordinates for master and slave symbols."""

from .image_processing.alignment_sampler import get_ap_positions


def metadata_coordinates(width: int, height: int, count: int) -> list[tuple[int, int]]:
    """Walk the four interleaved master metadata tracks."""
    coordinates: list[tuple[int, int]] = []
    x, y = 6, 1
    for index in range(1, count + 1):
        coordinates.append((x, y))
        if index % 2 == 0:
            y = height - 1 - y
        else:
            x = width - 1 - x
        if index % 4 == 0:
            if (
                index <= 20
                or 44 <= index <= 68
                or 96 <= index <= 124
                or 156 <= index <= 188
                or 224 <= index <= 260
                or 300 <= index <= 340
            ):
                y += 1
            elif (
                (20 < index < 44)
                or (68 < index < 96)
                or (124 < index < 156)
                or (188 < index < 224)
                or (260 < index < 300)
                or (340 < index < 384)
            ):
                x -= 1
            if index in (44, 96, 156, 224, 300, 384):
                x, y = y, x
    return coordinates


def pattern_modules(
    version_x: int, version_y: int, is_master: bool, color_count: int = 8
) -> dict[tuple[int, int], int]:
    """Return finder and alignment pattern module colors for the specified color count."""
    black = 0
    cyan = {4: 3, 8: 3, 16: 3, 32: 7, 64: 15}.get(color_count, 3)
    yellow = {4: 2, 8: 6, 16: 14, 32: 30, 64: 60}.get(color_count, 6)
    modules: dict[tuple[int, int], int] = {}
    xs, ys = get_ap_positions(version_x), get_ap_positions(version_y)
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            corner = row in (0, len(ys) - 1) and column in (0, len(xs) - 1)
            diagonal = 1 if (row + column) % 2 == 0 else -1
            if corner:
                diagonal = 1 if row == 0 else -1
            offsets = {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -diagonal), (1, diagonal)}
            color = (cyan if column == 0 else yellow) if corner else yellow
            for dx, dy in offsets:
                modules[x + dx, y + dy] = black if corner and row > 0 else color
            modules[x, y] = color if corner and row > 0 else black
            if not corner:
                for dx, dy in offsets:
                    modules[x + dx, y + dy] = cyan
                modules[x, y] = yellow
            elif not is_master:
                for dx, dy in offsets:
                    modules[x + dx, y + dy] = yellow
                modules[x, y] = cyan
            if corner and is_master:
                for sign in (-1, 1):
                    for offset in range(3):
                        modules[x + sign * 2, y + sign * diagonal * offset] = color if row > 0 else black
                        modules[x + sign * offset, y + sign * diagonal * 2] = color if row > 0 else black
    return modules


def slave_palette_coordinates(width: int, height: int, count: int) -> list[tuple[int, int]]:
    """Return palette modules grouped by palette entry and then side."""
    coordinates: list[tuple[int, int]] = []
    for index in range(count):
        x = 4 + index // 8
        y = 5 + (index % 8 if index // 8 % 2 == 0 else 7 - index % 8)
        coordinates.extend(((x, y), (width - 1 - y, x), (width - 1 - x, height - 1 - y), (y, height - 1 - x)))
    return coordinates


def reserved_coordinates(
    version_x: int, version_y: int, color_count: int, is_master: bool, default_mode: bool = False
) -> set[tuple[int, int]]:
    """Return finder, alignment, palette, and metadata coordinates."""
    reserved = set(pattern_modules(version_x, version_y, is_master, color_count))
    width, height = 17 + 4 * version_x, 17 + 4 * version_y
    palette_count = min(color_count - 2, 62)
    if is_master:
        bits_per_module = color_count.bit_length() - 1
        count = 4 * palette_count
        if not default_mode:
            count += 4 + (38 + bits_per_module - 1) // bits_per_module
        reserved.update(metadata_coordinates(width, height, count))
    else:
        reserved.update(slave_palette_coordinates(width, height, palette_count))
    return reserved
