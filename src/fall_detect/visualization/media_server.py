"""Local static handler with single-byte-range support for seekable MP4 playback."""

import os
import re
from http.server import SimpleHTTPRequestHandler


def byte_range(header, size):
    if header is None:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if match is None or size <= 0 or not any(match.groups()):
        raise ValueError("Unsupported or unsatisfiable byte range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError("Empty suffix range")
        return max(0, size - length), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise ValueError("Unsatisfiable byte range")
    return start, end


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        self.selected_range = None
        if os.path.isdir(path):
            return super().send_head()
        try:
            # do_GET/do_HEAD own and close the returned file after sending the response.
            stream = open(path, "rb")  # noqa: SIM115
        except OSError:
            self.send_error(404, "File not found")
            return None
        info = os.fstat(stream.fileno())
        try:
            selected = byte_range(self.headers.get("Range"), info.st_size)
        except ValueError:
            stream.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{info.st_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        self.send_response(206 if selected else 200)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Last-Modified", self.date_time_string(info.st_mtime))
        self.send_header(
            "Content-Length", str(selected[1] - selected[0] + 1 if selected else info.st_size)
        )
        if selected:
            self.send_header("Content-Range", f"bytes {selected[0]}-{selected[1]}/{info.st_size}")
            stream.seek(selected[0])
            self.selected_range = selected
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        if self.selected_range is None:
            return super().copyfile(source, outputfile)
        remaining = self.selected_range[1] - self.selected_range[0] + 1
        while remaining:
            block = source.read(min(64 * 1024, remaining))
            if not block:
                break
            outputfile.write(block)
            remaining -= len(block)
