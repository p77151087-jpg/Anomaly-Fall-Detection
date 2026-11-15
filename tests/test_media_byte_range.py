from email.message import Message
from io import BytesIO

import pytest

from fall_detect.visualization.media_server import RangeRequestHandler, byte_range


@pytest.mark.parametrize(
    "header,expected",
    [
        (None, None),
        ("bytes=0-3", (0, 3)),
        ("bytes=5-", (5, 9)),
        ("bytes=-3", (7, 9)),
        ("bytes=-20", (0, 9)),
        ("bytes=7-100", (7, 9)),
    ],
)
def test_byte_range_bounds(header, expected):
    assert byte_range(header, 10) == expected


@pytest.mark.parametrize(
    "header", ["bytes=10-", "bytes=8-3", "bytes=-0", "bytes=-", "bytes=0-3,7-9", "items=0-3"]
)
def test_invalid_ranges(header):
    with pytest.raises(ValueError):
        byte_range(header, 10)


class FakeHandler(RangeRequestHandler):
    def __init__(self, path, header=None):
        self.path = str(path)
        self.headers = Message()
        self.response_headers = {}
        if header:
            self.headers["Range"] = header

    def translate_path(self, path):
        return path

    def send_response(self, code, *args):
        self.status = code

    def send_header(self, key, value):
        self.response_headers[key] = value

    def end_headers(self):
        pass

    def send_error(self, code, *args):
        self.status = code


@pytest.mark.parametrize(
    "header,status,body",
    [
        (None, 200, b"0123456789"),
        ("bytes=2-5", 206, b"2345"),
        ("bytes=-3", 206, b"789"),
        ("bytes=20-", 416, b""),
    ],
)
def test_real_file_headers_and_content(tmp_path, header, status, body):
    p = tmp_path / "synthetic.mp4"
    p.write_bytes(b"0123456789")
    handler = FakeHandler(p, header)
    stream = handler.send_head()
    output = BytesIO()
    if stream:
        handler.copyfile(stream, output)
        stream.close()
    assert handler.status == status and output.getvalue() == body
    assert int(handler.response_headers["Content-Length"]) == len(body)
    if status == 206:
        assert handler.response_headers["Accept-Ranges"] == "bytes"
    assert p.read_bytes() == b"0123456789"
