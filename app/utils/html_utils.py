"""HTML utilities: template engine + markdown-to-HTML conversion."""
import html
import re
from pathlib import Path

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=str(Path(__file__).parents[1] / "templates"))


def analysis_to_html(analysis: str) -> str:
    result = html.escape(analysis)
    result = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", result)
    result = re.sub(
        r"(?m)^#{2,3}\s+(.+)$",
        r"<h3 style='font-size:15px;font-weight:700;color:#1e3a8a;margin:16px 0 6px'>\1</h3>",
        result,
    )
    result = re.sub(
        r"(?m)^(\d+\.\s+)([A-Z &]+(?:\s+&amp;\s+[A-Z]+)*)$",
        r"<h3 style='font-size:15px;font-weight:700;color:#1e3a8a;margin:18px 0 6px'>\1\2</h3>",
        result,
    )
    result = re.sub(
        r"(?m)^\s*[-•]\s+(.+)$",
        r"<li style='margin:4px 0 4px 18px'>\1</li>",
        result,
    )
    result = re.sub(
        r"(<li[^>]*>.*?</li>(?:\s*<li[^>]*>.*?</li>)*)",
        r"<ul style='list-style:disc;padding:0;margin:6px 0'>\1</ul>",
        result,
        flags=re.DOTALL,
    )
    result = result.replace("\n", "<br>")
    return result
