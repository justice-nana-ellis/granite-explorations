from typing import Optional, Any
from typing_extensions import TypedDict
import pandas as pd


class SessionData(TypedDict, total=False):
    system: str
    messages: list[dict]
    display: list[dict]
    file: Optional[str]
    file_content_type: str
    file_raw: Optional[bytes]
    file_summary: Optional[str]
    cloudinary_id: Optional[str]
    cloudinary_url: Optional[str]
    state_cloudinary_id: Optional[str]
    df: Optional[Any]          # pd.DataFrame — typed as Any to avoid forward-ref issues
    df_summary: Optional[str]
    last_accessed: float
