# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import tempfile
from typing import Optional, Any, Union

from fastmcp import FastMCP
from openai import OpenAI
import video_analyzer_core as core

app = FastMCP("video-analyzer")


def _apply_api_key(explicit: Optional[str]) -> str:
    """
    core 透過環境變數讀取 API Key；若外部有傳 key，就寫入環境變數
    """
    key = explicit or os.getenv("OPENAI_API_KEY")
    if explicit:
        os.environ["OPENAI_API_KEY"] = explicit
    if not key:
        raise ValueError(
            "OpenAI API key not provided. Please pass `openai_api_key` or set env `OPENAI_API_KEY`."
        )
    return key


def _b64_to_bytes(data: Union[str, bytes]) -> bytes:
    """
    穩健的 base64 解碼：
    - 移除 data:*;base64, 前綴
    - 去除空白
    - 自動補齊 padding
    - 先用標準 base64，再回退 urlsafe base64
    """
    if isinstance(data, bytes):
        s = data.decode("ascii", errors="ignore")
    else:
        s = str(data)
    s = s.strip()
    if s.startswith("data:"):
        comma = s.find(",")
        if comma != -1:
            s = s[comma + 1:].strip()
    s = "".join(s.split())
    pad = (-len(s)) % 4
    if pad:
        s += "=" * pad
    import base64 as _b64
    try:
        return _b64.b64decode(s, validate=True)
    except Exception:
        return _b64.urlsafe_b64decode(s)


def _infer_suffix(filename: Optional[str]) -> str:
    if not filename or "." not in filename:
        return ".mp4"
    return "." + filename.rsplit(".", 1)[1].lower()


def _materialize_to_tempfile(data: bytes, suffix: str = ".mp4") -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        f.write(data)
        f.flush()
    finally:
        f.close()
    return f.name


@app.tool()
def analyze_video_b64(
    video_b64: str, 
    filename: Optional[str] = None, 
    desired_frames: Optional[Union[int, str]] = None,  
    openai_api_key: Optional[str] = None,
    include_logs: bool = True):
    
    _apply_api_key(openai_api_key)
    video_bytes = _b64_to_bytes(video_b64)
    core.start_log_capture()
    try:
        core.print_info(f"[MCP] desired_frames(param)={desired_frames} type={type(desired_frames).__name__}")
        result = core.analyze_video_bytes(
            video_bytes,
            filename=filename or "input.mp4",
            desired_frames=desired_frames
        )
        logs = core.get_captured_logs()
    finally:
        core.end_log_capture()
    return {"result": result, "logs": logs} if include_logs else result


@app.tool()
def image_analysis_b64(
    video_b64: str,
    filename: Optional[str] = None,
    desired_frames: Optional[Union[int, str]] = None,
    openai_api_key: Optional[str] = None,
    include_logs: bool = True,
    model: str = "gpt-4o"
) -> Any:
    """
    僅圖像分析（擷取影格→多輪視覺→彙整），輸入 base64/64Binary。
    先落地暫存檔，再呼叫 core.image_analysis_from_path，最後刪除暫存檔。
    可傳 desired_frames 指定要擷取的影格數。未傳或異常將由 core 回退到時長規則。
    """
    api_key = _apply_api_key(openai_api_key)
    client = OpenAI(api_key=api_key)

    video_bytes = _b64_to_bytes(video_b64)
    suffix = _infer_suffix(filename)
    tmp_path = _materialize_to_tempfile(video_bytes, suffix=suffix)

    core.start_log_capture()
    try:
        result = core.image_analysis_from_path(tmp_path, client, model=model, desired_frames=desired_frames)
        logs = core.get_captured_logs()
    finally:
        core.end_log_capture()
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return {"result": result, "logs": logs} if include_logs else result


@app.tool()
def audio_transcribe_b64(
    video_b64: str,
    filename: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    include_logs: bool = True
) -> Any:
    """
    僅音訊轉錄（從影片中萃取音訊→上傳轉錄），輸入 base64/64Binary。
    先落地暫存檔，走 core 的 path 版音訊偵測與萃取，最後刪除暫存檔。
    """
    api_key = _apply_api_key(openai_api_key)
    client = OpenAI(api_key=api_key)

    video_bytes = _b64_to_bytes(video_b64)
    suffix = _infer_suffix(filename)
    tmp_path = _materialize_to_tempfile(video_bytes, suffix=suffix)

    core.start_log_capture()
    try:
        audio_present = core.detect_audio_with_ffprobe_path(tmp_path)
        if not audio_present:
            result = "No audio detected or unable to verify audio"
        else:
            wav_bytes = core.extract_audio_bytes_with_ffmpeg_path(tmp_path)
            if wav_bytes is None:
                result = "Audio detected but extraction failed"
            else:
                result = core.audio_analysis_from_memory(wav_bytes, client)
        logs = core.get_captured_logs()
    finally:
        core.end_log_capture()
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return {"result": result, "logs": logs} if include_logs else result


@app.tool()
def test() -> Any:
    return "test"


if __name__ == "__main__":
    # 視需求調整 transport/host/port/path（埠保持你現在的 8000，不需更動）
    app.run(transport="http", host="0.0.0.0", port=8000, path="/mcp")
