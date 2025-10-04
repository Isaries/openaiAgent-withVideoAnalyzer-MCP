import os
import io
import subprocess
import base64
import tempfile
import json
import re
from typing import List, Tuple, Optional, Union
from shutil import which as _which
from contextvars import ContextVar
import base64 as _b64mod

# -------------------------------
# OpenAI（保險匯入）
# -------------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # 避免型別檢查報錯；實際執行前仍需存在


# -------------------------------
# 全域常數與設定
# -------------------------------
N_FRAMES_MIN = 1
N_FRAMES_MAX = 60  # 使用者可指定的最大影格數（可視需求調整）


# -------------------------------
# 日誌收集
# -------------------------------
_log_list_var: ContextVar[Optional[list]] = ContextVar("_log_list", default=None)

def start_log_capture():
    _log_list_var.set([])

def get_captured_logs() -> List[str]:
    logs = _log_list_var.get()
    return list(logs) if logs is not None else []

def end_log_capture():
    _log_list_var.set(None)

def print_info(msg: str):
    print(f"[INFO] {msg}")
    logs = _log_list_var.get()
    if logs is not None:
        logs.append(msg)


# -------------------------------
# 工具：環境/解碼
# -------------------------------
def has_ffmpeg() -> bool:
    return _which("ffmpeg") is not None

def has_ffprobe() -> bool:
    return _which("ffprobe") is not None

def _guess_image_mime(b: bytes) -> str:
    if not b or len(b) < 4:
        return "image/jpeg"
    if b[:2] == b"\xFF\xD8":
        return "image/jpeg"
    if b[:4] == b"\x89PNG":
        return "image/png"
    return "image/jpeg"

def encode_image_bytes_to_data_url(image_bytes: bytes, mime: str = None) -> str:
    mime = mime or _guess_image_mime(image_bytes)
    return f"data:{mime};base64," + _b64mod.b64encode(image_bytes).decode("utf-8")

def seconds_to_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


# -------------------------------
# 影格數決策與取樣點
# -------------------------------
def decide_frame_count_by_duration(duration_sec: float) -> int:
    if duration_sec < 60:
        return 4
    elif 60 <= duration_sec < 180:
        return 7
    elif 180 <= duration_sec < 300:
        return 11
    else:
        return 15

def linspace(start: float, end: float, num: int, include_end: bool = False):
    if num <= 1:
        return [start]
    span = end - start
    if span <= 0:
        return [start for _ in range(num)]
    if include_end:
        step = span / (num - 1)
        return [start + step * i for i in range(num)]
    else:
        step = span / num
        return [start + step * i for i in range(num)]

def _sanitize_desired_frames(val: Optional[Union[int, str]]) -> Optional[int]:
    """
    嘗試將輸入轉成合法整數影格數：
    - 接受 int，或可轉 float 的字串（'4'、'4.0'、' 7 '）
    - 拒絕布林
    - 範圍限制在 [N_FRAMES_MIN, N_FRAMES_MAX]
    - 非法回傳 None
    """
    if val is None:
        return None
    # 避免 True/False 被當成 1/0
    if isinstance(val, bool):
        return None
    try:
        s = str(val).strip()
        if s == "":
            return None
        # 先轉成 float 再轉 int（允許 '4.0' 類型字串）
        n = int(float(s))
        if n < N_FRAMES_MIN or n > N_FRAMES_MAX:
            return None
        return n
    except Exception:
        return None

def choose_frame_count(duration: float, desired_frames: Optional[Union[int, str]]) -> Tuple[int, str]:
    """
    回傳 (final_n_frames, reason_msg)
    - 若 desired_frames 合法（N_FRAMES_MIN~N_FRAMES_MAX）→ 採用之
    - 否則回退 decide_frame_count_by_duration
    """
    user_n = _sanitize_desired_frames(desired_frames)
    if user_n is not None:
        return user_n, f"使用者指定影格數：raw={desired_frames} → 採用 {user_n} 張"
    fallback = decide_frame_count_by_duration(duration)
    return fallback, (
        f"未提供或數值異常：raw={desired_frames}，允許範圍 {N_FRAMES_MIN}~{N_FRAMES_MAX} → "
        f"依時長規則取 {fallback} 張"
    )


# -------------------------------
# OpenAI 請求封裝
# -------------------------------
def chat_vision_analyze_image_from_bytes(client, model: str, prompt_text: str, image_bytes: bytes, temperature: float = 0.4) -> str:
    data_url = encode_image_bytes_to_data_url(image_bytes, mime=_guess_image_mime(image_bytes))
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                }
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI 影像分析失敗: {e}")

def chat_text_only(client, model: str, prompt: str, temperature: float = 0.4) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI 文字處理失敗: {e}")

def transcribe_audio_bytes(client, wav_bytes: bytes) -> str:
    buf = io.BytesIO(wav_bytes)
    buf.name = "audio.wav"
    buf.seek(0)
    try:
        tr = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=buf
        )
        return (tr.text or "").strip()
    except Exception:
        buf.seek(0)
        tr = client.audio.transcriptions.create(
            model="whisper-1",
            file=buf
        )
        return (tr.text or "").strip()

def audio_analysis_from_memory(wav_bytes: bytes, client) -> str:
    if not isinstance(wav_bytes, (bytes, bytearray)) or len(wav_bytes) == 0:
        raise RuntimeError("音訊資料無效，無法轉文字")
    return transcribe_audio_bytes(client, wav_bytes)

def combine_image_and_audio(image_desc: str, audio_text: str, client, model: str = "gpt-4o") -> str:
    prompt = (
        "我將提供兩段文字：一段是基於影格的影片畫面分析摘要（不含音訊），"
        "另一段是影片音訊的完整轉錄。請你融合兩者資訊，產生一段對整支影片的完整描述，"
        "要求包含：\n"
        "- 全片主題、結構與時間脈絡\n"
        "- 圖像與音訊共同指向的重點內容\n"
        "- 關鍵事件、角色/物件與意義\n"
        "- 可辨識文字（若音訊或畫面有）、地點、人名或特定名詞\n"
        "- 若畫面與音訊有差異或補充，請說明其對理解的影響\n"
        "請避免過度臆測，並以中立、清楚的語氣總結。\n\n"
        f"[圖像分析摘要]\n{image_desc}\n\n"
        f"[音訊轉錄全文]\n{audio_text}\n\n"
        "請輸出最終的完整描述。"
    )
    return chat_text_only(client, model, prompt, temperature=0.4)


# -------------------------------
# 子行程與暫存檔
# -------------------------------
def _run_cmd(cmd: list) -> Tuple[int, bytes, bytes]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return p.returncode, out or b"", err or b""

def _materialize_to_tempfile(video_bytes: bytes, suffix: str = ".mp4") -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        f.write(video_bytes)
        f.flush()
    finally:
        f.close()
    return f.name


# -------------------------------
# ffprobe/ffmpeg（走 path）
# -------------------------------
def detect_audio_with_ffprobe_path(video_path: str) -> bool:
    if not has_ffprobe():
        return False
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path
    ]
    code, out, _ = _run_cmd(cmd)
    if code == 0:
        return len(out.decode("utf-8", "ignore").strip()) > 0
    return False

def extract_audio_bytes_with_ffmpeg_path(video_path: str) -> Optional[bytes]:
    if not has_ffmpeg():
        return None
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v", "error",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "pipe:1"
    ]
    code, out, _ = _run_cmd(cmd)
    return out if code == 0 and out else None

def _get_duration_via_ffmpeg_decode_path(video_path: str) -> Optional[float]:
    if not has_ffmpeg():
        return None
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "info",
        "-i", video_path, "-f", "null", "-"
    ]
    code, _, err = _run_cmd(cmd)
    if err:
        txt = err.decode("utf-8", "ignore")
        m = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
        if m:
            h, m_, s = m[-1]
            return int(h) * 3600 + int(m_) * 60 + float(s)
    return None

def get_video_duration_seconds_from_path(video_path: str) -> Optional[float]:
    if not has_ffprobe():
        return None
    # 1) format.duration（加大探測量）
    cmd1 = [
        "ffprobe", "-v", "error", "-hide_banner",
        "-probesize", "100M", "-analyzeduration", "100M",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    code, out, _ = _run_cmd(cmd1)
    if code == 0 and out:
        try:
            dur = float(out.decode("utf-8", "ignore").strip())
            if dur > 0:
                return dur
        except Exception:
            pass
    # 2) streams.duration（加大探測量）
    cmd2 = [
        "ffprobe", "-v", "error", "-hide_banner",
        "-probesize", "100M", "-analyzeduration", "100M",
        "-select_streams", "v:0",
        "-show_entries", "stream=index,duration",
        "-of", "json",
        video_path
    ]
    code, out, _ = _run_cmd(cmd2)
    if code == 0 and out:
        try:
            obj = json.loads(out.decode("utf-8", "ignore"))
            for st in obj.get("streams") or []:
                d = st.get("duration")
                if d and d not in ("N/A", ""):
                    val = float(d)
                    if val > 0:
                        return val
        except Exception:
            pass
    # 3) 最後回退：解碼到尾
    return _get_duration_via_ffmpeg_decode_path(video_path)


# -------------------------------
# 影格擷取（走 path，含 mjpeg→png 後備、-ss 前/後）
# -------------------------------
def extract_frames_bytes_ffmpeg_pipe_from_path(
    video_path: str,
    timestamps: List[float],
    max_side: int = 1280,
    quality_param: int = 2
) -> List[Optional[bytes]]:
    results: List[Optional[bytes]] = [None] * len(timestamps)
    if not has_ffmpeg():
        return results

    scale_filter = f"scale=w={max_side}:h={max_side}:force_original_aspect_ratio=decrease"

    def try_one(codec: str, ss_before: bool) -> List[Optional[bytes]]:
        outs: List[Optional[bytes]] = [None] * len(timestamps)
        for idx, t in enumerate(timestamps):
            t = max(0.0, float(t))
            cmd = ["ffmpeg", "-hide_banner", "-v", "error"]
            if ss_before and t > 0:
                cmd += ["-ss", f"{t:.3f}"]
            cmd += ["-i", video_path, "-an"]
            if (not ss_before) and t > 0:
                cmd += ["-ss", f"{t:.3f}"]
            cmd += [
                "-frames:v", "1",
                "-vf", scale_filter,
                "-f", "image2pipe",
                "-vcodec", codec,
            ]
            if codec == "mjpeg":
                cmd += ["-q:v", str(quality_param), "-pix_fmt", "yuvj422p"]
            elif codec == "png":
                cmd += ["-pix_fmt", "rgb24"]
            cmd += ["pipe:1"]
            code, out, _ = _run_cmd(cmd)
            if code == 0 and out:
                outs[idx] = out
        return outs

    for codec, ss_before in [("mjpeg", False), ("mjpeg", True), ("png", False), ("png", True)]:
        tmp = try_one(codec, ss_before)
        if any(x is not None for x in tmp):
            return tmp

    return results


# -------------------------------
# 高階分析：圖像、音訊、融合
# -------------------------------
def gather_frames_in_memory_from_path(video_path: str, timestamps: List[float], expected_count: int) -> List[Tuple[int, float, bytes]]:
    print_info("影格將以記憶體方式處理（路徑讀取），不會在本地建立暫存影像檔")
    n = len(timestamps)
    frames: List[Optional[bytes]] = [None] * n
    ff = extract_frames_bytes_ffmpeg_pipe_from_path(video_path, timestamps, max_side=1280, quality_param=2)
    for i in range(n):
        if ff[i] is not None:
            frames[i] = ff[i]

    out: List[Tuple[int, float, bytes]] = []
    for i in range(n):
        if frames[i] is not None:
            out.append((i + 1, timestamps[i], frames[i]))

    if len(out) == 0:
        raise RuntimeError("無法擷取任何影格（路徑流程）")

    out.sort(key=lambda x: x[0])
    return out


def image_analysis_from_path(video_path: str, client: OpenAI, model: str = "gpt-4o", desired_frames: Optional[Union[int, str]] = None) -> str:
    print_info("開始圖像分析（路徑）")
    print_info(f"desired_frames(raw)={desired_frames} type={type(desired_frames).__name__}")
    duration = get_video_duration_seconds_from_path(video_path)
    if duration is None:
        raise RuntimeError("無法取得影片時長，圖像分析中止")

    n_frames, reason = choose_frame_count(duration, desired_frames)
    print_info(f"{reason}")
    print_info(f"影片時長約 {seconds_to_hms(duration)}，計畫擷取 {n_frames} 張影格（路徑處理）")

    timestamps = linspace(0.0, max(0.001, duration), n_frames, include_end=False)
    inmem_frames = gather_frames_in_memory_from_path(video_path, timestamps, expected_count=n_frames)

    per_frame_descriptions = []
    for idx, (frame_index, t, jpeg_or_png_bytes) in enumerate(inmem_frames, start=1):
        print_info(f"分析第 {idx}/{n_frames} 張影格")
        if idx == 1:
            prompt = (
                "你是一位嚴謹的影片畫面分析助手。請針對下方的單張影格，描述："
                "可見的場景/環境、人物或物件、動作與互動、可讀取的文字、情緒氛圍、"
                "可能的故事線索或情節脈絡。避免臆測與主觀判斷，請以可觀察到的細節為主。"
            )
        else:
            prev_summary = per_frame_descriptions[-1]["desc"]
            prompt = (
                "以下為第 n-1 張影格的簡述，請先理解上下文，再描述接下來的這張影格；"
                "如果與前一張有變化，請指出關鍵變化點：\n\n"
                f"[前一張影格摘要]\n{prev_summary}\n\n"
                "現在請描述本張影格，聚焦於畫面可見的元素、行動與脈絡。"
                "避免臆測與主觀判斷。"
            )

        desc = chat_vision_analyze_image_from_bytes(client, model, prompt, jpeg_or_png_bytes, temperature=0.4)
        per_frame_descriptions.append({"index": idx, "timestamp": t, "desc": desc})

    print_info("彙整所有影格的觀察，產生整體描述")
    bullets = []
    for item in per_frame_descriptions:
        ts = seconds_to_hms(item["timestamp"]) if item["timestamp"] is not None else "N/A"
        bullets.append(f"# 影格 {item['index']} @ {ts}\n{item['desc']}")
    joined = "\n\n".join(bullets)

    final_prompt = (
        "以下是針對多張等距取樣的影格所做的觀察，請你根據它們，"
        "產出對整部影片的詳細描述。請包含：\n"
        "- 影片主題與整體脈絡\n"
        "- 主要場景、人物/物件與其關係\n"
        "- 重要事件/轉折與合理的時間脈絡\n"
        "- 可辨識的文字資訊（若有）及其可能含義\n"
        "- 畫面風格/情緒與觀感\n"
        "請避免臆測無憑據的細節，僅以影格可支持的推論為主。以下是影格觀察：\n\n"
        f"{joined}\n\n"
        "請輸出一段條理清晰、可獨立閱讀的完整描述。"
    )
    final_description = chat_text_only(client, model, final_prompt, temperature=0.4)
    print_info("圖像分析完成")
    return final_description


# -------------------------------
# 主流程（路徑）
# -------------------------------
def _analyze_video_at_path(video_path: str, desired_frames: Optional[Union[int, str]] = None) -> str:
    print_info(f"ffprobe: {'found' if has_ffprobe() else 'missing'}, ffmpeg: {'found' if has_ffmpeg() else 'missing'}")
    try:
        size = os.path.getsize(video_path)
    except Exception:
        size = -1
    print_info(f"temp file: {video_path}, size={size} bytes")
    print_info("啟動主程式（路徑）")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("環境變數 OPENAI_API_KEY 未設置")

        client = OpenAI()

        audio_present = detect_audio_with_ffprobe_path(video_path)
        wav_bytes = None
        if audio_present:
            print_info("偵測到音訊，嘗試萃取（路徑）")
            wav_bytes = extract_audio_bytes_with_ffmpeg_path(video_path)
            if wav_bytes is None:
                print_info("音訊萃取失敗（路徑）")

        if not audio_present or wav_bytes is None:
            if not audio_present:
                print_info("流程：未偵測到音訊 → 僅執行圖像分析（路徑）")
            else:
                print_info("流程：偵測到音訊但萃取失敗 → 僅執行圖像分析（路徑）")
            image_summary = image_analysis_from_path(video_path, client, model="gpt-4o", desired_frames=desired_frames)
            print_info("主程式完成（僅圖像分析）")
            return image_summary

        print_info("流程：圖像分析 與 音訊分析（路徑）")
        image_summary = image_analysis_from_path(video_path, client, model="gpt-4o", desired_frames=desired_frames)
        audio_text = audio_analysis_from_memory(wav_bytes, client)
        final_text = combine_image_and_audio(image_summary, audio_text, client, model="gpt-4o")
        print_info("主程式完成（圖像 + 音訊已融合）")
        return final_text

    except Exception as e:
        print_info(f"主程式發生錯誤：{e}")
        return f"分析失敗：{e}"


# -------------------------------
# 對外 API：Binary/Base64 → 路徑分析（自動刪檔）
# -------------------------------
def analyze_video_bytes(video_bytes: Union[bytes, bytearray], filename: Optional[str] = None, desired_frames: Optional[Union[int, str]] = None) -> str:
    suffix = ".mp4"
    if filename and "." in filename:
        suffix = "." + filename.rsplit(".", 1)[1].lower()
    tmp_path = _materialize_to_tempfile(bytes(video_bytes), suffix=suffix)
    try:
        return _analyze_video_at_path(tmp_path, desired_frames=desired_frames)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def _b64_to_bytes(data: Union[str, bytes]) -> bytes:
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
    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        return base64.urlsafe_b64decode(s)

def analyze_video_base64(b64_data: Union[str, bytes], filename: Optional[str] = None, desired_frames: Optional[Union[int, str]] = None) -> str:
    video_bytes = _b64_to_bytes(b64_data)
    return analyze_video_bytes(video_bytes, filename=filename or "input.mp4", desired_frames=desired_frames)
