"""Capture-path facts, recorded in every run header.

Responsibility: describe the microphone as the experiment actually saw it.
The reSpeaker XVF3800 is a DSP, not a bare microphone: in USB-audio mode it
presents a processed 2-channel 16 kHz stream (beamformed + AEC + de-reverb on
chip), not the raw 4-mic array. Anything measured about STT accuracy or
endpoint timing is therefore a property of "this pipeline behind THIS
front-end firmware", and the paper's setup section has to say which.

Invariant: every field is read from the machine. A probe that fails records
its error rather than a plausible-looking default.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

XVF3800_USB_ID = "2886:001a"


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"error: {e}"
    return (out.stdout or out.stderr).strip()


def usb_descriptor(usb_id: str = XVF3800_USB_ID) -> dict[str, str]:
    """Firmware revision (bcdDevice), strings and audio-class channel counts."""
    text = _run(["lsusb", "-d", usb_id, "-v"], timeout=15.0)
    if text.startswith("error:") or not text:
        return {"present": "false", "probe": text or "no output"}
    info: dict[str, str] = {"present": "true", "usb_id": usb_id}
    for key, pattern in (
        ("firmware_bcd_device", r"bcdDevice\s+([0-9a-zA-Z.]+)"),
        ("product", r"iProduct\s+\d+\s+(.+)"),
        ("manufacturer", r"iManufacturer\s+\d+\s+(.+)"),
    ):
        m = re.search(pattern, text)
        if m:
            info[key] = m.group(1).strip()
    channels = sorted({m.group(1) for m in re.finditer(r"bNrChannels\s+(\d+)", text)})
    info["audio_class_channels"] = ",".join(channels) if channels else "unknown"
    return info


def alsa_capture_params(card: str = "Array") -> dict[str, str]:
    """Format/channels/rate the capture device actually offers."""
    text = _run(
        ["arecord", "-D", f"hw:CARD={card},DEV=0", "--dump-hw-params", "-d", "1", "/dev/null"],
        timeout=15.0,
    )
    params: dict[str, str] = {}
    for field in ("FORMAT", "CHANNELS", "RATE"):
        m = re.search(rf"^{field}:\s*(.+)$", text, re.MULTILINE)
        if m:
            params[field.lower()] = m.group(1).strip()
    return params or {"probe": "unavailable"}


def pipewire_node(substr: str = "reSpeaker") -> str:
    """The capture node's negotiated format, as PipeWire reports it."""
    text = _run(["pactl", "list", "short", "sources"])
    for line in text.splitlines():
        if substr.lower() in line.lower() and ".monitor" not in line:
            return line.strip()
    return "not found"


def capture_path_info(capture_channel: int | None = None) -> dict[str, str]:
    """Everything about the capture front-end, for the run header.

    ``mode`` is inferred, not assumed: 2 channels at 16 kHz from a 4-mic array
    means the on-chip DSP is delivering its processed (beamformed) output,
    whereas a raw array would expose 4+ channels.
    """
    usb = usb_descriptor()
    alsa = alsa_capture_params()
    info: dict[str, str] = {f"usb_{k}": v for k, v in usb.items()}
    info.update({f"alsa_{k}": v for k, v in alsa.items()})
    info["pipewire_node"] = pipewire_node()
    channels = alsa.get("channels", "")
    if channels.strip() == "2":
        info["mode"] = "usb-audio, DSP-processed 2ch (beamformed; not raw 4-mic)"
    elif channels:
        info["mode"] = f"usb-audio, {channels} channels"
    else:
        info["mode"] = "unknown"
    if capture_channel is not None:
        info["capture_channel_used"] = str(capture_channel)
    info["playback_default_sink"] = _run(["pactl", "get-default-sink"])
    cards = Path("/proc/asound/cards")
    if cards.exists():
        # Card entries are two lines each; the first carries index and id.
        names = [ln.strip() for ln in cards.read_text().splitlines() if "]:" in ln]
        info["asound_cards"] = " | ".join(names) if names else "none"
    else:
        info["asound_cards"] = "unavailable"
    return info
