"""Windows bridge for round-tripping equations through desktop MathType."""

from __future__ import annotations

import ctypes
import subprocess
import time
from ctypes import wintypes
from pathlib import Path


MATHTYPE_PATHS = (
    Path(r"C:\Program Files (x86)\MathType\MathType.exe"),
    Path(r"C:\Program Files\MathType\MathType.exe"),
)

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
SW_RESTORE = 9
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_A = 0x41
VK_C = 0x43
VK_V = 0x56
VK_N = 0x4E

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM), wintypes.LPARAM]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.OpenClipboard.argtypes = [wintypes.HWND]
user32.GetClipboardData.argtypes = [wintypes.UINT]
user32.GetClipboardData.restype = wintypes.HANDLE
kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalLock.restype = wintypes.LPVOID
kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalSize.restype = ctypes.c_size_t
kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = wintypes.HGLOBAL


class MathTypeBridgeError(RuntimeError):
    pass


def find_mathtype():
    for path in MATHTYPE_PATHS:
        if path.is_file():
            return path
    return None


def _window_title(hwnd):
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def find_mathtype_window():
    matches = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def callback(hwnd, _):
        title = _window_title(hwnd)
        if user32.IsWindowVisible(hwnd) and title.startswith("MathType"):
            matches.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return matches[0] if matches else None


def _open_clipboard(retries=12):
    for _ in range(retries):
        if user32.OpenClipboard(None):
            return
        time.sleep(0.05)
    raise MathTypeBridgeError("The Windows clipboard is busy; try again")


def set_clipboard_text(text):
    encoded = (str(text) + "\0").encode("utf-16-le")
    _open_clipboard()
    handle = None
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
        if not handle:
            raise MathTypeBridgeError("Could not allocate clipboard memory")
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            raise MathTypeBridgeError("Could not lock clipboard memory")
        ctypes.memmove(pointer, encoded, len(encoded))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            raise MathTypeBridgeError("Could not place the equation on the clipboard")
        handle = None  # clipboard owns it
    finally:
        user32.CloseClipboard()
        if handle:
            kernel32.GlobalFree(handle)


def _press_ctrl(key):
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    user32.keybd_event(key, 0, 0, 0)
    user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def activate_mathtype():
    hwnd = find_mathtype_window()
    if not hwnd:
        return None
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    return hwnd


def launch_mathtype():
    executable = find_mathtype()
    if executable is None:
        raise MathTypeBridgeError("Desktop MathType was not found")
    hwnd = activate_mathtype()
    if hwnd:
        _press_ctrl(VK_N)
        return hwnd
    subprocess.Popen([str(executable)])
    return None


def paste_current_formula():
    hwnd = activate_mathtype()
    if not hwnd:
        return False
    _press_ctrl(VK_V)
    return True


def copy_current_formula():
    hwnd = activate_mathtype()
    if not hwnd:
        raise MathTypeBridgeError("MathType is not open")
    _press_ctrl(VK_A)
    time.sleep(0.08)
    _press_ctrl(VK_C)


def _read_format(format_name):
    format_id = user32.RegisterClipboardFormatW(format_name)
    _open_clipboard()
    try:
        handle = user32.GetClipboardData(format_id)
        if not handle:
            return None
        size = kernel32.GlobalSize(handle)
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return None
        try:
            raw = ctypes.string_at(pointer, size)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()
    if format_name == "MathML Presentation":
        return raw.decode("utf-8-sig", errors="ignore").rstrip("\x00")
    return raw.decode("utf-16-le", errors="ignore").rstrip("\x00")


def read_mathml_clipboard():
    for name in ("MathML Presentation", "MathML", "application/mathml+xml"):
        value = _read_format(name)
        if value and "<math" in value:
            return value[value.find("<?xml") if "<?xml" in value else value.find("<math"):]
    raise MathTypeBridgeError(
        "No editable MathType equation was found on the clipboard. Select the equation in MathType and copy it."
    )

