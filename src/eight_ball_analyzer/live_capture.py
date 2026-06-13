from __future__ import annotations

import ctypes
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import win32con
import win32gui
import win32ui


class ChromeWebCapture:
    """Visible Chrome/Edge window capture for the 8 Ball Pool web page.

    Chrome GPU/WebGL content is often blank through PrintWindow, so the primary
    path captures the visible desktop pixels under the browser client rect.
    """

    CHROME_EXES = ("chrome.exe", "msedge.exe", "brave.exe", "opera.exe")

    def __init__(self, window_name: str, debug_dir: str | Path = ".") -> None:
        self.window_name = window_name
        self.debug_dir = Path(debug_dir)
        self.hwnd: int | None = None
        self.latest_frame: np.ndarray | None = None
        self.frame_id = 0
        self.last_fetched_id = -1
        self.is_running = False
        self.thread = None
        self.startup_state = "Initialized"
        self.start_time = 0.0
        self.error: str | None = None
        self.table_bounds: tuple[int, int, int, int] | None = None
        self.table_corners: list[tuple[float, float]] | None = None
        self.table_status = "Table: not locked"
        self.prefer_print_window = True
        self._debug_written = False
        self._last_table_probe = 0.0
        self._set_dpi_awareness()

    @staticmethod
    def _set_dpi_awareness() -> None:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    @staticmethod
    def _process_path(hwnd: int) -> str:
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
            )
            if not handle:
                return ""
            buf = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_ulong(len(buf))
            ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
            ctypes.windll.kernel32.CloseHandle(handle)
            return buf.value.lower()
        except Exception:
            return ""

    def _find_best_hwnd(self) -> int:
        target = self.window_name.lower().strip()
        candidates: list[tuple[int, int, str, str, tuple[int, int, int, int]]] = []
        browser_fallbacks: list[tuple[int, int, str, str, tuple[int, int, int, int]]] = []

        def cb(hwnd: int, _extra) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True
            title_l = title.lower()
            exe_path = self._process_path(hwnd)
            exe_name = Path(exe_path).name
            is_browser = exe_name in self.CHROME_EXES
            title_match = target in title_l if target else False
            browser_target = target in {"chrome", "google chrome", "edge", "browser"}
            if "8 ball analyzer" in title_l:
                return True

            left, top, right, bottom = self._client_screen_rect(hwnd)
            width = right - left
            height = bottom - top
            if width < 300 or height < 200:
                return True

            score = width * height
            if is_browser:
                score += 2_000_000
            if "8 ball" in title_l or "8ballpool" in title_l:
                score += 4_000_000
            if title_match:
                score += 1_000_000
            if not title_match and not (browser_target and is_browser):
                if is_browser and ("8 ball" in target or "8ball" in target or "pool" in target):
                    browser_fallbacks.append((score, hwnd, title, exe_path, (left, top, right, bottom)))
                return True
            candidates.append((score, hwnd, title, exe_path, (left, top, right, bottom)))
            return True

        win32gui.EnumWindows(cb, None)
        if not candidates and browser_fallbacks:
            browser_fallbacks.sort(reverse=True, key=lambda item: item[0])
            candidates = browser_fallbacks
            print("[ChromeCapture] No exact 8 Ball Pool title found; using largest visible browser window.")

        if not candidates:
            return 0

        candidates.sort(reverse=True, key=lambda item: item[0])
        score, hwnd, title, exe_path, rect = candidates[0]
        print(f"[ChromeCapture] Selected HWND={hwnd} score={score} title={title!r} exe={exe_path} rect={rect}")
        return hwnd

    @staticmethod
    def _client_screen_rect(hwnd: int) -> tuple[int, int, int, int]:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        sx, sy = win32gui.ClientToScreen(hwnd, (left, top))
        ex, ey = win32gui.ClientToScreen(hwnd, (right, bottom))
        return sx, sy, ex, ey

    @staticmethod
    def _window_size(hwnd: int) -> tuple[int, int]:
        left, top, right, bottom = ChromeWebCapture._client_screen_rect(hwnd)
        return max(1, right - left), max(1, bottom - top)

    @staticmethod
    def _screen_grab_rect(rect: tuple[int, int, int, int]) -> np.ndarray | None:
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return None

        screen_dc = None
        src_dc = None
        mem_dc = None
        bmp = None
        try:
            screen_dc = win32gui.GetDC(0)
            src_dc = win32ui.CreateDCFromHandle(screen_dc)
            mem_dc = src_dc.CreateCompatibleDC()
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(src_dc, width, height)
            mem_dc.SelectObject(bmp)
            mem_dc.BitBlt((0, 0), (width, height), src_dc, (left, top), win32con.SRCCOPY)
            bmp_info = bmp.GetInfo()
            bmp_bits = bmp.GetBitmapBits(True)
            frame = np.frombuffer(bmp_bits, dtype=np.uint8).reshape(
                (bmp_info["bmHeight"], bmp_info["bmWidth"], 4)
            ).copy()
            return frame
        except Exception as exc:
            print(f"[ChromeCapture] Screen grab failed: {exc}")
            return None
        finally:
            try:
                if bmp:
                    win32gui.DeleteObject(bmp.GetHandle())
                if mem_dc:
                    mem_dc.DeleteDC()
                if src_dc:
                    src_dc.DeleteDC()
                if screen_dc:
                    win32gui.ReleaseDC(0, screen_dc)
            except Exception:
                pass

    @staticmethod
    def _print_window(hwnd: int, width: int, height: int) -> np.ndarray | None:
        bmp = None
        save_dc = None
        mfc_dc = None
        hwnd_dc = None
        try:
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bmp)
            result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)
            if result == 0:
                return None
            bmp_info = bmp.GetInfo()
            bmp_bits = bmp.GetBitmapBits(True)
            return np.frombuffer(bmp_bits, dtype=np.uint8).reshape(
                (bmp_info["bmHeight"], bmp_info["bmWidth"], 4)
            ).copy()
        except Exception as exc:
            print(f"[ChromeCapture] PrintWindow fallback failed: {exc}")
            return None
        finally:
            try:
                if bmp:
                    win32gui.DeleteObject(bmp.GetHandle())
                if save_dc:
                    save_dc.DeleteDC()
                if mfc_dc:
                    mfc_dc.DeleteDC()
                if hwnd_dc:
                    win32gui.ReleaseDC(hwnd, hwnd_dc)
            except Exception:
                pass

    @staticmethod
    def _is_blank(frame: np.ndarray | None) -> bool:
        if frame is None or frame.size == 0:
            return True
        bgr = frame[:, :, :3]
        return float(bgr.std()) < 2.0 or float(bgr.mean()) < 2.0

    @staticmethod
    def _felt_mask(frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        green_blue = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([118, 255, 245]))
        bright_blue = cv2.inRange(hsv, np.array([85, 45, 70]), np.array([130, 255, 255]))
        mask = cv2.bitwise_or(green_blue, bright_blue)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    @staticmethod
    def _order_points(points: np.ndarray) -> list[tuple[float, float]]:
        pts = np.array(points, dtype=float).reshape(4, 2)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]

    def _update_table_lock(self, frame_bgra: np.ndarray) -> None:
        now = time.time()
        if now - self._last_table_probe < 0.15:
            return
        self._last_table_probe = now

        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        mask = self._felt_mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        best = None
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < frame_area * 0.025:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < frame_bgr.shape[1] * 0.40 or h < frame_bgr.shape[0] * 0.40:
                continue
            aspect = w / max(h, 1)
            if not (1.35 <= aspect <= 3.8):
                continue
            fill = area / float(max(1, w * h))
            if fill < 0.30:
                continue
            score = area * (1.0 - min(abs(aspect - 2.0), 1.2) * 0.15) * fill
            if score > best_score:
                best_score = score
                best = contour

        if best is None:
            self.table_bounds = None
            self.table_corners = None
            self.table_status = "Table: searching"
            self._write_debug(frame_bgr, mask, None)
            return

        x, y, w, h = cv2.boundingRect(best)
        pad_x = max(2, int(w * 0.012))
        pad_y = max(2, int(h * 0.018))
        x = max(0, x + pad_x)
        y = max(0, y + pad_y)
        w = min(frame_bgr.shape[1] - x, max(1, w - 2 * pad_x))
        h = min(frame_bgr.shape[0] - y, max(1, h - 2 * pad_y))
        self.table_bounds = (x, y, w, h)
        self.table_corners = [
            (float(x), float(y)),
            (float(x + w), float(y)),
            (float(x + w), float(y + h)),
            (float(x), float(y + h)),
        ]
        self.table_status = f"Table: locked x={x} y={y} w={w} h={h}"
        self._write_debug(frame_bgr, mask, self.table_bounds)

    def _write_debug(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray,
        bounds: tuple[int, int, int, int] | None,
    ) -> None:
        if self._debug_written:
            return
        self._debug_written = True
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.debug_dir / "debug_chrome_capture.png"), frame_bgr)
            cv2.imwrite(str(self.debug_dir / "debug_table_mask.png"), mask)
            if bounds is not None:
                x, y, w, h = bounds
                cv2.imwrite(str(self.debug_dir / "debug_table_roi.png"), frame_bgr[y : y + h, x : x + w])
        except Exception as exc:
            print(f"[ChromeCapture] Debug write failed: {exc}")

    def start(self) -> None:
        import threading

        self.startup_state = "Thread starting"
        self.is_running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run_capture, daemon=True)
        self.thread.start()

    def _capture_once(self) -> np.ndarray | None:
        if not self.hwnd:
            return None
        width, height = self._window_size(self.hwnd)
        rect = self._client_screen_rect(self.hwnd)
        if self.prefer_print_window:
            frame = self._print_window(self.hwnd, width, height)
            if not self._is_blank(frame):
                return frame
            return self._screen_grab_rect(rect)

        frame = self._screen_grab_rect(rect)
        if not self._is_blank(frame):
            return frame
        return self._print_window(self.hwnd, width, height)

    def _run_capture(self) -> None:
        try:
            self.startup_state = "Finding Chrome window"
            self.hwnd = self._find_best_hwnd()
            if not self.hwnd:
                self.error = f"Chrome window not found: {self.window_name}"
                self.startup_state = "Error: Chrome window not found"
                print(self.error)
                return

            self.startup_state = "Capture session starting"
            frame = self._capture_once()
            if self._is_blank(frame):
                self.error = "First Chrome capture was empty. Keep the browser visible on screen."
                self.startup_state = "Error: first frame empty"
                print(self.error)
                return

            self.latest_frame = frame
            self.frame_id += 1
            self._update_table_lock(frame)
            self.startup_state = "First frame received"
            print(f"[ChromeCapture] First frame received shape={frame.shape}")

            while self.is_running:
                frame = self._capture_once()
                if not self._is_blank(frame):
                    self.latest_frame = frame
                    self.frame_id += 1
                    self._update_table_lock(frame)
                time.sleep(0.033)
        except Exception as exc:
            self.error = str(exc)
            self.startup_state = f"Error: {exc}"
            print("[ChromeCapture] Capture failed:", exc)
            traceback.print_exc()

    def get_frame(self) -> np.ndarray | None:
        if self.frame_id == self.last_fetched_id:
            return None
        self.last_fetched_id = self.frame_id
        return self.latest_frame

    def stop(self) -> None:
        self.is_running = False
