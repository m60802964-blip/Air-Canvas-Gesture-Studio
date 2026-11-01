"""Air Canvas Gesture Studio — single-file, webcam-controlled teaching canvas.

Dependencies: pip install opencv-python numpy mediapipe pymupdf
Run: python Air_Canvas_Gesture_Studio.py

Use two raised fingers to hover over buttons; use only your index finger to
draw. You can also click toolbar buttons and draw or erase with a mouse drag.
Closed fist pauses. Open-palm horizontal swipe changes PDF pages. Keyboard Q/Esc
is an emergency exit. Files can be selected using the gesture-based browser.
"""
from __future__ import annotations

import os
import time
import math
import urllib.request
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import fitz
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

WIDTH, HEIGHT = 1280, 720
HEADER = 114
FOOTER = 70
HOLD_SECONDS = 0.85
MAX_UNDO = 18
LANDMARK_SMOOTHING = 0.45
POSE_VOTE_FRAMES = 3
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
ROOT = Path(__file__).resolve().parent
DOCS = ROOT / 'air_canvas_documents'
EXPORTS = ROOT / 'air_canvas_exports'
MODELS = ROOT / 'models'
for folder in (DOCS, EXPORTS, MODELS):
    folder.mkdir(exist_ok=True)
MODEL = MODELS / 'hand_landmarker.task'

# BGR colours, as expected by OpenCV.
PALETTE = [
    ('BLUE', (255, 80, 30)), ('GREEN', (70, 185, 45)),
    ('RED', (45, 45, 240)), ('YELLOW', (0, 220, 255)),
    ('BLACK', (15, 15, 15)), ('PURPLE', (180, 65, 165)),
]
TOOLS = ['DRAW', 'LINE', 'RECT', 'CIRCLE', 'ERASE']
ACTIONS = ['UNDO', 'REDO', 'CLEAR', 'SAVE', 'PDF', 'PRESENT', 'IMAGE', 'EXPORT', 'TEXT', 'EXIT']
BG = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, txt, x, y, size=.48, color=(30, 30, 30), thick=1):
    cv2.putText(img, str(txt), (int(x), int(y)), FONT, size, color, thick, cv2.LINE_AA)


def in_box(pt, box):
    x, y = pt
    a, b, c, d = box
    return a <= x <= c and b <= y <= d


def landmark_xy(lm, index):
    point = lm[index]
    if isinstance(point, tuple):
        return point[0], point[1]
    return point.x, point.y


def fit_image(image, width, height):
    h, w = image.shape[:2]
    factor = min(width / w, height / h)
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)


def fit_page_to_canvas(page):
    """Render a PDF page as an editable full-size BGR canvas."""
    # Render at enough resolution for legible PDF text, while limiting memory.
    zoom = min(2.0, 1250 / max(page.rect.width, 1))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    channels = pix.n
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, channels)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR if channels == 4 else cv2.COLOR_RGB2BGR)
    return place_center(arr)


def place_center(image):
    result = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    resized = fit_image(image, WIDTH - 20, HEIGHT - HEADER - FOOTER - 12)
    h, w = resized.shape[:2]
    x = (WIDTH - w) // 2
    y = HEADER + (HEIGHT - HEADER - FOOTER - h) // 2
    result[y:y + h, x:x + w] = resized
    return result


def create_detector():
    if not MODEL.exists():
        print('Downloading official MediaPipe hand landmark model (one-time setup)...')
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL)
        except Exception as exc:
            raise RuntimeError(
                f'Cannot download the model automatically: {exc}\n'
                f'Download hand_landmarker.task from the official MediaPipe model page '
                f'and save it at: {MODEL}'
            ) from exc
    options = vision.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=.55,
        min_hand_presence_confidence=.55,
        min_tracking_confidence=.55,
    )
    return vision.HandLandmarker.create_from_options(options)


def finger_extended(lm, tip, pip, mcp):
    # Distances from knuckle work better than a vertical-Y-only test.
    def dist(i, j):
        xi, yi = landmark_xy(lm, i)
        xj, yj = landmark_xy(lm, j)
        return math.hypot(xi - xj, yi - yj)
    return dist(tip, mcp) > dist(pip, mcp) * 1.12


def finger_pose(lm):
    fingers = [finger_extended(lm, *ids) for ids in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17))]
    if all(fingers):
        return 'PALM'
    if not any(fingers):
        return 'FIST'
    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
        return 'SELECT'
    if fingers[0] and not any(fingers[1:]):
        return 'DRAW'
    return 'PAUSE'


class GestureStudio:
    def __init__(self):
        self.tool = 'DRAW'
        self.color = 0
        self.brush = 4
        self.mode = 'CANVAS'
        self.pages = [np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)]
        self.index = 0
        self.source_pdf = None
        self.undo = [[]]
        self.redo = [[]]
        self.last_point = None
        self.shape_start = None
        self.shape_preview = None
        self.stroke_active = False
        self.hover_target = None
        self.hover_since = 0.0
        self.hover_activated = False
        self.nav_feedback = ''
        self.palm_history = deque(maxlen=20)
        self.last_swipe = 0.0
        self.message = 'Ready: index finger DRAW | two fingers SELECT | fist PAUSE'
        self.message_until = time.monotonic() + 5
        self.file_items = []
        self.file_page = 0
        self.file_kind = 'PDF'
        self.ocr_reader = None
        self.text_overlay = ''
        self.last_ocr = None
        self.buttons = []
        self.last_timestamp = 0
        self.smoothed_landmarks = None
        self.pose_history = deque(maxlen=POSE_VOTE_FRAMES)
        self.stable_pose = None
        self.mouse_stroke_active = False

    @property
    def canvas(self):
        return self.pages[self.index]

    def smooth_landmarks(self, landmarks):
        current = [(float(point.x), float(point.y)) for point in landmarks]
        if self.smoothed_landmarks is None or len(self.smoothed_landmarks) != len(current):
            self.smoothed_landmarks = current
        else:
            alpha = LANDMARK_SMOOTHING
            self.smoothed_landmarks = [
                (old_x + alpha * (x - old_x), old_y + alpha * (y - old_y))
                for (old_x, old_y), (x, y) in zip(self.smoothed_landmarks, current)
            ]
        return self.smoothed_landmarks

    def stabilize_pose(self, pose):
        self.pose_history.append(pose)
        if self.stable_pose is None:
            self.stable_pose = pose
        candidate = max(
            self.pose_history,
            key=lambda item: (self.pose_history.count(item), item == self.stable_pose),
        )
        if self.pose_history.count(candidate) >= 2:
            self.stable_pose = candidate
        return self.stable_pose

    def note(self, msg, seconds=3):
        self.message, self.message_until = msg, time.monotonic() + seconds
        print(msg)

    def snapshot(self):
        stack = self.undo[self.index]
        stack.append(self.canvas.copy())
        if len(stack) > MAX_UNDO:
            del stack[0]
        self.redo[self.index].clear()

    def undo_action(self):
        if self.undo[self.index]:
            self.redo[self.index].append(self.canvas.copy())
            self.pages[self.index] = self.undo[self.index].pop()
            self.note('Undo completed')
        else:
            self.note('Nothing to undo')

    def redo_action(self):
        if self.redo[self.index]:
            self.undo[self.index].append(self.canvas.copy())
            self.pages[self.index] = self.redo[self.index].pop()
            self.note('Redo completed')
        else:
            self.note('Nothing to redo')

    def reset_pages(self, pages, pdf_path=None):
        self.pages = pages
        self.index = 0
        self.undo = [[] for _ in pages]
        self.redo = [[] for _ in pages]
        self.source_pdf = pdf_path
        self.reset_stroke()
        self.mode = 'CANVAS'

    def reset_stroke(self):
        self.last_point = None
        self.shape_start = None
        self.shape_preview = None
        self.stroke_active = False

    def change_page(self, delta):
        new_index = max(0, min(len(self.pages) - 1, self.index + delta))
        if new_index != self.index:
            self.index = new_index
            self.reset_stroke()
            self.note(f'Page {self.index + 1}/{len(self.pages)}')

    def open_browser(self, kind):
        self.file_kind = kind
        extensions = {'.pdf'} if kind in ('PDF', 'PRESENT') else {'.png', '.jpg', '.jpeg', '.bmp'}
        self.file_items = sorted(p for p in DOCS.iterdir() if p.is_file() and p.suffix.lower() in extensions)
        self.file_page = 0
        self.mode = 'BROWSER'
        self.reset_stroke()
        if not self.file_items:
            self.note(f'No {kind} files found. Put files in: {DOCS}', 7)

    def load_file(self, path):
        try:
            if path.suffix.lower() == '.pdf':
                with fitz.open(str(path)) as doc:
                    if not len(doc):
                        raise ValueError('PDF has no pages')
                    pages = [fit_page_to_canvas(page) for page in doc]
                self.reset_pages(pages, path)
            else:
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError('Cannot read image')
                self.reset_pages([place_center(img)])
            self.note(f'Opened {path.name} ({len(self.pages)} page(s))')
        except Exception as exc:
            self.note(f'Could not open file: {exc}', 7)

    def export(self, pdf=False):
        stamp = time.strftime('%Y%m%d_%H%M%S')
        try:
            if pdf:
                output = EXPORTS / f'annotated_{stamp}.pdf'
                with fitz.open() as doc:
                    for img in self.pages:
                        # Export visible content excluding the toolbar (no webcam overlay).
                        ok, encoded = cv2.imencode('.png', img[HEADER:HEIGHT - FOOTER])
                        if not ok:
                            raise RuntimeError('Image encoding failed')
                        page = doc.new_page(width=WIDTH, height=HEIGHT - HEADER - FOOTER)
                        page.insert_image(page.rect, stream=encoded.tobytes())
                    doc.save(str(output))
            else:
                output = EXPORTS / f'drawing_{stamp}.png'
                if not cv2.imwrite(str(output), self.canvas[HEADER:HEIGHT - FOOTER]):
                    raise RuntimeError('Could not write image')
            self.note(f'Saved: {output.name}', 5)
        except Exception as exc:
            self.note(f'Save failed: {exc}', 6)

    def ocr(self):
        # OCR is optional: the core project works without EasyOCR.
        try:
            import easyocr
            if self.ocr_reader is None:
                self.note('Loading handwriting OCR (may take time)...', 15)
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            crop = self.canvas[HEADER:HEIGHT - FOOTER]
            parts = self.ocr_reader.readtext(crop, detail=0, paragraph=True)
            self.text_overlay = ' | '.join(parts) if parts else 'No text recognized'
            self.last_ocr = self.text_overlay
            output = EXPORTS / f'handwriting_{time.strftime("%Y%m%d_%H%M%S")}.txt'
            output.write_text(self.text_overlay, encoding='utf-8')
            self.note('OCR text saved: ' + output.name, 6)
        except ImportError:
            self.note('TEXT requires: python -m pip install easyocr', 7)
        except Exception as exc:
            self.note(f'OCR unavailable: {exc}', 7)

    def activate(self, name):
        self.reset_stroke()
        if name.startswith('TOOL:'):
            self.tool = name[5:]
            self.note('Tool: ' + self.tool, 1.5)
        elif name.startswith('COLOR:'):
            self.color = int(name[6:])
            self.note('Colour: ' + PALETTE[self.color][0], 1.5)
        elif name.startswith('SIZE:'):
            self.brush = int(name[5:])
            self.note(f'Brush size: {self.brush}', 1.5)
        elif name == 'UNDO':
            self.undo_action()
        elif name == 'REDO':
            self.redo_action()
        elif name == 'CLEAR':
            self.snapshot()
            self.canvas[HEADER:HEIGHT - FOOTER] = 255
            self.note('Canvas cleared')
        elif name == 'SAVE':
            self.export(pdf=False)
        elif name == 'EXPORT':
            self.export(pdf=True)
        elif name in ('PDF', 'PRESENT', 'IMAGE'):
            self.open_browser(name)
        elif name == 'TEXT':
            self.ocr()
        elif name == 'EXIT':
            self.mode = 'EXIT'
        elif name == 'PREV':
            self.change_page(-1)
        elif name == 'NEXT':
            self.change_page(1)
        elif name == 'BROWSER:BACK':
            self.mode = 'CANVAS'
        elif name == 'BROWSER:PREV':
            self.file_page = max(0, self.file_page - 1)
        elif name == 'BROWSER:NEXT':
            self.file_page = min(max(0, (len(self.file_items) - 1) // 7), self.file_page + 1)
        elif name.startswith('BROWSER:FILE:'):
            idx = int(name.split(':')[-1])
            if 0 <= idx < len(self.file_items):
                self.load_file(self.file_items[idx])

    def add_button(self, image, name, caption, rect, active=False):
        x1, y1, x2, y2 = rect
        fill = (207, 234, 255) if active else (240, 243, 246)
        cv2.rectangle(image, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (110, 115, 125), 1)
        text_size = cv2.getTextSize(caption, FONT, .43, 1)[0]
        label(image, caption, x1 + (x2 - x1 - text_size[0]) // 2,
              y1 + (y2 - y1 + text_size[1]) // 2, .43)
        self.buttons.append((name, rect))

    def draw_toolbar(self, frame):
        cv2.rectangle(frame, (0, 0), (WIDTH, HEADER), (249, 249, 251), -1)
        cv2.rectangle(frame, (0, HEIGHT - FOOTER), (WIDTH, HEIGHT), (247, 248, 250), -1)
        self.buttons = []
        x, y, w, h, gap = 8, 8, 76, 38, 5
        for tool in TOOLS:
            self.add_button(frame, 'TOOL:' + tool, tool, (x, y, x + w, y + h), tool == self.tool)
            x += w + gap
        for action in ACTIONS:
            self.add_button(frame, action, action, (x, y, x + w, y + h))
            x += w + gap
            if x + w > WIDTH:
                break
        # Remaining actions appear on a second toolbar row.
        shown = (x - 8) // (w + gap)
        all_buttons = TOOLS + ACTIONS
        for i, name in enumerate(all_buttons[shown:], start=0):
            self.add_button(frame, name, name, (8 + i * (w + gap), 51, 8 + i * (w + gap) + w, 89))
        # Colors and brush sizes along bottom row.
        for i, (name, color) in enumerate(PALETTE):
            left = 10 + i * 55
            rect = (left, HEIGHT - 57, left + 46, HEIGHT - 17)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]),
                          (0, 180, 255) if i == self.color else (115, 115, 115), 3 if i == self.color else 1)
            self.buttons.append((f'COLOR:{i}', rect))
        for i, sz in enumerate((2, 4, 7, 11, 17, 25)):
            left = 355 + i * 62
            self.add_button(frame, f'SIZE:{sz}', str(sz),
                            (left, HEIGHT - 57, left + 54, HEIGHT - 17), self.brush == sz)
        self.add_button(frame, 'PREV', 'PREV PAGE', (730, HEIGHT - 64, 846, HEIGHT - 8))
        self.add_button(frame, 'NEXT', 'NEXT PAGE', (852, HEIGHT - 64, 968, HEIGHT - 8))
        label(frame, f'{self.index + 1}/{len(self.pages)}', 978, HEIGHT - 31, .57)
        label(frame, 'Mouse: click / drag', 1040, HEIGHT - 42, .40)
        label(frame, '2 fingers select | 1 draw', 1040, HEIGHT - 22, .40)
        label(frame, 'AIR CANVAS | GESTURE TEACHING STUDIO', 8, 106, .45, (90, 90, 90))

    def draw_browser(self, frame):
        frame[:] = (249, 249, 252)
        self.buttons = []
        label(frame, f'SELECT {self.file_kind} FILE (2 fingers: hover 0.85 seconds)', 35, 60, .83)
        label(frame, 'Place your PDFs/images in: air_canvas_documents (created beside this .py file)',
              35, 94, .55)
        start = self.file_page * 7
        for row, path in enumerate(self.file_items[start:start + 7]):
            y = 126 + row * 68
            self.add_button(frame, f'BROWSER:FILE:{start + row}',
                            path.name[:85], (40, y, 1240, y + 53))
        if not self.file_items:
            label(frame, 'No matching files. Add files to the documents folder, then reopen this menu.',
                  45, 180, .66, (30, 30, 200))
        self.add_button(frame, 'BROWSER:BACK', 'BACK', (40, 638, 220, 690))
        self.add_button(frame, 'BROWSER:PREV', 'PREVIOUS', (440, 638, 620, 690))
        self.add_button(frame, 'BROWSER:NEXT', 'NEXT', (650, 638, 830, 690))
        label(frame, f'List page {self.file_page + 1}', 920, 670, .6)

    def hit_button(self, point):
        for name, box in self.buttons:
            if in_box(point, box):
                return name
        return None

    def handle_mouse(self, event, x, y, flags, _parameter):
        point = (max(0, min(WIDTH - 1, x)), max(0, min(HEIGHT - 1, y)))
        if event == cv2.EVENT_LBUTTONDOWN:
            target = self.hit_button(point)
            if target:
                self.mouse_stroke_active = False
                self.activate(target)
            elif self.mode == 'CANVAS' and HEADER <= point[1] < HEIGHT - FOOTER:
                self.mouse_stroke_active = True
                self.draw_at(point, None)
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_stroke_active:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self.draw_at(point, None)
        elif event == cv2.EVENT_LBUTTONUP and self.mouse_stroke_active:
            self.end_stroke()
            self.mouse_stroke_active = False

    def hover(self, point, frame):
        target = self.hit_button(point)
        now = time.monotonic()
        if target != self.hover_target:
            self.hover_target = target
            self.hover_since = now
            self.hover_activated = False
        if target is None:
            return
        for name, box in self.buttons:
            if name == target:
                x1, y1, x2, y2 = box
                required_hold = 0.55 if name in ('PREV', 'NEXT') else HOLD_SECONDS
                progress = min(1, (now - self.hover_since) / required_hold)
                cv2.rectangle(frame, (x1, y2 - 5),
                              (x1 + int((x2 - x1) * progress), y2), (30, 165, 55), -1)
                if progress >= 1 and not self.hover_activated:
                    self.hover_activated = True
                    self.activate(target)
                break

    def draw_shape(self, frame, start, end):
        color = PALETTE[self.color][1]
        if self.tool == 'LINE':
            cv2.line(frame, start, end, color, self.brush, cv2.LINE_AA)
        elif self.tool == 'RECT':
            cv2.rectangle(frame, start, end, color, self.brush)
        elif self.tool == 'CIRCLE':
            radius = int(math.dist(start, end))
            cv2.circle(frame, start, radius, color, self.brush, cv2.LINE_AA)

    def draw_at(self, point, frame):
        x, y = point
        if y < HEADER or y >= HEIGHT - FOOTER:
            self.end_stroke()
            return
        if not self.stroke_active:
            self.snapshot()
            self.stroke_active = True
            self.last_point = point
            self.shape_start = point
        color = PALETTE[self.color][1]
        if self.tool == 'DRAW':
            cv2.line(self.canvas, self.last_point, point, color, self.brush, cv2.LINE_AA)
        elif self.tool == 'ERASE':
            diameter = max(24, self.brush * 4)
            cv2.line(self.canvas, self.last_point, point, BG, diameter, cv2.LINE_AA)
            cv2.circle(self.canvas, point, diameter // 2, BG, -1, cv2.LINE_AA)
        else:
            self.shape_preview = point
        self.last_point = point

    def end_stroke(self):
        if self.stroke_active and self.shape_start and self.shape_preview and self.tool in ('LINE', 'RECT', 'CIRCLE'):
            self.draw_shape(self.canvas, self.shape_start, self.shape_preview)
        self.reset_stroke()

    def track_palm_swipe(self, lm):
        now = time.monotonic()
        cx = sum(landmark_xy(lm, i)[0] for i in (0, 5, 9, 13, 17)) / 5
        self.palm_history.append((now, cx))
        while self.palm_history and now - self.palm_history[0][0] > .45:
            self.palm_history.popleft()
        if len(self.palm_history) >= 4 and now - self.last_swipe > 1.1:
            dx = cx - self.palm_history[0][1]
            if abs(dx) > .16:
                self.change_page(1 if dx < 0 else -1)
                self.last_swipe = now
                self.palm_history.clear()

    def render(self, lm):
        frame = self.canvas.copy() if self.mode == 'CANVAS' else np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
        if self.mode == 'CANVAS':
            self.draw_toolbar(frame)
        else:
            self.draw_browser(frame)
        if lm is not None:
            lm = self.smooth_landmarks(lm)
            x, y = landmark_xy(lm, 8)
            x = int(x * WIDTH)
            y = int(y * HEIGHT)
            point = (max(0, min(WIDTH - 1, x)), max(0, min(HEIGHT - 1, y)))
            pose = self.stabilize_pose(finger_pose(lm))
            # Navigation accepts a two-finger gesture even if another finger
            # is briefly misclassified by the camera. Other buttons remain SELECT-only.
            nav_target = self.hit_button(point) if self.mode == 'CANVAS' else None
            nav_two_fingers = (finger_extended(lm, 8, 6, 5) and
                               finger_extended(lm, 12, 10, 9))
            nav_select = nav_target in ('PREV', 'NEXT') and nav_two_fingers
            if self.mouse_stroke_active:
                pass
            elif pose == 'SELECT' or nav_select:
                self.end_stroke()
                self.palm_history.clear()
                self.hover(point, frame)
            else:
                self.hover_target = None
                self.hover_activated = False
                if pose == 'DRAW' and self.mode == 'CANVAS':
                    self.draw_at(point, frame)
                else:
                    self.end_stroke()
                if pose == 'PALM' and self.mode == 'CANVAS':
                    self.track_palm_swipe(lm)
                else:
                    self.palm_history.clear()
            cv2.circle(frame, point, 11, (240, 30, 240), 2)
            label(frame, pose + (' | NAV' if nav_select else ''),
                  max(0, point[0] - 20), max(120, point[1] - 16), .5, (130, 25, 130))
        else:
            if not self.mouse_stroke_active:
                self.end_stroke()
            self.hover_target = None
            self.palm_history.clear()
            self.smoothed_landmarks = None
            self.pose_history.clear()
            self.stable_pose = None
        if self.shape_start and self.shape_preview and self.mode == 'CANVAS':
            self.draw_shape(frame, self.shape_start, self.shape_preview)
        if time.monotonic() < self.message_until:
            cv2.rectangle(frame, (12, HEIGHT - 110), (min(WIDTH - 10, 1140), HEIGHT - 76), (235, 245, 255), -1)
            label(frame, self.message[:115], 20, HEIGHT - 88, .49)
        if self.text_overlay:
            label(frame, 'OCR: ' + self.text_overlay[:100], 12, HEIGHT - 120, .5, (100, 50, 130))
        return frame

    def run(self):
        detector = create_detector()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if not cap.isOpened():
            detector.close()
            raise RuntimeError('Webcam unavailable. Check Windows camera permissions and close other camera apps.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cv2.namedWindow('Air Canvas Gesture Studio', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Air Canvas Gesture Studio', WIDTH, HEIGHT)
        cv2.setMouseCallback('Air Canvas Gesture Studio', self.handle_mouse)
        try:
            while self.mode != 'EXIT':
                ok, camera = cap.read()
                if not ok:
                    raise RuntimeError('Webcam stopped returning frames')
                camera = cv2.flip(camera, 1)
                camera = cv2.resize(camera, (WIDTH, HEIGHT))
                rgb = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int(time.monotonic() * 1000)
                timestamp = max(timestamp, self.last_timestamp + 1)
                self.last_timestamp = timestamp
                detection = detector.detect_for_video(mp_image, timestamp)
                lm = detection.hand_landmarks[0] if detection.hand_landmarks else None
                view = self.render(lm)
                cv2.imshow('Air Canvas Gesture Studio', view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # Emergency exit only.
                    break
        finally:
            cap.release()
            detector.close()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        GestureStudio().run()
    except Exception as error:
        print('\nERROR:', error)
        print('Install dependencies: python -m pip install opencv-python numpy mediapipe pymupdf')
        raise
