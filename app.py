import os
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR / 'outputs'
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
MAX_IMAGES = 20
A4_SIZE = (1654, 2339)
TOP_SPACE = 180
MARGIN = 80

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET', 'dev-secret-change-me')
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def shape_arabic_text(text: str) -> str:
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def get_font(size=48):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def detect_and_align_document(pil_image: Image.Image) -> Image.Image:
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    original = image.copy()

    if image.shape[0] == 0:
        return pil_image

    ratio = image.shape[0] / 1000.0 if image.shape[0] > 1000 else 1.0
    new_height = 1000 if image.shape[0] > 1000 else image.shape[0]
    new_width = int(image.shape[1] / ratio)
    resized = cv2.resize(image, (new_width, new_height))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    screen_cnt = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is None:
        return pil_image

    pts = screen_cnt.reshape(4, 2) * ratio
    warped = four_point_transform(original, pts)
    result = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    return result


def enhance_image(pil_image: Image.Image) -> Image.Image:
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l_channel)
    lab = cv2.merge((l2, a_channel, b_channel))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = ImageOps.autocontrast(pil)
    pil = ImageEnhance.Sharpness(pil).enhance(1.3)
    pil = ImageEnhance.Contrast(pil).enhance(1.1)
    return pil


def prepare_image(path: Path) -> Image.Image:
    pil = Image.open(path)
    pil = ImageOps.exif_transpose(pil).convert('RGB')
    pil = detect_and_align_document(pil)
    pil = enhance_image(pil)
    return pil


def build_a4_page(image: Image.Image, name_text: str) -> Image.Image:
    page = Image.new('RGB', A4_SIZE, 'white')
    draw = ImageDraw.Draw(page)

    if name_text:
        font = get_font(50)
        display_text = shape_arabic_text(f'الاسم: {name_text}')
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (A4_SIZE[0] - text_w) // 2
        y = 45
        draw.rounded_rectangle(
            (x - 25, y - 15, x + text_w + 25, y + text_h + 15),
            radius=20,
            fill=(245, 245, 245),
            outline=(220, 220, 220),
            width=2,
        )
        draw.text((x, y), display_text, fill=(30, 30, 30), font=font)

    available_w = A4_SIZE[0] - (MARGIN * 2)
    available_h = A4_SIZE[1] - TOP_SPACE - MARGIN

    img_w, img_h = image.size
    ratio = min(available_w / img_w, available_h / img_h)
    new_size = (int(img_w * ratio), int(img_h * ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    x = (A4_SIZE[0] - new_size[0]) // 2
    y = TOP_SPACE + ((available_h - new_size[1]) // 2)
    page.paste(image, (x, y))
    return page


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/convert', methods=['POST'])
def convert():
    name = (request.form.get('name') or '').strip()
    files = request.files.getlist('images')

    valid_files = [f for f in files if f and f.filename and allowed_file(f.filename)]

    if not valid_files:
        flash('من فضلك ارفع صورة واحدة على الأقل بصيغة JPG أو PNG أو WEBP', 'error')
        return redirect(url_for('index'))

    if len(valid_files) > MAX_IMAGES:
        flash(f'الحد الأقصى {MAX_IMAGES} صورة في الملف الواحد', 'error')
        return redirect(url_for('index'))

    job_id = uuid.uuid4().hex
    temp_paths = []
    pages = []
    pdf_path = OUTPUT_DIR / f'{job_id}.pdf'

    try:
        for idx, file in enumerate(valid_files, start=1):
            ext = Path(file.filename).suffix.lower() or '.jpg'
            save_path = UPLOAD_DIR / f'{job_id}_{idx}{ext}'
            file.save(save_path)
            temp_paths.append(save_path)

        for path in temp_paths:
            prepared = prepare_image(path)
            page = build_a4_page(prepared, name)
            pages.append(page.convert('RGB'))

        first, *rest = pages
        first.save(pdf_path, 'PDF', resolution=100.0, save_all=True, append_images=rest)

        result_name = 'converted.pdf' if not name else f"{name.replace(' ', '_')}.pdf"
        return send_file(pdf_path, as_attachment=True, download_name=result_name)

    except Exception as exc:
        flash(f'حدث خطأ أثناء إنشاء الملف: {exc}', 'error')
        return redirect(url_for('index'))

    finally:
        for p in temp_paths:
            if p.exists():
                p.unlink(missing_ok=True)
        for page in pages:
            try:
                page.close()
            except Exception:
                pass


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
