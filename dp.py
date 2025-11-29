import sys, os, io, tempfile, traceback, re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import re
from collections import Counter
# …
EMP_BRACKET_RE = re.compile(
    r"Employer's name, address, and ZIP code.*?\[(.*?)\]",
    re.IGNORECASE | re.DOTALL
)

from PyPDF2 import PdfMerger, PdfReader, PdfWriter
import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.layout import LAParams
from PyPDF2 import PdfReader, PdfMerger

import platform
import pytesseract


# ✅ Auto-detect OS and set Tesseract path accordingly
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Vinod Kumar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


#rom pdf2image import convert_from_path
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import logging

# Add the helper at the [To get bookmark for]
PHRASE = "Employer's name, address, and ZIP code"
INT_PHRASE = "Interest income"

# Add the helper at the [To get bookmark for]
PHRASE = "Employer's name, address, and ZIP code"
INT_PHRASE = "Interest income"


def print_phrase_context(text: str, phrase: str = PHRASE, num_lines: int = 2):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if phrase.lower() in line.lower():
            for j in range(i, min(i + 1 + num_lines, len(lines))):
                print(lines[j], file=sys.stderr)
            break
       
# ── Unicode console on Windows
def configure_unicode():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
configure_unicode()

# ── Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration
#OPPLER_PATH = os.environ.get("POPPLER_PATH")  # e.g. "C:\\poppler\\Library\\bin"
OCR_MIN_CHARS = 50
PDFMINER_LA_PARAMS = LAParams(line_margin=0.2, char_margin=2.0)

# ── Priority tables
income_priorities = {
    'W-2': 1,
    'Consolidated-1099': 2,        # << add this line
    '1099-NEC': 3,
    '1099-PATR': 4,
    '1099-MISC': 5,
    '1099-OID': 6,
    '1099-G': 7,
    'W-2G': 8,
    '1065': 10,
    '1120-S': 9,
    '1041': 11,
    '1099-INT': 12,
    '1099-DIV': 13,
    '1099-R': 14,
    '1099-Q': 15,
    'K-1': 16,
    '1099-SA': 17
}
expense_priorities = {'1098-Mortgage':1,'1095-A':2,'1095-B':3,'1095-C':4,'5498-SA':5,'1098-T':6,'Property Tax':7,'Child Care Expenses':8,'1098-Other':9,'529-Plan':10}

def get_form_priority(ftype: str, category: str) -> int:
    table = income_priorities if category=='Income' else (expense_priorities if category=='Expenses' else {})
    return table.get(ftype, max(table.values())+1 if table else 9999)

# ── Logging helper
def log_extraction(src: str, method: str, text: str):
    snippet = text[:2000].replace('\n',' ') + ('...' if len(text)>2000 else '')
    logger.info(f"[{method}] {os.path.basename(src)} → '{snippet}'")

# to extract text from image

import io
import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
# ── Prevent PIL DecompressionBombError for large tax PDFs
Image.MAX_IMAGE_PIXELS = None  # Safe because inputs are trusted (W-2/1099 client docs)


def pdf_page_to_image(path: str, page_index: int, dpi: int = 300) -> Image.Image:
    """
    Convert a PDF page to a preprocessed PIL image optimized for OCR.
    Adds automatic rotation correction for 0°, 90°, 180°, 270° pages.
    Steps (no OpenCV):
      - Detect & fix PDF metadata rotation
      - OCR-based auto-rotation (Tesseract OSD)
      - High DPI render
      - Convert to grayscale
      - Auto-contrast & brightness boost
      - Sharpen twice
      - Adaptive dual-thresholding (light & dark)
      - Rescale small text images
    """
    doc = fitz.open(path)
    page = doc.load_page(page_index)

    # 🧭 Step 1: Correct rotation using PDF metadata
    rotation = int(page.rotation or 0)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom).prerotate(-rotation)

    pix = page.get_pixmap(matrix=mat, alpha=False)

    # Convert to RGB image
    try:
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    except Image.DecompressionBombError:
        logger.warning(f"⚠️ Skipping OCR: page too large in {path} p{page_index+1}")
        doc.close()
        return Image.new("L", (100, 100), color=255)

    # 🧠 Step 2: OCR-based auto-rotation (for scanned sideways pages)
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        if angle != 0:
            print(f"[Rotation Fix] Auto-rotating page {page_index+1} by {angle}°")
            img = img.rotate(-angle, expand=True)
    except Exception as e:
        print(f"[WARN] Tesseract OSD rotation failed on page {page_index+1}: {e}")

    doc.close()

    # 🖼 Step 3: Continue your original preprocessing
    img = img.convert("L")  # grayscale
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Brightness(img).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

    # Rescale if small
    w, h = img.size
    if w < 2000:
        scale = 2000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Dual thresholding
    def threshold(im, cutoff):
        return im.point(lambda x: 0 if x < cutoff else 255, "1")

    light = threshold(img, 160)
    dark = threshold(img, 200)
    black_ratio_light = sum(light.getdata()) / (255 * light.size[0] * light.size[1])
    black_ratio_dark = sum(dark.getdata()) / (255 * dark.size[0] * dark.size[1])
    img_final = light if black_ratio_light < black_ratio_dark else dark

    return img_final


def preprocess_old_safe(img: Image.Image) -> Image.Image:
    """
    Gentle, safe OCR preprocessing that improves clarity
    WITHOUT breaking W-2 text.
    """
    # 1. Convert to grayscale
    img = img.convert("L")

    # 2. Light auto-contrast (safe)
    img = ImageOps.autocontrast(img, cutoff=1)

    # 3. Slight sharpness boost
    img = ImageEnhance.Sharpness(img).enhance(1.2)

    # 4. Light contrast/brightness (safe)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Brightness(img).enhance(1.05)

    # 5. Light noise reduction
    img = img.filter(ImageFilter.MedianFilter(size=3))

    # 6. Upscale if image is small
    w, h = img.size
    if w < 1800:
        scale = 1800 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img
import concurrent.futures
import pytesseract
import sys
import traceback
from PyPDF2 import PdfReader

import threading
from pdfminer.layout import LAParams

# ---------------------------------------
# UPDATED PDFMiner LAParams (fixes broken words & broken lines)
# ---------------------------------------
PDFMINER_LA_PARAMS = LAParams(
    line_overlap=0.5,
    char_margin=2.5,     # Groups characters into words better
    line_margin=0.5,     # Groups words into lines better
    word_margin=1.0      # Prevents word-by-word line breaks
)


def extract_text(path: str, page_index: int) -> str:
    text = ""

    ocr_result = [""]     # list so it is mutable inside thread
    pdf_result = [""]

    # -----------------------
    # THREAD 1 → OCR
    # -----------------------
    def do_ocr():
        try:
            dpi = 300
            img = pdf_page_to_image(path, page_index, dpi=dpi)
            gray = img.convert("L")
            bw = gray.point(lambda x: 0 if x < 180 else 255, '1')

            t_ocr = pytesseract.image_to_string(
                bw,
                lang="eng",
                config="--oem 3 --psm 6"
            ) or ""

            print(f"[OCR dpi={dpi}]\n{t_ocr}", file=sys.stderr)
            ocr_result[0] = t_ocr
        except Exception:
            traceback.print_exc()
            ocr_result[0] = ""

    # -----------------------
    # THREAD 2 → PDFMiner
    # -----------------------
    def do_pdfminer():
        try:
            t1 = pdfminer_extract(
                path,
                page_numbers=[page_index],
                laparams=PDFMINER_LA_PARAMS   # <<< UPDATED HERE
            ) or ""

            print(f"[PDFMiner full]\n{t1}", file=sys.stderr)
            pdf_result[0] = t1
        except Exception:
            traceback.print_exc()
            pdf_result[0] = ""

    # -----------------------
    # RUN OCR & PDFMiner AT SAME TIME
    # -----------------------
    t1 = threading.Thread(target=do_ocr)
    t2 = threading.Thread(target=do_pdfminer)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # -----------------------
    # Now use EXACT original logic
    # -----------------------

    # OCR first
    if len(ocr_result[0].strip()) > len(text.strip()):
        text = ocr_result[0]

    # PDFMiner next
    if len(pdf_result[0].strip()) > len(text.strip()):
        text = pdf_result[0]

    # PyPDF2 fallback (unchanged)
    if len(text.strip()) < OCR_MIN_CHARS:
        try:
            reader = PdfReader(path)
            t2 = reader.pages[page_index].extract_text() or ""
            print(f"[PyPDF2 full]\n{t2}", file=sys.stderr)
            if len(t2.strip()) > len(text): 
                text = t2
        except Exception:
            traceback.print_exc()

    return text

# ── OCR for images
def extract_text_from_image(file_path: str) -> str:
    text = ""
    try:
        img = Image.open(file_path)
        if img.mode!='RGB': img = img.convert('RGB')
        et = pytesseract.image_to_string(img)
        if et.strip():
            print_phrase_context(et)
            text = f"\n--- OCR Image {os.path.basename(file_path)} ---\n" + et
        else: text = f"No text in image: {os.path.basename(file_path)}"
    except Exception as e:
        logger.error(f"Error OCR image {file_path}: {e}")
        text = f"Error OCR image: {e}"
    return text

#For rotating pages
import io
from PIL import Image
import pytesseract
import fitz

#For rotating pages
def is_unused_page(text: str) -> bool:
    """
    Detect pages that are just year-end messages, instructions,
    or generic investment details (not real 1099 forms).
    """
    import re
    lower = text.lower()
    # normalize multiple spaces to single
    norm = re.sub(r"\s+", " ", lower)

    investment_details = re.search(r"\b\d{4}\s+investment details", norm)

    return (
        "understanding your form 1099" in norm
        or "year-end messages" in norm
        or "important: if your etrade account transitioned" in norm
        or "please visit etrade.com/tax" in norm
        or "tax forms for robinhood markets" in norm
        or "robinhood retirements accounts" in norm
        or "new for 2023 tax year" in norm
        or "new for 2024 tax year" in norm
        or "new for 2025 tax year" in norm
        or "that are necessary for tax" in norm
        or "please note there may be a slight timing" in norm
        or "account statement will not have included" in norm
        #1099-SA
        or "fees and interest earnings are not considered" in norm
        or "an hsa distribution" in norm
        or "death is includible in the account" in norm
        or "the account as of the date of death" in norm
        or "amount on the account holder" in norm
        #1099-Mortgage
        or "for clients with paid mortgage insurance" in norm
        or "you can also contact the" in norm
        #or "" in norm
       
        or "may be requested by the mortgagor" in norm
       
        or "you should contact a competent" in norm
        or "tax lot closed on a first in" in norm
        or "your form 1099 composite may include the following internal revenue service " in norm
        or "schwab provides your form 1099 tax information as early" in norm
        or "if you have any questions or need additional information about your" in norm
        or "schwab is not providing cost basis" in norm
        or "the amount displayed in this column has been adjusted for option premiums" in norm
        or "you may select a different cost basis method for your brokerage" in norm
        or "to view and change your default cost basis" in norm
        or "this information is not intended to be a substitue for specific individualized" in norm
        or "shares will be gifted based on your default cost basis" in norm
        or "if you sell shares at a loss and buy additional shares" in norm
        or "we are required to send you a corrected from with the revisions clearly marked" in norm
        or "referenced to indicate individual items that make up the totals appearing" in norm
        or "issuers of the securities in your account reallocated certain income distribution" in norm
        or "the amount shown may be dividends a corporation paid directly" in norm
        or "if this form includes amounts belonging to another person" in norm
        or "spouse is not required to file a nominee return to show" in norm
        or "character when passed through or distributed to its direct or in" in norm
        or "brokers and barter exchanges must report proceeds from" in norm
        or "first in first out basis" in norm
        or "see the instructions for your schedule d" in norm
        or "other property received in a reportable change in control or capital" in norm
        or "enclosed is your" in norm and "consolidated tax statement" in norm
        or "filing your taxes" in norm and "turbotax" in norm
        or ("details of" in norm and "investment activity" in norm)
        or bool(investment_details)
    )


def extract_account_number(text: str) -> str:
    """
    Extract and normalize the account number or ORIGINAL number from page text.
    - Handles 'Account Number: ####' format
    - Handles 'ORIGINAL: ####' format
    Returns None if nothing found.
    """
    #Account Number
    #2360-9180
    match = re.search(r"Account\s*Number[:\s]*([\d\-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(" ", "").strip()
    # First try to capture "Account Number:"
    match = re.search(r"Account Number:\s*([\d\s]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(" ", "").strip()

    # If not found, try to capture "ORIGINAL:"
    match = re.search(r"ORIGINAL:\s*([\d\s]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(" ", "").strip()
   
    #Account 697296887
    match = re.search(r"Account\s+(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
   

    return None


# consolidated-1099 forms bookmark
def has_nonzero_misc(text: str) -> bool:
    patterns = [
        r"1\.RENTS\s*\$([0-9,]+\.\d{2})",
        r"2\.ROYALTIES\s*\$([0-9,]+\.\d{2})",
        r"3\.OTHER INCOME\s*\$([0-9,]+\.\d{2})",
        r"4\.FEDERAL INCOME TAX WITHHELD\s*\$([0-9,]+\.\d{2})",
        r"8\.SUBSTITUTE PAYMENTS.*\$\s*([0-9,]+\.\d{2})",
    ]
    return _check_nonzero(patterns, text)
def has_nonzero_oid(text: str) -> bool:
    patterns = [
        r"1\.ORIGINAL ISSUE DISCOUNT.*\$\s*([0-9,]+\.\d{2})",
        r"2\.OTHER PERIODIC INTEREST.*\$\s*([0-9,]+\.\d{2})",
        r"4\.FEDERAL INCOME TAX WITHHELD.*\$\s*([0-9,]+\.\d{2})",
        r"5\.MARKET DISCOUNT.*\$\s*([0-9,]+\.\d{2})",
        r"6\.ACQUISITION PREMIUM.*\$\s*([0-9,]+\.\d{2})",
        r"8\.OID ON.*\$\s*([0-9,]+\.\d{2})",
        r"9\.INVESTMENT EXPENSES.*\$\s*([0-9,]+\.\d{2})",
        r"10\.BOND PREMIUM.*\$\s*([0-9,]+\.\d{2})",
        r"11\.TAX-EXEMPT OID.*\$\s*([0-9,]+\.\d{2})",
    ]
    return _check_nonzero(patterns, text)
def has_nonzero_b(text: str) -> bool:
    """
    Detects if a 1099-B form is present.
    Returns True if there are nonzero dollar values OR if structural
    summary keywords (SHORT-TERM, LONG-TERM, UNKNOWN TERM with FORM 8949)
    are present.
    """
    # 1. Numeric value checks
    patterns = [
        r"1d\.PROCEEDS.*\$\s*([0-9,]+\.\d{2})",
        r"COVERED SECURITIES.*\$\s*([0-9,]+\.\d{2})",
        r"NONCOVERED SECURITIES.*\$\s*([0-9,]+\.\d{2})",
        r"1e\.COST OR OTHER BASIS.*\$\s*([0-9,]+\.\d{2})",
        r"1f\.ACCRUED MARKET DISCOUNT.*\$\s*([0-9,]+\.\d{2})",
        r"1g\.WASH SALE LOSS DISALLOWED.*\$\s*([0-9,]+\.\d{2})",
        r"4\.FEDERAL INCOME TAX WITHHELD.*\$\s*([0-9,]+\.\d{2})",
    ]
    if _check_nonzero(patterns, text):
        return True

    lower = text.lower()

    # 2. Structural fallback (existing)
    if (
        "short-term gains or (losses)" in lower
        or "long-term gains or (losses)" in lower
        or "unknown term" in lower
    ) and "form 8949" in lower:
        return True

    # 3. ✅ NEW: catch summary table headers even if all values are 0
    if any(kw in lower for kw in [
        "short a", "short b", "short c",
        "long d", "long e", "long f",
        "total short-term", "total long-term", "total undetermined"
    ]):
        return True

    return False


def has_nonzero_div(text: str) -> bool:
    """
    Detects if a 1099-DIV form has any nonzero amounts.
    Works for both UPPERCASE and lowercase extractions, with or without spaces.
    """
    patterns = [
        r"1a\s*\.?\s*.*ordinary dividends.*?\$\s*([0-9,]+\.\d{2})",
        r"1b\s*\.?\s*.*qualified dividends.*?\$\s*([0-9,]+\.\d{2})",
        r"2a\s*\.?\s*.*capital gain.*?\$\s*([0-9,]+\.\d{2})",
        r"2b\s*\.?\s*.*1250 gain.*?\$\s*([0-9,]+\.\d{2})",
        r"2c\s*\.?\s*.*1202 gain.*?\$\s*([0-9,]+\.\d{2})",
        r"2d\s*\.?\s*.*collectibles.*?\$\s*([0-9,]+\.\d{2})",
        r"2e\s*\.?\s*.*897 ordinary dividends.*?\$\s*([0-9,]+\.\d{2})",
        r"2f\s*\.?\s*.*897 capital.*?\$\s*([0-9,]+\.\d{2})",
        r"3\s*\.?\s*.*non[- ]?dividend.*?\$\s*([0-9,]+\.\d{2})",
        r"4\s*\.?\s*.*federal income tax withheld.*?\$\s*([0-9,]+\.\d{2})",
        r"5\s*\.?\s*.*199a dividends.*?\$\s*([0-9,]+\.\d{2})",
        r"6\s*\.?\s*.*investment expenses.*?\$\s*([0-9,]+\.\d{2})",
        r"7\s*\.?\s*.*foreign tax paid.*?\$\s*([0-9,]+\.\d{2})",
        r"9\s*\.?\s*.*cash liquidation.*?\$\s*([0-9,]+\.\d{2})",
        r"10\s*\.?\s*.*non[- ]?cash liquidation.*?\$\s*([0-9,]+\.\d{2})",
        r"12\s*\.?\s*.*exempt[- ]?interest dividends.*?\$\s*([0-9,]+\.\d{2})",
        r"13\s*\.?\s*.*specified private activity.*?\$\s*([0-9,]+\.\d{2})",
    ]

    return _check_nonzero(patterns, text)

def has_nonzero_int(text: str) -> bool:
    patterns = [
        r"1[\.\-,)]?\s*INTEREST\s+INCOME.*\$\s*([0-9,]+\.\d{2})",
        r"2[\.\-,)]?\s*EARLY\s+WITHDRAWAL\s+PENALTY.*\$\s*([0-9,]+\.\d{2})",
        r"3[\.\-,)]?\s*INTEREST\s+ON\s+U\.?S\.?\s+SAVINGS.*\$\s*([0-9,]+\.\d{2})",
        r"4[\.\-,)]?\s*FEDERAL\s+INCOME\s+TAX\s+WITHHELD.*\$\s*([0-9,]+\.\d{2})",
        r"5[\.\-,)]?\s*INVESTMENT\s+EXPENSES.*\$\s*([0-9,]+\.\d{2})",
        r"6[\.\-,)]?\s*FOREIGN\s+TAX\s+PAID.*\$\s*([0-9,]+\.\d{2})",
        r"8[\.\-,)]?\s*TAX[-\s]*EXEMPT\s+INTEREST.*\$\s*([0-9,]+\.\d{2})",
        r"9[\.\-,)]?\s*SPECIFIED\s+PRIVATE\s+ACTIVITY.*\$\s*([0-9,]+\.\d{2})",
        r"10[\.\-,)]?\s*MARKET\s+DISCOUNT.*\$\s*([0-9,]+\.\d{2})",
        r"(?:11|41|iS)[\.\-,)]?\s*BOND\s+PREMIUM.*\$\s*([0-9,]+\.\d{2})",   # OCR confusion: 11 ↔ 41 ↔ iS
        r"12[\.\-,)]?\s*BOND\s+PREMIUM\s+ON\s+TREASURY.*\$\s*([0-9,]+\.\d{2})",
        r"13[\.\-,)]?\s*BOND\s+PREMIUM\s+ON\s+TAX[-\s]*EXEMPT.*\$\s*([0-9,]+\.\d{2})",
    ]

    return _check_nonzero(patterns, text)
def _check_nonzero(patterns, text: str) -> bool:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            try:
                val = float(m.group(1).replace(",", "").replace("$", "").strip())
                if val != 0.0:
                    return True
            except:
                continue
    return False
# --- Post-processing cleanup for bookmarks ---
def filter_bookmarks(bookmarks: list[str]) -> list[str]:
    """
    If both '1099-B' and 'ST-A/B/C OR LT-D/E/F' appear,
    keep only 'ST-A/B/C OR LT-D/E/F'.
    """
    if "1099-B" in bookmarks and any(
        b for b in bookmarks if "ST-" in b or "LT-" in b
    ):
        return [b for b in bookmarks if b != "1099-B"]
    return bookmarks
def classify_text_multi(text: str) -> list[str]:
    """Return a list of form names detected in the page text."""
    lower = text.lower()
    matches = []

    has_int = ("1099-int" in lower or "form 1099-int" in lower) and has_nonzero_int(text)
    has_div = ("total ordinary dividends" in lower or "qualified dividends" in lower) and has_nonzero_div(text)

    # Other forms
    if "1099-b" in lower or "form 1099-b" in lower or has_nonzero_b(text):                                        
        if has_nonzero_b(text):
            matches.append("1099-B")

    if "1099-misc" in lower or "form 1099-misc" in lower:
        if has_nonzero_misc(text):
            matches.append("1099-MISC")

    if "1099-oid" in lower or "form 1099-oid" in lower:
        if has_nonzero_oid(text):
            matches.append("1099-OID")

    # ✅ NEW: Form 8949 Box conditions (ST/LT with A–F)
    box_map = {
        "box a checked": "ST-A",
        "box b checked": "ST-B",
        "box c checked": "ST-C",
        "box d checked": "LT-D",
        "box e checked": "LT-E",
        "box f checked": "LT-F",
    }
   
    for key, label in box_map.items():
        if key in lower:
            matches.append(label)

    # Combined condition for INT + DIV
    if has_int and has_div:
        cond1 = ("total federal income tax withheld" in lower
                 and "total interest income 1099-int box 1" in lower)
        cond2 = ("total qualified dividends" in lower
                 and "interest income" in lower)

        if cond1 or cond2:
            matches.append("1099-INT & DIV Description")
        elif cond1:
            matches.append("1099-DIV Description")
        elif cond2:
            matches.append("1099-INT Description")
        else:
            matches.append("1099-INT")
            matches.append("1099-DIV")
    else:
        if has_int:
            matches.append("1099-INT")
        if has_div:
            matches.append("1099-DIV")

    return matches

#k1helper
def extract_ein_number(text: str) -> str | None:
    """
    Extract EIN from text — tolerant to OCR dash variants and missing dashes.
    Example matches:
      12-3456789, 12–3456789, 12—3456789, 123456789
    """
    import re

    # Normalize all dash variants to a normal hyphen
    normalized = re.sub(r"[–—−‐‒﹘﹣]", "-", text)

    # Try standard EIN pattern (with dash)
    m = re.search(r"\b\d{2}-\d{7}\b", normalized)
    if m:
        return m.group(0)

    # Try continuous 9-digit pattern, and reformat
    m = re.search(r"\b(\d{9})\b", normalized)
    if m:
        raw = m.group(1)
        return f"{raw[:2]}-{raw[2:]}"  # insert dash

    return None




# --- Classification Helper



def classify_text(text: str) -> Tuple[str, str]:
    normalized = re.sub(r'\s+', '', text.lower())
    t = text.lower()
    lower = text.lower()
      # Detect W-2 pages by their header phrases
    t = re.sub(r"\s+", " ", text.lower()).strip()

    if (
        "fees and interest earnings are not considered contributions" in t
        or "contact a competent tax advisor or the irs" in t
        #or "instructions for recipient" in t
        #or "department of the treasury" in t
        #or "internal revenue service" in t
    ):
        return "Unused", "Others"

    if (
        "child care" in lower
        or "day care" in lower
        or "to the parents" in lower
       
        or "provider information" in lower
        or "total payments paid by" in lower
        #or "dates of service" in lower
        #or "total payments" in lower
        or "assistant business administrator" in lower
        or "preschool tuition payments" in lower
        or "the student named above has" in lower
    ):
        print(f"[DEBUG] CHILD CARE EXPENSE DETECTED in page: {text[:120]}...", file=sys.stderr)
        return "Expenses", "Child Care Expenses"
   
    unuseddiv = [
        "fundrise strives to provide your",
        "#although the fundrise team seeks to",
        "fundrise receives updated information for",
        #1099-SA
        "fees and interest earnings",
        "if you have questions regarding",
        "you should contact a competent tax advisor"
        "Fees and interest earnings are not considered contributions",
        "contact a competent tax advisor or the irs",
        "contributions or distributions and are not",
        "if you have questions regarding specific circumstances",
        "if you have questions regarding specific circumstances",
        "if you have questions regarding specific circumstances",
        #1098-T
        #"for the latest information about developments"
        "may result in an increase in tax",
        "reimbursements or refunds for the calendar",
       
       
    ]
    for pat in unuseddiv:
        if pat in lower:
            return "Others", "Unused"
    # --------------------------- 529 Plan / College Savings --------------------------- #
    # Detect 529 college savings plan statements or transaction notices
    clean_text = re.sub(r'[^A-Za-z0-9\s]', '', text.lower())  # normalize OCR artifacts
   
    if (
        "529" in clean_text
        and (
            #"indiana529" in clean_text
            "indiana 529" in clean_text
            or "529 direct savings plan" in clean_text
            or "education savings authority" in clean_text
            or "college savings" in clean_text
            or "qualified tuition program" in clean_text
            or "investment allocations" in clean_text
            or "investment portfolio" in clean_text
            or "funding information" in clean_text
            or "recurring contribution" in clean_text
            or "bank information" in clean_text
            or "electronic bank transfer" in clean_text
            #or "indiana529directcom" in clean_text
            #or "indiana 529 direct com" in clean_text
            or "indiana education savings" in clean_text
            or "contribution ebt" in clean_text
            or "please see below for details pertaining to" in clean_text
        )
    ):
        return "Expenses", "529-Plan"
    if "#bwnjgwm" in normalized:
        return "Others", "Unused"
   
    sa_front_patterns = [
        r"earnings\s+on\s+excess\s+cont",   # will also match 'cont.'
        #r"form\s+1099-?sa",                 # matches '1099-SA' or '1099SA'
        r"fmv\s+on\s+date\s+of\s+death",
    ]

    found_sa_front = any(re.search(pat, lower) for pat in sa_front_patterns)

    # 🔁 Priority: 1099-SA > Unused
    if found_sa_front:
        return "Income", "1099-SA"


   
    # 1) Detect W-2 pages by key header phrases
    if (
        "wages, tips, other compensation" in lower or
        ("employer's name" in lower and "address" in lower)
    ):
        return "Income", "W-2"

    #5498-SA
    # --- 5498-SA detection (more tolerant OCR patterns) ---
    sa5498_front_patterns = [
        r"form\s+[s§5]\s*498-?\s*sa",             # catches “5498-SA”, “S498-SA”, “§498-SA”
        r"form\s+5498sa",                         # no dash
        r"form\s+s498-sa",                        # OCR “5”→“S”
        r"form\s+§498-sa",                        # OCR “5”→“§”
        r"total\s+contributions\s+made\s+in\s+\d{4}",
        r"fair\s+market\s+value\s+of\s+(account|hsa)",
        r"\b2[\.\-)]?\s*rollover\s+contributions",
        r"\b5[\.\-)]?\s*fair\s+market\s+value\s+of\s+(account|hsa)",
        r"\b7[\.\-)]?\s*ira\s+type",
        r"\b11[\.\-)]?\s*required\s+minimum\s+distribution.*\d{4}"
    ]
    if any(re.search(pat, lower) for pat in sa5498_front_patterns):
        return "Expenses", "5498-SA"

        # --- Detect Schedule K-1 (Form 1065) ---
    # --- Detect Schedule K-1 (Form 1065 / 1120-S / 1041) and Statement A (QBI) pages ---
    if any(
        kw in lower
        for kw in [
            "schedule k-1",
            "form 1065",
            "form 1120-s",
            "form 1041",
            "statement a",
            "qualified business income",
            "section 199a",
            "qbi pass-through",
            "qbi pass through",
            "partnership",
            "accumulated differences may occur",
            "K-1 rental real estate activity",
            "for owners of pass-through entities",
            "tax paid on form or-oc filed on owner's behalf",
            "don’t submit with your individual tax return or the pte return",
        ]
    ):
        ein_match = re.search(r"\b\d{2}[-–]\d{7}\b", text)
        if ein_match:
            print(f"[DEBUG] classify_text: Detected K-1 Form 1065 EIN={ein_match.group(0)}", file=sys.stderr)
        return "Income", "K-1"

    if is_unused_page(text):
        return "Unknown", "Unused"
    if '1098-t' in t: return 'Expenses', '1098-T'
   
    # If page matches any instruction patterns, classify as Others → Unused
    instruction_patterns = [
    # full “Instructions for Employee…” block (continued from back of Copy C)
    # W-2 instructions
    "box 1. enter this amount on the wages line of your tax return",
    "box 2. enter this amount on the federal income tax withheld line",
    "box 5. you may be required to report this amount on form 8959",
    "box 6. this amount includes the 1.45% medicare tax withheld",
    "box 8. this amount is not included in box 1, 3, 5, or 7",
    "you must file form 4137",
    "box 10. this amount includes the total dependent care benefits",
    "instructions for form 8949",
    "regulations section 1.6045-1",
    "recipient's taxpayer identification number",
    "fata filing requirement",
    "payer’s routing transit number",
    "refer to the form 1040 instructions",
    "earned income credit",
    "if your name, SSN, or address is incorrect",
    "corrected wage and tax statement",
    "credit for excess taxes",
    "instructions for employee  (continued from back of copy c) "
    "box 12 (continued)",
    "f—elective deferrals under a section 408(k)(6) salary reduction sep",
    "g—elective deferrals and employer contributions (including  nonelective ",
    "deferrals) to a section 457(b) deferred compensation plan",
    "h—elective deferrals to a section 501(c)(18)(d) tax-exempt  organization ",
    "plan. see the form 1040 instructions for how to deduct.",
    "j—nontaxable sick pay (information only, not included in box 1, 3, or 5)",
    "k—20% excise tax on excess golden parachute payments. see the ",
    "form 1040 instructions.",
    "l—substantiated employee business expense reimbursements ",
    "(nontaxable)",
    "m—uncollected social security or rrta tax on taxable cost  of group-",
    "term life insurance over $50,000 (former employees only). see the form ",
    "1040 instructions.",
    "n—uncollected medicare tax on taxable cost of group-term  life ",
    "insurance over $50,000 (former employees only). see the form 1040 ",
    "instructions.",
    "p—excludable moving expense reimbursements paid directly to a ",
    "member of the u.s. armed forces (not included in box 1, 3, or 5)",
    "q—nontaxable combat pay. see the form 1040 instructions for details ",
    "on reporting this amount.",
    # 1099-INT instructions
    "box 1. shows taxable interest",
    "box 2. shows interest or principal forfeited",
    "box 3. shows interest on u.s. savings bonds",
    "box 4. shows backup withholding",
    "box 5. any amount shown is your share",
    "box 6. shows foreign tax paid",
    "box 7. shows the country or u.s. territory",
    "box 8. shows tax-exempt interest",
    "box 9. shows tax-exempt interest subject",
    "box 10. for a taxable or tax-exempt covered security",
    "box 11. for a taxable covered security",
    "box 12. for a u.s. treasury obligation",
    "box 13. for a tax-exempt covered security",
    "box 14. shows cusip number",
    "boxes 15-17. state tax withheld",
    # 1098-T instruction lines
    "you, or the person who can claim you as a dependent, may be able to claim an education credit",
    "student’s taxpayer identification number (tin)",
    "box 1. shows the total payments received by an eligible educational institution",
    "box 2. reserved for future use",
    "box 3. reserved for future use",
    "box 4. shows any adjustment made by an eligible educational institution",
    "box 5. shows the total of all scholarships or grants",
    "tip: you may be able to increase the combined value of an education credit",
    "box 6. shows adjustments to scholarships or grants for a prior year",
    "box 7. shows whether the amount in box 1 includes amounts",
    "box 8. shows whether you are considered to be carrying at least one-half",
    "box 9. shows whether you are considered to be enrolled in a program leading",
    "box 10. shows the total amount of reimbursements or refunds",
    "future developments. for the latest information about developments related to form 1098-t",
    # 1098-Mortgage
    ]
    for pat in instruction_patterns:
        if pat in lower:
            return "Others", "Unused"
       
       
        #---------------------------1099-DIV----------------------------------#
    #1099-INT for page 1
    div_front = [
        "form 1099-div",
        "dividends and distributions",
        "1a total ordinary dividends",
        "1b qualified dividends distributions",
        "2a Total capital gain distr",
        "specified private activity bond interest dividends",
        "qualified dividends",
        "total capital gain distr",
        "section 1202 gain",
        "section 1250 gain",
    ]

    div_unused = [
       
        "the information contained herein",
        "please note that we have changed",
        "your redeemed shares has not been",
        "we are requested by trh irs",
        ]
    lower = text.lower()
    found_div_front = any(pat.lower() in lower for pat in div_front)
    found_div_unused = any(pat.lower() in lower for pat in div_unused)

# 🔁 Priority: 1099-INT > Unused
    if found_div_front:
        return "Income", "1099-DIV"
    elif found_div_unused:
        return "Others", "Unused"
           
    # --- 1099-MISC ---
    misc_category = [
        "form 1099-misc",
        "miscellaneous information",
        "1.rents",
        "2.royalties",
        "3.other income",
        "8.substitute payments in lieu of dividends or interest"
    ]
    for pat in misc_category:
        if pat in lower:
            return "Income", "1099-MISC"

    # --- 1099-OID ---
    oid_category = [
        "form 1099-oid",
        "original issue discount",
        "1.original issue discount",
        "2.other periodic interest",
        "5.market discount",
        "6.acquisition premium",
        "8.oid on u.s. treasury obligations",
        "10.bond premium",
        "11.tax-exempt oid"
    ]
    for pat in oid_category:
        if pat in lower:
            return "Income", "1099-OID"

    # --- 1099-B ---
    b_category = [
        "form 1099-b",
        "proceeds from broker and barter exchange transactions",
        "1d.proceeds",
        "covered securities",
        "noncovered securities",
        "1e.cost or other basis of covered securities",
        "1f.accrued market discount",
        "1g.wash sale loss disallowed"
    ]
    for pat in b_category:
        if pat in lower:
            return "Income", "1099-B"

    #---------------------------Consolidated-1099----------------------------------#
   
     # E*TRADE text in parts
   
   

    con_unused = [
        "etrade from morgan stanley 1099 consolidated tax statement for 2023 provides your official tax information",
        "income information that was reported on your december account statement will not have included certain adjustments",
        "if your etrade account was transferred to morgan stanley smith barney llc in 2023 you may receive a separate 1099 consolidated tax statement",
        "consider and review both consolidated tax statements when preparing your 2023 income tax return",
        "for more information on what to expect, visit etrade.com/taxyear2023",
        "the following tax documents are not included in this statement and are sent individually",
        "forms 1099-q, 1042-s, 2439, 5498, 5498-esa, remic information statement, schedule k-1 and puerto rico forms 480.6a, 480.6b, 480.6c and 480.6d"
    ]
   
    for pat in con_unused:
        if pat in lower:
            return "Others", "Unused"  
    #---------------------------Consolidated-1099----------------------------------#

    #---------------------------1099-INT----------------------------------#
    #1099-INT for page 1
    int_front = [
        "3 Interest on U.S. Savings Bonds and Treasury obligations",
        #"Investment expenses",
        "Tax-exempt interest",
        "ond premium on Treasury obligations",
        "withdrawal penalty",
   
    ]

    int_unused = [
        "Box 1. Shows taxable interest paid to you ",
        "Box 2. Shows interest or principal forfeited",
        "Box 3. Shows interest on U.S. Savings Bonds",
        "Box 8. Shows tax-exempt interest paid to",
        "Box 10. For a taxable or tax-exempt covered security",
        "if you are registered in the account",
        "subject to reporting when paid regardless",
        "if we are required to withhold tax"
    ]
    lower = text.lower()
    found_int_front = any(pat.lower() in lower for pat in int_front)
    found_int_unused = any(pat.lower() in lower for pat in int_unused)

# 🔁 Priority: 1099-INT > Unused
    if found_int_front:
        return "Income", "1099-INT"
    elif found_int_unused:
        return "Others", "Unused"
    #---------------------------1099-SA----------------------------------#
    #1099-INT for page 1

   
    #---------------------------1098-Mortgage----------------------------------#    
    #1098-Mortgage form page 1
    mort_front = [
    "Mortgage insurance premiums",
    "Mortgage origination date",
    "Number of properties securing the morgage",  # typo here, maybe fix to "mortgage"
    "Address or description of property securing",
    "form 1098 mortgage",
    "limits based on the loan amount",
    "refund of overpaid",
    "Mortgage insurance important tax Information",
    #"Account number (see instructions)"
    ]
    mort_unused = [
        "instructions for payer/borrower",
        "payer’s/borrower’s taxpayer identification number",
        "box 1. shows the mortgage interest received",
        "Box 1. Shows the mortgage interest received by the recipient",
        "Box 3. Shows the date of the mortgage origination",
        "Box 5. If an amount is reported in this box",
        "Box 8. Shows the address or description",  # ← this line was missing a comma
        "This information is being provided to you as",
        "We’re providing the mortgage insurance",
        "If you received this statement as the payer of",
        "If your mortgage payments were subsidized"
       
    ]
    lower = text.lower()
    found_front = any(pat.lower() in lower for pat in mort_front)
    found_unused = any(pat.lower() in lower for pat in mort_unused)

# 🔁 Priority: 1098-Mortgage > Unused
    if found_front:
        return "Expenses", "1098-Mortgage"
    elif found_unused:
        return "Others", "Unused"

    #---------------------------1098-Mortgage----------------------------------#
#3) fallback form detectors
    if 'w-2' in t or 'w2' in t: return 'Income', 'W-2'
    if '1099-int' in t or 'interest income' in t: return 'Income', '1099-INT'
    #if '1099-div' in t: return 'Income', '1099-DIV'
    #if 'form 1099-div' in t: return 'Income', '1099-DIV'
   
    #if '1099' in t: return 'Income', '1099-Other'
    if 'donation' in t: return 'Expenses', 'Donation'
    return 'Unknown', 'Unused'

   
    # Detect W-2 pages by their header phrases
    if 'wage and tax statement' in t or ("employer's name" in t and 'address' in t):
        return 'Income', 'W-2'
   
# ── Parse W-2 fields bookmarks

import re
from typing import Dict, List
from difflib import SequenceMatcher
# ────────────────────────────────────────────────
# 🧹 Helper to clean duplicate employer names from OCR
# ────────────────────────────────────────────────
import re

def clean_employer_name(name: str) -> str:
    """
    Cleans repeated OCR fragments and stray symbols from employer names.
    Example:
        "UNITEO SERVICES AUTOMOBILE ASSN \\ UNITED SERVICES AUTOMOBILE ASSN j | UNITEO SERVICES AUTOMOBILE ASSN"
        -> "UNITED SERVICES AUTOMOBILE ASSN"
    """
    if not name:
        return ""

    # Normalize spacing and remove noisy characters
    name = re.sub(r"[\|\\/]+", " ", name)     # replace |, \, / with space
    name = re.sub(r"[^A-Za-z0-9&.,' ]+", " ", name)  # remove junk symbols
    name = re.sub(r"\s+", " ", name).strip()

    # Handle repeated full-name duplicates like "XYZ XYZ XYZ"
    parts = re.split(r"\s{2,}|\\|/", name)
    if parts:
        # pick the longest clean part (usually one full repetition)
        name = max(parts, key=len).strip()

    # Remove repeated uppercase sequences like "ABC CORP ABC CORP"
    name = re.sub(r"(\b[A-Z][A-Z\s&.,']{3,}\b)(?:\s+\1)+", r"\1", name)

    return name


def normalize_entity_name(raw: str) -> str:
    if not raw:
        return "N/A"
    raw = re.split(
        r"\b(employer|employee|ein|ssn|address|social security|withheld)\b",
        raw,
        flags=re.IGNORECASE
    )[0].strip()
    BAD_PREFIXES = (
        "employee", "wages", "social security", "medicare",
        "withheld", "tax", "omb", "form w-2", "department", "irs",
        "c employer", "© employer", "¢ employer", "= employer"
    )
    INLINE_JUNK = ["less:", "gross pay", "deductions", "earnings", "withheld", "retirement"]
    JUNK_SUFFIXES = ["TAX WITHHELD", "WITHHELD", "COPY", "VOID", "DUPLICATE"]

    stripped = raw.strip()

    # 🚫 skip if it's a header/junk line
    if any(stripped.lower().startswith(b) for b in BAD_PREFIXES):
        return "N/A"

    # Remove inline junk
    for jt in INLINE_JUNK:
        idx = stripped.lower().find(jt)
        if idx != -1:
            stripped = stripped[:idx].strip()
            break

    # Remove SSN/EIN patterns
    stripped = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '', stripped)  # SSN
    stripped = re.sub(r'\b\d{2}-\d{7}\b', '', stripped)        # EIN
    stripped = stripped.strip()

    # Collapse whole-line duplicates
    whole_dup = re.match(r'^(?P<seq>.+?)\s+(?P=seq)(?:\s+(?P=seq))*$', stripped, flags=re.IGNORECASE)
    if whole_dup:
        stripped = whole_dup.group('seq')

    # Collapse repeated adjacent words
    collapsed = re.sub(r'\b(.+?)\b(?:\s+\1\b)+', r'\1', stripped, flags=re.IGNORECASE)

    # Remove trailing numbers
    collapsed = re.sub(r'(?:\s+\d+(?:[\.,]\d+)?)+\s*$', '', collapsed)

    # Remove trailing junk suffixes
    words = collapsed.split()
    cleaned = True
    while cleaned and words:
        cleaned = False
        for junk in JUNK_SUFFIXES:
            parts = junk.split()
            if len(words) >= len(parts) and [w.upper() for w in words[-len(parts):]] == [p.upper() for p in parts]:
                words = words[:-len(parts)]
                cleaned = True
                break
    collapsed = " ".join(words)

    # Remove duplicated trailing employer names (fuzzy match)
    parts = collapsed.split()
    for cut in range(1, len(parts)):
        left = " ".join(parts[:cut])
        right = " ".join(parts[cut:])
        if right:
            ratio = SequenceMatcher(None, left.lower(), right.lower()).ratio()
            if ratio > 0.75:
                collapsed = left
                break

    # Drop stray numeric tokens at the end
    collapsed = re.sub(r'(\s+\d[\d\-\.,]*)+$', '', collapsed).strip()
    # --- 🔹 New: remove trailing junk tokens like TAX, A:, PAYROLL, etc. ---
    collapsed = re.sub(
        r'[\s,:;]+(?:tax|payroll|a\b|a:|dept|info|withheld|data|report|copy|stub)\b.*$',
        '',
        collapsed,
        flags=re.IGNORECASE,
    ).strip(" ,.-")


    return ' '.join(collapsed.split()).strip() or "N/A"


def next_valid_line(lines: List[str], start: int) -> str:
    """Return the next non-empty, non-header line after `start` index."""
    j = start
    while j < len(lines):
        cand = lines[j].strip()
        if cand and not ("employer" in cand.lower() and "address" in cand.lower() and "zip" in cand.lower()):
            return cand
        j += 1
    return ""
def is_name_like(s: str) -> bool:
    """Return True if s looks like a company name, not just numbers or junk."""
    if not s:
        return False
    # Must contain at least 2 alphabetic characters
    if sum(c.isalpha() for c in s) < 2:
        return False
    # Reject if it's only numbers, EIN, SSN, or amounts
    if re.fullmatch(r"[\d\-\.,]+", s):
        return False
    return True

def next_valid_line(lines, start_index, junk_phrases=None):
    if junk_phrases is None:
        junk_phrases = [
            "omb no",
            "control number",
            "payrol",
            "allocated tips",
            "social security wages",
            "social security tax withheld",
        ]

    j = start_index
    while j < len(lines):
        raw = lines[j].strip()
        if raw and not any(p in raw.lower() for p in junk_phrases):
            if is_name_like(raw):  # ⬅ ensure it looks like a name
                return raw
        j += 1
    return None

def parse_w2(text: str) -> Dict[str, str]:
    # SSN & EIN
    ssn_m = re.search(r"\b(\d{3}-\d{2}-\d{4})\b", text)
    ssn = ssn_m.group(1) if ssn_m else "N/A"
    ein_m = re.search(r"\b(\d{2}-\d{7})\b", text)
    ein = ein_m.group(1) if ein_m else "N/A"

    lines = text.splitlines()
    emp_name = emp_addr = "N/A"
    bookmark = None
    full_lower = text.lower()
   
    # 🚨 Hard-coded override for Salesforce
       # 🚨 Hard-coded employer overrides (handles OCR variants for known employers)
    # 🚨 Hard-coded employer overrides (handles OCR variants for known employers)
    EMP_OVERRIDES = {
        r"\bSALESFORCE[, ]+INC\.?\b": "SALESFORCE, INC",
    # ✅ TATA Consultancy Services robust OCR pattern
        r"tata\s*consult[a-z]{4,10}\s*serv[a-z]{4,10}.*limit[e3]d": "TATA CONSULTANCY SERVICES LIMITED",
    }

    for pattern, emp_name in EMP_OVERRIDES.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return {
                'ssn': ssn,
                'ein': ein,
                'employer_name': emp_name,
                'employer_address': emp_addr,
                'employee_name': 'N/A',
                'employee_address': 'N/A',
                'bookmark': emp_name
            }
        # 🔹 Detect NC HEALTH AFFILIATES LLC
    if re.search(r"health\s+affiliates\s+llc", text, flags=re.IGNORECASE):
        emp_name = "NC HEALTH AFFILIATES LLC"
        return {
            'ssn': ssn,
            'ein': ein,
            'employer_name': emp_name,
            'employer_address': "DURHAM, NC 27702-2291",
            'employee_name': 'N/A',
            'employee_address': 'N/A',
            'bookmark': emp_name
        }
        # 🧠 Try to detect employer name from OCR text near 'LLC' or 'INC'
    match = re.search(r"([A-Z][A-Z\s,&]+(?:LLC|INC|L\.L\.C\.|CORP|COMPANY))", text, flags=re.IGNORECASE)
    if match:
        emp_name = match.group(1).strip()
        emp_name = clean_employer_name(emp_name)
        return {
            'ssn': ssn,
            'ein': ein,
            'employer_name': emp_name.upper(),
            'employer_address': emp_addr,
            'employee_name': 'N/A',
            'employee_address': 'N/A',
            'bookmark': emp_name.upper()
        }


    # 🔹 1) FCA US LLC override
    if any(v in full_lower for v in ("fca us llc", "fca us, llc", "fcaus llc")):
        emp_name = "FCA US LLC"
        return {
            'ssn': ssn, 'ein': ein,
            'employer_name': emp_name,
            'employer_address': emp_addr,
            'employee_name': 'N/A',
            'employee_address': 'N/A',
            'bookmark': emp_name
        }
    # 🔹 3) Standard W-2 parsing
    for i, line in enumerate(lines):
        if "allocated tips" in line.lower() and "social security" in line.lower():
            raw = next_valid_line(lines, i + 1)
            if raw:
                emp_name = normalize_entity_name(raw)
                bookmark = emp_name
            emp_addr = next_valid_line(lines, i + 2)
            break
    #DOTCOM TEAM LLC B Employer Verification number …
    for i, line in enumerate(lines):
    # Match anything ending with "- PAYROL"
        if re.search(r".+\s*-\s*PAYROL", line, re.IGNORECASE):
            raw = next_valid_line(lines, i + 1)   # ⬅ skip "PAYROL" line and junk
            if raw:
            # If line has "b Employer..." trailing text, strip it out
                raw = re.sub(r"\bb\s*employer.*", "", raw, flags=re.IGNORECASE).strip()

            # Normalize (remove trailing numbers, extra spaces, etc.)
                emp_name = normalize_entity_name(raw)
                bookmark = emp_name
            break

           
    # 🔹 2) Marker block
    marker = (
        "c employer's name, address, and zip code 3 social security wages"
        "c Employer's name, address, and ZIP code "
        "8 Allocated tips 3 Social security wages 4 Social security tax withheld"
       
    ).lower()
    for i, L in enumerate(lines):
        if marker in L.lower():
            raw = next_valid_line(lines, i + 1)
            if raw:
                emp_name = normalize_entity_name(raw)
                bookmark = emp_name
                return {
                    'ssn': ssn, 'ein': ein,
                    'employer_name': emp_name,
                    'employer_address': emp_addr,
                    'employee_name': 'N/A',
                    'employee_address': 'N/A',
                    'bookmark': bookmark
                }

    # 🔹 3) Standard W-2 parsing
    for i, line in enumerate(lines):
        if "employer" in line.lower() and "name" in line.lower():
            raw = next_valid_line(lines, i + 1)
            if raw:
                emp_name = normalize_entity_name(raw)
                bookmark = emp_name
            emp_addr = next_valid_line(lines, i + 2)
            break

    # 🔹 4) PAYROL fallback
    if emp_name == "N/A":
        for i, line in enumerate(lines):
            if re.search(r".+\s*-\s*PAYROL", line, re.IGNORECASE):
                raw = next_valid_line(lines, i + 1)
                if raw:
                    emp_name = normalize_entity_name(raw)
                    bookmark = emp_name
                break

    # 🔹 5) © / triple marker fallbacks
    triple_markers = [
        "© Employer's name, address, and ZIP code",
        "c Employer's name, address, and ZIP code",
        "¢ Employer's name, address and ZIP code",
        "= EMPLOYER'S name, address, and ZIP code",
        "c Employer's name, address and ZIP code t c Employer's name, address and ZIP code"
    ]
    if emp_name == "N/A":
        for marker in triple_markers:
            if marker.lower() in full_lower:
                for i, line in enumerate(lines):
                    if marker.lower() in line.lower():
                        raw = next_valid_line(lines, i + 1)
                        if raw:
                            emp_name = normalize_entity_name(raw)
                            bookmark = emp_name
                        break

    # 🔹 Final cleanup
    if emp_name != "N/A":
        emp_name = normalize_entity_name(emp_name)
        bookmark = emp_name

    return {
        'ssn': ssn,
        'ein': ein,
        'employer_name': clean_employer_name(emp_name),
        'employer_address': emp_addr,
        'employee_name': 'N/A',
        'employee_address': 'N/A',
        'bookmark': clean_employer_name(emp_name)
    }


   
def print_w2_summary(info: Dict[str, str]):
    print("\n=== W-2 Summary ===\n")
    print(f"Employer: {info['employer_name']}, Address: {info['employer_address']}, EIN: {info['ein']}")
    print("===================\n")

#---------------------------W2----------------------------------#
#---------------------------1099-INT----------------------------------#
import re
from typing import List

def extract_1099int_bookmark(text: str) -> str:
    """
    Extract a clean payer/institution name for Form 1099-INT.
   
    Priority:
    1. Known overrides (US Bank, Capital One, Bank of America, etc.)
    2. First ALL-CAPS / title-cased line after 'foreign postal code, and telephone no.'
    3. Fallback: first line that looks like a bank/credit union name
    4. Default: '1099-INT'
    """
   

    import re
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lower_lines = [l.lower() for l in lines]

    # --- Step 1: Overrides for common institutions ---
    overrides = {
        "us bank na": "US Bank NA",
        "u.s. bank na": "US Bank NA",
        "capital one": "Capital One NA",
        "bank of america": "Bank of America",
        "digital federal credit union": "Digital Federal Credit Union",
        "fifth third bank": "FIFTH THIRD BANK, N.A.",   # ✅ new override
        "discover bank": "Discover Bank",
        "goldman sachs bank usa": "Goldman Sachs Bank USA",  # ✅ new override
    }
    for key, val in overrides.items():
        if key in text.lower():
            return val

    # --- Step 2: Top-down scan for bank-like names ---
    for cand in lines:
        cand_lower = cand.lower()
        if any(word in cand_lower for word in ["bank", "credit union", "mortgage", "trust", "financial"]):
            # strip trailing garbage like punctuation
            return re.sub(r"[^\w\s.&,'-]+$", "", cand).strip()
       
    # --- Step 3: Look after payer header (if available) ---
    for i, l in enumerate(lower_lines):
        if ("payer" in l and "information" in l) or ("foreign postal code" in l and "telephone" in l):
            for offset in range(1, 4):
                if i + offset >= len(lines):
                    break
                cand = lines[i + offset].strip()
                cand_lower = cand.lower()

                # skip junk
                bad_tokens = ["payer", "recipient", "federal id", "tin",
                              "street", "road", "apt", "zip"]
                if any(bad in cand_lower for bad in bad_tokens):
                    continue
                if re.match(r"^\d+[\s.]", cand):  # skip box lines
                    continue

                if (re.match(r"^[A-Z][A-Z\s&.,'-]{5,}$", cand) and not re.search(r"\d", cand)) \
                   or any(word in cand_lower for word in ["bank", "credit union", "mortgage", "trust", "financial"]):
                    return re.sub(r"[^\w\s.&'-]+$", "", cand).strip()
    # --- Step 4: Global scan again as a last resort ---
    for cand in lines:
        cand_lower = cand.lower()
        bad_tokens = ["payer", "recipient", "federal id", "tin",
                      "street", "road", "apt", "zip"]
        if any(bad in cand_lower for bad in bad_tokens):
            continue
        if re.match(r"^\d+[\s.]", cand):
            continue

        if any(word in cand_lower for word in ["bank", "credit union", "mortgage", "trust", "financial"]):
            return re.sub(r"[^\w\s.&'-]+$", "", cand).strip()


    # --- Step 4: Fallback ---
    return "1099-INT"


#---------------------------1099-INT----------------------------------#
# --- Issuer display aliases ---
ISSUER_ALIASES = {
    "morgan stanley capital management, llc": "E*TRADE",
    # add more mappings here if needed
}

def alias_issuer(name: str) -> str:
    return ISSUER_ALIASES.get(name.lower().strip(), name)

# --------------------------- Consolidated-1099 issuer name --------------------------- #
def extract_consolidated_issuer(text: str) -> str | None:
    """
    Returns a friendly issuer name for consolidated 1099 pages when detectable.
    Currently supports an explicit match for 'Morgan Stanley Capital Management, LLC'
    and a light heuristic fallback near 'Consolidated 1099'/'Composite 1099'.
    """
    lower = text.lower()

    # Explicit ask: Morgan Stanley Capital Management, LLC
    if re.search(r"morgan\s+stanley\s+capital\s+management,\s*llc", lower):
        return "Morgan Stanley Capital Management, LLC"
    # Explicit: Robinhood Markets Inc
    if re.search(r"robinhood\s+markets?\s+inc", lower):
        return "Robinhood Markets Inc"
    if re.search(r"robinhood\s+markets?\s+inc", lower):
        return "Charles Schwab"
    # Heuristic fallback: if the page looks like a consolidated/composite cover,
    # grab the first plausible line that looks like an issuer/legal name.
    if "consolidated 1099" in lower or "composite 1099" in lower:
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            # skip headings / noisy bits
            if re.search(r"(form|1099|copy|page|\baccount\b)", s, re.IGNORECASE):
                continue
            # something that looks like a firm name
            if re.search(r"(LLC|Bank|Securities|Wealth|Brokerage|Advisors?)", s):
                return re.sub(r"[^\w\s,&.\-]+$", "", s)

    return None
# --------------------------- Consolidated-1099 issuer name --------------------------- #
#---------------------------1099-DIV----------------------------------#
def extract_1099div_bookmark(text: str) -> str:
    """
    Extract the payer name for Form 1099-DIV.
    Handles OCR noise, skips junk lines, and applies direct overrides
    for known payers like Fundrise, Bank of America, etc.
    """
    import re

    # --- Step 1: normalize text for pattern matching ---
    normalized_text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    # --- Step 2: hardcoded overrides (fast exact detection) ---
    OVERRIDES = {
        "fundrise income real estate fund": "Fundrise Income Real Estate Fund, LLC",
        "fundrise income fund": "Fundrise Income Fund, LLC",
        # 🔹 Morgan Stanley (new)
        "morgan stanley domestic holdings": "Morgan Stanley Domestic Holdings, Inc",
        "morgan stanley domestic holding": "Morgan Stanley Domestic Holdings, Inc",
        "morgan stanley holdings inc": "Morgan Stanley Domestic Holdings, Inc",
        "morgan stanley holdings": "Morgan Stanley Domestic Holdings, Inc",
   
    }

    for key, val in OVERRIDES.items():
        if key in normalized_text:
            return val  # ✅ immediate return on match

    # --- Step 3: fallback pattern-based extraction ---
    lines = text.splitlines()
    lower_lines = [L.lower() for L in lines]
        # Normalize apostrophes to avoid OCR mismatch between ’ and '
    def normalize_apostrophes(s: str) -> str:
        return s.replace("’", "'").replace("`", "'")

    lower_lines = [normalize_apostrophes(L) for L in lower_lines]

    # Header detection pattern
    header_keywords = [
        "payer's name",
        "street address",
        "city or town",
        "state or province",
        "country",
        "zip",
        "telephone",
    ]

    # 1️⃣ Find header line that matches all key parts
    for i, L in enumerate(lower_lines):
        if all(k in L for k in header_keywords):
            # 2️⃣ Get the next non-empty line as bookmark
            for j in range(i + 1, len(lines)):
                candidate = lines[j].strip()
                if not candidate:
                    continue

                # Clean unwanted right-hand text
                candidate = re.sub(r"\s*\|.*$", "", candidate)   # remove trailing table/columns
                candidate = re.sub(r"\s*\$.*$", "", candidate)   # remove dollar values
                candidate = re.sub(r"[^\w\s,&.\-]+$", "", candidate).strip()

                if candidate:
                    return candidate

    def find_after(header_pred):
        for i, L in enumerate(lower_lines):
            if header_pred(L):
                for j in range(i + 1, len(lines)):
                    cand = lines[j].strip()
                    if not cand:
                        continue
                    cand_lower = cand.lower()

                    # Skip junk and header lines
                    if any(x in cand_lower for x in [
                        "foreign postal code", "telephone", "omb", "dividends", "distributions",
                        "copy b", "for recipient", "calendar year", "recipient’s tin",
                        "payer’s tin", "section", "gain", "tax withheld", "account number",
                    ]):
                        continue

                    if re.search(r"\bform\b", cand_lower):
                        continue
                    if len(cand) < 5 or not re.search(r"[A-Za-z]", cand):
                        continue

                    # If looks like an organization name
                    if re.search(r"\b(LLC|Inc|Fund|Trust|Bank|Corp|Company|Services|Advisors)\b", cand, re.IGNORECASE):
                        cand = re.sub(r"\s*\$.*$", "", cand)
                        return cand.strip(" ,.-")

                    # fallback
                    fallback = re.sub(r"[^\w\s,&.-]+$", "", cand).strip()
                    if fallback:
                        return fallback
        return None

    # Try payer header
    payer = find_after(lambda L: "payer's name" in L and "street address" in L)
    if payer:
        return payer

    # Fallback: recipient header
    recip = find_after(lambda L: "recipient's name" in L and "street address" in L)
    if recip:
        return recip

    return "1099-DIV"

#---------------------------1099-DIV----------------------------------#

def clean_bookmark(name: str) -> str:
    # Remove any trailing junk starting from 'Interest' and strip whitespace
    cleaned = re.sub(r"\bInterest.*$", "", name, flags=re.IGNORECASE)
    return cleaned.strip()
# 1099-SA

def clean_institution_name(raw: str) -> str:
    """
    Post-process extracted institution name.
    Keeps the full institution name like 'Optum Bank',
    'The Bank of New York Mellon', etc.
    Trims copyright, FDIC notes, and OCR garbage tails like "we Til SAS Ne ee".
    """
    import re, unicodedata

    if not raw:
        return "1099-SA"

    # --- Step 1: Unicode normalization and invisible cleanup ---
    text = unicodedata.normalize("NFKC", raw)
    text = text.encode("ascii", "ignore").decode("ascii")  # drop weird OCR chars
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)  # remove zero-width chars
    text = re.sub(r"\s+", " ", text).strip()

    # --- Step 2: Core extraction ---
    m = re.search(
        r"\b([A-Z][A-Za-z& ]{0,60}?(?:Bank|Trust|Credit Union|Financial Services|Savings)[A-Za-z& ]{0,60})\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip(" ,.-")
    else:
        name = text

    # --- Step 3: Remove known OCR garbage (robust pattern) ---
    name = re.sub(
        r"(?i)\bwe\s*[t1i|l]+\s*s[a4@]s+\s*n[e3]+\s*e[e3]*\b.*$",
        "",
        name,
    )

    # --- Step 4: Remove trailing punctuation or leftover junk ---
    name = re.sub(r"[\s,.\-]+$", "", name).strip()

    return name or "1099-SA"

def normalize_text(s: str) -> str:
    import re
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)  # collapse multiple spaces
    return s.strip().lower()


def is_junk_line(s: str) -> bool:
    """
    Return True if the line looks like IRS instructions or generic text,
    not a payer/institution name.
    """
    import re
    junk_patterns = [
        r"providing the trustee allows the repayment",
        r"you may repay a mistaken distribution",
        r"see the instructions",
        r"report the fmv",
        r"include the earnings",
        r"this information is being furnished",
        r"department of the treasury",
        r"internal revenue service",
        r"form 1099-sa",
        r"instructions for recipient",
        r"omb no",
        r"copy b",
    ]
    for pat in junk_patterns:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False


def extract_1099sa_bookmark(text: str) -> str:
    """
    Extract the payer/issuer name from 1099-SA text.
    Priority:
      0. Institution glued with 'Form 1099-SA From an HSA'
      1. Inline 'From an HSA, <institution>'
      1.5. First candidate after 'foreign postal code, and telephone'
      2. First candidate after header with address keywords
      3. Any line in whole text containing Bank/Trust/Credit Union/Equity/Corporate
      4. Fallback: 1099-SA
    """
    import re

    lines = text.splitlines()
    lower_lines = [normalize_text(L) for L in lines]
   
    skip_phrases = (
        "omb no",
        "form 1099-sa",
        "distributions",
        "recipient",
        "payer's tin",
        "recipient's tin",
        "account number",
        "street address",
        "city or town",
        "state or province",
        "zip",
        "telephone",
    )
    # --- Rule -1: Explicit overrides ---
    OVERRIDES = {
        "national financial services llc": "National Financial Services LLC",
        "national financial serves llc": "National Financial Services LLC",  # OCR fallback
        "bank of america": "Bank of America",  # ✅ added
        "bark of america": "Bank of America",  # common OCR typo
        "bank of amerlca": "Bank of America",  # OCR 'l'→'i' or 'I'
        "bank of amerlca na": "Bank of America",  # with suffix variation
    }
   
    normalized_text = normalize_text(text)
    for key, val in OVERRIDES.items():
        if key in normalized_text:
            return val
    # --- Rule 0: Handle glued "Form 1099-SA From an HSA" ---
    for L in lines:
        if re.search(r"form\s*1099-sa.*from an hsa", L, flags=re.IGNORECASE):
            cand = re.split(r"form\s*1099-sa", L, flags=re.IGNORECASE)[0].strip(" ,|-")
            if cand:
                return clean_institution_name(cand)

    # --- Rule 1: Inline "From an HSA, Optum Bank ..." ---
    for L in lines:
        match = re.search(r"from an hsa.*?(bank|trust|credit union|corporate)[^,]*", L, flags=re.IGNORECASE)
        if match:
            cand = match.group(0)
            cand = re.sub(r"from an hsa[, ]*", "", cand, flags=re.IGNORECASE)
            return clean_institution_name(cand)

    # --- Rule 1.5: Immediately after "foreign postal code, and telephone" ---
    for i, L in enumerate(lower_lines):
        if "foreign postal code, and telephone" in L:
            for offset in range(1, 4):  # look ahead up to 3 lines
                idx = i + offset
                if idx >= len(lines):
                    break
                candidate = lines[idx].strip()
                candidate_lower = normalize_text(candidate)

                if not candidate or len(candidate) <= 3:
                    continue
                if any(skip in candidate_lower for skip in skip_phrases) or is_junk_line(candidate_lower):
                    continue

                candidate = re.split(r"(form\s*1099-sa|from an hsa)", candidate, flags=re.IGNORECASE)[0].strip(" ,|-")
                if candidate:
                    return clean_institution_name(candidate)

    # --- Rule 2: After generic header line with address keywords ---
    for i, L in enumerate(lower_lines):
        if "country" in L and "zip" in L and "telephone" in L:
            candidates = []
            for j in range(i + 1, len(lines)):
                cand = lines[j].strip()
                cand_lower = normalize_text(cand)
                if not cand:
                    continue
                if any(skip in cand_lower for skip in skip_phrases) or is_junk_line(cand_lower):
                    continue
                cand = re.split(r"(form 1099-sa|from an hsa)", cand, flags=re.IGNORECASE)[0].strip(" ,|-")
                if cand:
                    candidates.append(cand)
                if re.search(r"\b(po box|p\.?o\.?|drive|street|road|ave|blvd)\b", cand_lower):
                    break
            for cand in candidates:
                if re.search(r"(bank|trust|credit union|equity|corporate)", cand, flags=re.IGNORECASE):
                    return clean_institution_name(cand)
            if candidates:
                return clean_institution_name(candidates[0])

    # --- Rule 3: Global scan for institution names ---
    for cand in lines:
        cand_norm = normalize_text(cand)
        if re.search(r"(bank|trust|credit union|equity|corporate)", cand_norm):
            if not is_junk_line(cand_norm):
                return clean_institution_name(cand)

    # --- Rule 4: Last-resort fallback ---
    return "1099-SA"




# 1099-SA
#---------------------------1098-Mortgage----------------------------------#
import re
from typing import List

def clean_bookmark(name: str) -> str:
    """Helper to normalize bookmark names."""
    name = name.strip()
    name = re.sub(r"[^\w\s.,&-]+$", "", name)  # strip trailing junk
    return name

def extract_1098mortgage_bookmark(text: str) -> str:
    """
    Extract lender name for Form 1098-Mortgage.
    Prints which rule fired for debugging.
    Detects lenders like 'NEWREZ LLC DBA SHELLPOINT MORTGAGE SERVICING'.
    """
    lines: List[str] = text.splitlines()
    lower_lines = [L.lower() for L in lines]

    # 1) Rocket Mortgage override
    for L in lines:
        if re.search(r"rocket\s+mortgage", L, flags=re.IGNORECASE):
            bookmark = "ROCKET MORTGAGE LLC"
            print(f"[1098-MORTGAGE] Rule: Rocket Mortgage override → {bookmark}", file=sys.stderr)
            return finalize_bookmark(bookmark)

    # 2) Dovenmuehle Mortgage override
    for L in lines:
        if re.search(r"dovenmuehle\s+mortgage", L, flags=re.IGNORECASE):
            bookmark = "DOVENMUEHLE MORTGAGE, INC"
            print(f"[1098-MORTGAGE] Rule: Dovenmuehle override → {bookmark}", file=sys.stderr)
            return finalize_bookmark(bookmark)

    # 3) Huntington National Bank override
    for L in lines:
        if re.search(r"\bhuntington\s+national\s+bank\b", L, flags=re.IGNORECASE):
            bookmark = "THE HUNTINGTON NATIONAL BANK"
            print(f"[1098-MORTGAGE] Rule: Huntington Bank override → {bookmark}", file=sys.stderr)
            return finalize_bookmark(bookmark)

    # 4) UNITED NATIONS FCU override
    for L in lines:
        if re.search(r"\bunited\s+nations\s+fcu\b", L, flags=re.IGNORECASE):
            bookmark = "UNITED NATIONS FCU"
            print(f"[1098-MORTGAGE] Rule: United Nations FCU override → {bookmark}", file=sys.stderr)
            return finalize_bookmark(bookmark)

    # 5) LOANDEPOT COM LLC override
    for L in lines:
        if re.search(r"\bloan\s*depot\s*com\s*llc\b", L, flags=re.IGNORECASE):
            bookmark = "LOANDEPOT.COM LLC"
            print(f"[1098-MORTGAGE] Rule: LoanDepot override → {bookmark}", file=sys.stderr)
            return finalize_bookmark(bookmark)

    # 6) JPMORGAN CHASE BANK, N.A.
    for L in lines:
        if re.search(r"jp\s*morgan\s+chase", L, flags=re.IGNORECASE):
            bookmark = "JPMORGAN CHASE BANK, N.A."
            print(f"[1098-MORTGAGE] Rule: JPMorgan Chase override → {bookmark}", file=sys.stderr)
            return finalize_bookmark(bookmark)
            # 🔹 NEW Rule: handle "RECIPIENT'S/LENDER'S name..." header pattern
    for i, line in enumerate(lines):
        lline = line.lower()

        if "recipient" in lline and "lender" in lline and "telephone" in lline:
            # check the same line for lender name (sometimes merged)
            if re.search(r"(bank|mortgage|servicing|loan|llc|fcu|credit|trust)", line, re.IGNORECASE):
                # Extract only up to the first "may not be fully deductible" or "OMB" etc.
                cleaned = re.split(
                    r"may\s+not\s+be\s+fully\s+deductible|OMB|Form|Department|Treasury|Caution",
                    line,
                    maxsplit=1,
                    flags=re.IGNORECASE
                )[0].strip(" *-,")

                # If name seems valid, finalize it
                if len(cleaned) > 5 and re.search(r"[A-Za-z]{3,}", cleaned):
                    print(f"[1098-MORTGAGE] Rule: Inline RECIPIENT/LENDER line → {cleaned}", file=sys.stderr)
                    return finalize_bookmark(cleaned)

            # otherwise look at next line (most common pattern)
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                nxt = re.sub(
                    r"may\s+not\s+be\s+fully\s+deductible.*|OMB.*|Form.*|Department.*|Treasury.*|Caution.*",
                    "",
                    nxt,
                    flags=re.IGNORECASE
                ).strip(" *-,")

                if re.search(r"(bank|mortgage|servicing|loan|llc|fcu|credit|trust)", nxt, re.IGNORECASE):
                    print(f"[1098-MORTGAGE] Rule: Next-line after RECIPIENT/LENDER header → {nxt}", file=sys.stderr)
                    return finalize_bookmark(nxt)


    for i, line in enumerate(lines):
        lline = line.lower()

        # match the header line that includes the phrase "foreign postal code, and telephone no."
        if "foreign postal code" in lline and "telephone" in lline:
            # check the same line for any lender name words (rare but possible)
            if re.search(r"(bank|mortgage|servicing|loan|llc|fcu|credit|trust|dba|company|corp)", line, re.IGNORECASE):
                cleaned = re.split(
                    r"limits\s+based|may\s+not\s+be\s+fully\s+deductible|OMB|Form|Department|Treasury|Caution",
                    line,
                    maxsplit=1,
                    flags=re.IGNORECASE
                )[0].strip(" *-,")
                if len(cleaned) > 5 and re.search(r"[A-Za-z]{3,}", cleaned):
                    print(f"[1098-MORTGAGE] Rule: Inline FOREIGN POSTAL line → {cleaned}", file=sys.stderr)
                    return finalize_bookmark(cleaned)

            # otherwise look at the next line (typical OCR pattern)
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()

                # remove OCR junk like “and the cost…” or “Form…”
                nxt = re.sub(
                    r"and\s+the\s+cost.*|Form.*|OMB.*|Department.*|Treasury.*|Caution.*|may\s+not\s+be\s+fully\s+deductible.*",
                    "",
                    nxt,
                    flags=re.IGNORECASE
                ).strip(" *-,")

                # check if line contains any lender indicators
                if re.search(r"(bank|mortgage|servicing|loan|llc|fcu|credit|trust|dba|company|corp)", nxt, re.IGNORECASE):
                    # optionally merge with next line if it continues (like "MORTGAGE SERVICING")
                    if i + 2 < len(lines):
                        nxt2 = lines[i + 2].strip()
                        if re.search(
                            r"(mortgage|servicing|bank|llc|fcu|credit|company|association|trust|loan)",
                            nxt2,
                            re.IGNORECASE,
                        ):
                            nxt = f"{nxt} {nxt2}"
                    print(f"[1098-MORTGAGE] Rule: Next-line after FOREIGN POSTAL header → {nxt}", file=sys.stderr)
                    return finalize_bookmark(nxt)



    # 9) FCU fallback
    for L in lines:
        if re.search(r"\bfcu\b", L, flags=re.IGNORECASE):
            m = re.search(r"(.*?FCU)\b", L, flags=re.IGNORECASE)
            bookmark = m.group(1) if m else L.strip()
            print(f"[1098-MORTGAGE] Rule: FCU fallback → {bookmark}", file=sys.stderr)
            return finalize_bookmark(bookmark)

    # 10) Default fallback
    print("[1098-MORTGAGE] Rule: Fallback default → 1098-Mortgage", file=sys.stderr)
    return finalize_bookmark("1098-Mortgage")


def finalize_bookmark(bookmark: str) -> str:
    """Final cleanup of extracted bookmark."""
    bookmark = clean_bookmark(bookmark)

    # Trim if known noise appears inside
    noise_markers = [
        "ang the cost",   # from OCR line you mentioned
        "not be fully deductible",
        "limits based on",
    ]
    for marker in noise_markers:
        if marker.lower() in bookmark.lower():
            bookmark = bookmark.split(marker, 1)[0].strip()

    return bookmark


def group_by_type(entries: List[Tuple[str,int,str]]) -> Dict[str,List[Tuple[str,int,str]]]:
    d=defaultdict(list)
    for e in entries: d[e[2]].append(e)
    return d
#---------------------------1098-Mortgage----------------------------------#
#---------------------------529-Plan ----------------------------------#
def extract_529_bookmark(text: str) -> str:
    # Try to detect Indiana or state-specific plan first
   

    # Fallback for generic 529 terms
    if re.search(r'\b529\b', text, re.IGNORECASE):
        return "529 Plan"

    # Default fallback
    return "529-Plan"

#---------------------------529-Plan ----------------------------------#
#5498-SA


def clean_bookmark(name: str) -> str:
    """Normalize bookmark string."""
    name = name.strip()
    name = re.sub(r"[^\w\s.&-]", "", name)
    return name

def extract_5498sa_bookmark(text: str) -> str:
    """
    Extract trustee/institution name for Form 5498-SA.
    Works even when the name is glued with address/ZIP or preceded by junk text.
    Cleans common OCR headers like 'Do Not Cut', 'Separate Forms on This Page', etc.
    """
    import re

    # --- Normalize spaces ---
    cleaned = text.replace("\n", " ").replace("  ", " ")

    # ✅ Step 1: Remove known noisy prefixes before the real institution
    cleaned = re.sub(
        r"(?i)\b(do\s+not\s+cut.*?|separate\s+forms?\s+on\s+this\s+page.*?|see\s+instructions\s+on\s+back.*?)\b(?=[A-Z])",
        "",
        cleaned
    ).strip()

    # --- Step 2: Common OCR misreads for 'Optum Bank' ---
    ocr_variants = [
        r"\bOptum\s*Bank\b",
        r"\bOptum\s*Ban[kc]\b",
        r"\bOptun\s*Bank\b",
        r"\bOptm\s*Bank\b",
        r"\bO[t]um\s*Bank\b",
        r"\btum\s*Bank\b",
        r"\bOptum\s*Bamk\b",
    ]
    for pattern in ocr_variants:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            return "Optum Bank"

    # --- Step 3: Handle glued 'OptumBank' or 'Optum Financial' ---
    if re.search(r"OptumBank", cleaned, re.IGNORECASE):
        return "Optum Bank"

    if re.search(r"Optum\s*Financial", cleaned, re.IGNORECASE):
        # Prefer Bank version if both words appear
        if "bank" in cleaned.lower():
            return "Optum Bank"
        else:
            return "Optum Financial"

    # --- Step 4: Look for any other trustee-like name (ConnectYourCare, HealthEquity, etc.) ---
    m = re.search(
        r"\b([A-Z][A-Za-z& ]{2,40}?(?:Care|Corporate|Corporation|Bank|Trust|LLC|Inc|Financial))\b",
        cleaned
    )
    if m:
        return m.group(1).strip()

    # --- Step 5: Backup: check lines after postal code header ---
    lines = text.splitlines()
    lower_lines = [L.lower() for L in lines]
    for i, header in enumerate(lower_lines):
        if "foreign postal code" in header and "telephone" in header:
            for cand in lines[i + 1:]:
                s = cand.strip()
                if not s:
                    continue
                # Skip numbers or contributions text
                if re.search(r"\d{2,}", s) or "contribution" in s.lower():
                    continue
                raw = re.sub(r"[^\w\s]+$", "", s)
                raw = re.split(
                    r"contributions\s+made\s+in\s+\d{4}.*",
                    raw, 1, flags=re.IGNORECASE
                )[0].strip()
                if raw:
                    return raw

    # --- Step 6: Fallback ---
    return "5498-SA"




#1098-T
def extract_1098t_bookmark(text: str) -> str:
    """
    Extract institution name for 1098-T forms.

    Rules:
    1. If any line contains "univ" or "university", return that full line (after cleaning).
    2. Otherwise: find a line after "foreign postal code ... qualified tuition"
       and take the next line if it looks like an institution.
    3. Fallback: scan for College, Institute, Academy, Board of Regents.
    4. If nothing found, return "1098-T".
    """

    import re
    lines = text.splitlines()

    # normalizer to fix OCR junk
    def normalize_institution(name: str) -> str:
        # Remove unwanted symbols and multiple spaces
        name = re.sub(r"[^\w\s.&-]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()

        # Fix OCR artifacts and common typos
        name = re.sub(r"\bUniv\b", "University", name, flags=re.IGNORECASE)
        name = re.sub(r"\bTuiti\b", "Tuition", name, flags=re.IGNORECASE)
        name = re.sub(r"\bTution\b", "Tuition", name, flags=re.IGNORECASE)

        # --- Trim unwanted student/year/form fragments ---
        # Remove leading student info before " - University" or similar
        name = re.sub(r"^.*?-\s*(University|College|Institute|Academy|Board of Regents)", r"\1", name, flags=re.IGNORECASE)

        # Remove trailing year/form text like "2022 1098-T" or "Form 1098-T"
        name = re.sub(r"\b(19|20)\d{2}\b.*", "", name)              # remove trailing year + extras
        name = re.sub(r"\bForm\s*1098[-\s]*T.*", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\b1098[-\s]*T.*", "", name, flags=re.IGNORECASE)

        return name.strip()

    KEYWORDS = r"(University|College|Institute|Academy|Univ|Board of Regents|Tuition|Tuiti|Tution)"

    # 🔹 Rule 1: any line with "univ" or "university"
    for line in lines:
        if re.search(r"\b(univ|university)\b", line, flags=re.IGNORECASE):
            return normalize_institution(line)

    lower_lines = [l.lower() for l in lines]

    # 🔹 Rule 2: look for header, then next line
    for i, L in enumerate(lower_lines):
        if "foreign postal code" in L and "qualified tuition" in L:
            if i + 1 < len(lines):
                cand = lines[i + 1].strip()
                if re.search(KEYWORDS, cand, flags=re.IGNORECASE):
                    return normalize_institution(cand)

    # 🔹 Rule 3: fallback scan for other institution markers
    for line in lines:
        if re.search(KEYWORDS, line, flags=re.IGNORECASE):
            return normalize_institution(line)

    # 🔹 Rule 4: nothing found
    return "1098-T"

def print_pdf_bookmarks(path: str):
    try:
        reader = PdfReader(path)
        outlines = reader.outlines
        print(f"\n--- Bookmark structure for {os.path.basename(path)} ---")
        def recurse(bms, depth=0):
            for bm in bms:
                if isinstance(bm, list):
                    recurse(bm, depth+1)
                else:
                    title = getattr(bm, 'title', str(bm))
                    print("  " * depth + f"- {title}")
        recurse(outlines)
    except Exception as e:
        logger.error(f"Error reading bookmarks from {path}: {e}")
       
# ---------------------- Existing ------------------------


# ✅ ADD THIS RIGHT BELOW (new helper)
def cleanup_provider_name(name: str) -> str:
    """
    Trim addresses, ZIP codes, or EIN text from daycare/preschool names.
    Example:
      'STEMsteps 3281 Wexford Rd Gibsonia, PA 15044 EIN' -> 'STEMsteps'
    """
    import re
    if not name:
        return name

    cleaned = name.strip()
    cleaned = re.sub(r"\s+\d{3,}.*", "", cleaned)  # remove any address part
    cleaned = re.sub(
        r"\b(EIN|Zip|Address|Rd|Road|Street|St|Ave|Avenue|Blvd|Boulevard|Drive|Dr|PA|IL|TX|CA|NJ)\b.*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[,.\-]+$", "", cleaned).strip()
    return cleaned

   
def extract_daycare_bookmark(text: str) -> str:
    """
    Extract the daycare, preschool, or child care provider name
    from tuition/payment statements for Child Care Expenses.
    Returns a readable name like 'STEMsteps' or 'Kiddie Care'.
    """
    import re

    cleaned = text.replace("\n", " ").replace("  ", " ")
    lower = cleaned.lower()

    # 🔹 1) Kiddie Care override — detect by email or text
    if any(v in lower for v in ("mykiddiecare", "kiddiecare", "kiddecare", "kiddie care", "kidde care")):
        return "Kiddie Care"

    # 🔹 2) Provider Information header
    m = re.search(r"provider information[:\s]+([A-Z][A-Za-z0-9&\-,.' ]{2,60})", cleaned, re.IGNORECASE)
    if m:
        name = cleanup_provider_name(m.group(1))
        if name:
            return name

    # 🔹 3) Look for school/daycare-like institution names
    m = re.search(
        r"\b([A-Z][A-Za-z0-9&',.()\- ]{2,80}?(?:School|Schools|Academy|Learning|Center|Preschool|Daycare|Montessori|Care|Steps))\b",
        cleaned,
    )
    if m:
        return cleanup_provider_name(m.group(1))

    # 🔹 4) Fallback – scan for capitalized lines with “care”, “steps”, etc.
    lines = [L.strip() for L in text.splitlines() if L.strip()]
    for L in lines:
        if re.search(r"[A-Z][a-z]{2,}", L) and not re.search(r"\d{3,}", L):
            if any(x in L.lower() for x in ["school", "academy", "learning", "center", "care", "montessori", "steps"]):
                return cleanup_provider_name(L)

    # 🔹 5) Last resort – detect “Re:” or letterhead-style name
    for L in lines[:10]:
        if re.search(r"re:\s*(.+)", L, re.IGNORECASE):
            name = re.sub(r"re:\s*", "", L, flags=re.IGNORECASE).strip()
            return cleanup_provider_name(name)
        if re.search(r"public schools|academy|learning center|montessori|daycare|preschool|steps", L, re.IGNORECASE):
            return cleanup_provider_name(L)

    # 🔹 6) Final fallback
    return "Child Care Provider"



# ── Merge + bookmarks + multi-method extraction
nek = None
def classify_div_int(text: str) -> str | None:
    """
    Classify a page as 1099-DIV or 1099-INT if it matches the required
    header lines. Returns "1099-DIV", "1099-INT", or None.
    """
    lower = text.lower()

    div_match = (
        "1099-div" in lower
        and "dividends & distributions" in lower
        and "ordinary dividends" in lower
        and "description cusippay" in lower
    )
    int_match = (
        "1099-int" in lower
        and "interest income" in lower
        and "description cusippay" in lower
    )

    if div_match:
        return "1099-DIV"
    elif int_match:
        return "1099-INT"
    return None
#SSn for TP & SP
def detect_ssn_owner(text: str, tp_ssn: str, sp_ssn: str) -> str:
    """
    Detect whether a page belongs to the taxpayer or spouse based on SSN digits.
    Returns "TP", "SP", or "".
    """
    if not text or (not tp_ssn and not sp_ssn):
        return ""
 
    t = re.sub(r'\D', '', text)  # keep only digits
    if tp_ssn and tp_ssn in t:
        return "TP"
    if sp_ssn and sp_ssn in t:
        return "SP"
    return ""
# Tracks merged PDF page numbers for each real (path, idx) K-1 page
k1_page_map = {}
def reorder_k1_pages(pages):
    """
    Sort K-1 pages in correct IRS order:
    1. Main K-1 form page
    2. Partner/shareholder information
    3. Supplemental info
    4. Statement A (QBI)
    5. Section 199A
    6. Rental activity
    7. Basis worksheets
    8. Continuation statements
    """

    def get_score(text):
        t = text.lower()

        # 1. Main federal K-1 page
        if "schedule k-1" in t and ("form 1065" in t or "form 1120-s" in t or "form 1041" in t):
            return 1

        # 2. Personal info sections
        if "information about the partner" in t:
            return 2
        if "information about the shareholder" in t:
            return 2
        if "partner's share" in t:
            return 3
        if "shareholder's share" in t:
            return 3

        # 3. Supplement pages
        if "supplemental information" in t:
            return 4
        if "additional information" in t:
            return 4

        # 4. QBI pages
        if "statement a" in t:
            return 5
        if "section 199a" in t:
            return 5
        if "qbi" in t:
            return 5

        # 5. Rental real estate pages
        if "rental real estate activity" in t:
            return 6

        # 6. Basis worksheets
        if "basis worksheet" in t:
            return 7
        if "at-risk basis" in t:
            return 7

        # 7. Continuation statements
        if "continuation" in t:
            return 8

        # 8. Fallback → leave last
        return 99

    # Score every page
    scored = []
    for (p, idx, _) in pages:
        text = extract_text(p, idx)
        score = get_score(text)
        scored.append((score, p, idx))

    # Sort by score only (KEEP original order if score ties)
    scored.sort(key=lambda x: x[0])

    # Return sorted as original format
    return [(p, idx, "K-1") for (_, p, idx) in scored]
def k1_page_priority(text):
    t = text.lower()

    # --- TRUE MAIN K-1 FORM PAGE ---
    if (
        "schedule k-1" in t
        and ("form 1065" in t or "form 1120-s" in t or "form 1041" in t)
        and (
            "information about the partner" in t
            or "information about the shareholder" in t
            or ("part i" in t and "information about" in t)
        )
    ):
        return 1

    # Supplemental info (NOT main)
    if "supplemental information" in t:
        return 3

    # Rental real estate pages
    if "rental real estate" in t:
        return 4

    # QBI / Section 199A
    if (
        "statement a" in t
        or "section 199a" in t
        or "qualified business income" in t
        or "qbi" in t
    ):
        return 5

    # Other continuation
    if "continuation" in t:
        return 6

    # Fallback last
    return 99
def extract_k1_company(text: str) -> str | None:
    # Normalize line breaks and spaces
    clean = re.sub(r"[^\w\s,&.'\-]", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()

    # 1. Strongest pattern: any line ending with LLC/LP/INC/CORP/etc.
    multi = re.findall(
        r"([A-Z][A-Z\s,&.'\-]{2,80}?(?:LLC|L\.L\.C\.|INC|CORP|LP|L\.P\.|LLP|FUND))",
        text,
        flags=re.IGNORECASE,
    )
    if multi:
        # choose longest = most complete name
        return max(multi, key=len).strip()

    # 2. Multi-line OCR-broken names: join consecutive lines
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        joined = (lines[i] + " " + lines[i+1]).strip()
        m = re.search(
            r"([A-Z][A-Z\s,&.'\-]{2,80}?(?:LLC|INC|CORP|LP|LLP|FUND))",
            joined,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()

    # 3. fallback for things like “Desire Homes North Salem LLC”
    words = clean.split()
    for i in range(len(words)):
        for j in range(i+2, min(i+8, len(words))):
            segment = " ".join(words[i:j])
            if re.search(r"(LLC|INC|CORP|LP|LLP)$", segment, flags=re.IGNORECASE):
                return segment

    return None

def clean_k1_company_name(raw: str) -> str:
    """
    Clean K-1 company name by:
    - Removing multi-line noise ('Name of Partnership', 'Partnership EIN', etc.)
    - Keeping only the longest block ending with LLC/LP/INC/CORP
    - Removing leading/trailing garbage
    """

    if not raw:
        return None

    import re

    # Collapse newlines → spaces
    raw = raw.replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw).strip()

    # Remove known garbage phrases
    garbage = [
        "name of partnership",
        "partnership ein",
        "information about the partner",
        "information about the shareholder",
        "schedule k-1",
        "form 1065",
        "form 1120-s",
        "form 1041"
    ]

    lower = raw.lower()
    for g in garbage:
        if g in lower:
            # Remove the entire garbage phrase + anything BEFORE it
            idx = lower.index(g)
            raw = raw[idx + len(g):].strip()
            lower = raw.lower()

    # Strong match: pick the longest LLC/INC/CORP/LP ending
    m = re.findall(
        r"[A-Za-z0-9& ,.'\-]{3,100}?(?:LLC|L\.L\.C\.|INC|CORP|LP|LLP)",
        raw,
        flags=re.IGNORECASE
    )
    if m:
        # choose longest match
        return max(m, key=len).strip()

    return raw.strip()

# ── Merge + bookmarks + cleanup
def merge_with_bookmarks(input_dir: str, output_pdf: str, tp_ssn: str = "", sp_ssn: str = ""):
    # Prevent storing merged file inside input_dir
    abs_input = os.path.abspath(input_dir)
    abs_output = os.path.abspath(output_pdf)
    if abs_output.startswith(abs_input + os.sep):
        abs_output = os.path.join(os.path.dirname(abs_input), os.path.basename(abs_output))
        logger.warning(f"Moved output outside: {abs_output}")
    # ✅ Collect all candidate files
    all_files = sorted(
        f for f in os.listdir(abs_input)
        if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff'))
        and f != os.path.basename(abs_output)
    )
    import hashlib

    # --- Detect duplicate PDFs by MD5 hash ---
    hash_map = {}         # md5 -> first filename
    duplicate_files = []  # list of duplicate filenames

    for f in all_files:
        path = os.path.join(abs_input, f)
        try:
            with open(path, "rb") as fh:
                md5 = hashlib.md5(fh.read()).hexdigest()
            if md5 in hash_map:
                duplicate_files.append(f)
            else:
                hash_map[md5] = f
        except Exception as e:
            print(f"⚠️ Could not hash {f}: {e}", file=sys.stderr)

    # Keep only unique files for processing
    files = sorted(hash_map.values())
    logger.info(f"Found {len(files)} unique files, {len(duplicate_files)} duplicates.")

   
   # remove any zero‐byte files so PdfReader never sees them
    files = []
    for f in all_files:
        p = os.path.join(abs_input, f)
        if os.path.getsize(p) == 0:
           logger.warning(f"Skipping empty file: {f}")
           continue
        files.append(f)
    # 🔄 Convert images into PDFs so the rest of the pipeline sees only PDFs
    converted_files = []
    for f in list(files):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            path = os.path.join(abs_input, f)
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pdf_path = os.path.splitext(path)[0] + "_conv.pdf"
                img.save(pdf_path, "PDF", resolution=300.0)
                print(f"🖼️ Converted {f} → {os.path.basename(pdf_path)}", file=sys.stderr)

                # replace the image with its new PDF
                files.remove(f)
                files.append(os.path.basename(pdf_path))
                converted_files.append(pdf_path)
            except Exception as e:
                print(f"❌ Failed to convert {f}: {e}", file=sys.stderr)

    logger.info(f"Found {len(files)} files in {abs_input}")

    income, expenses, others = [], [], []
    # what bookmarks we want in workpapaer shoudl be add in this
    w2_titles = {}
    int_titles = {}
    div_titles = {} # <-- Add this line
    sa_titles = {}  
    mort_titles = {}
    sa5498_titles = {}
    t529_titles = {}
    t1098_titles = {}
    account_pages = {}  # {account_number: [(path, page_index, 'Consolidated-1099')]}
    account_names = {}
    k1_pages = {}        # {ein: [(path, page_index, 'K-1')]}
    k1_names = {}        # {ein: 'Partnership name'}
    k1_form_type = {}    # {ein: '1065' or '1120S' or '1041'}


    # ✅ Track seen page text hashes to detect duplicate pages (within or across files)
    seen_hashes = {}   # tracks duplicate page text
    seen_pages = {}    # tracks appended pages


    # --- Skip duplicates in main processing ---
    files = [f for f in files if f not in duplicate_files]

    for fname in files:
        last_ein_seen = None
        seen_pages = {}   # reset duplicate tracker per file
        path = os.path.join(abs_input, fname)
        if fname.lower().endswith('.pdf'):
            total = len(PdfReader(path).pages)
            for i in range(total):
                print("=" * 400, file=sys.stderr)
                print(f"Processing: {fname}, Page {i+1}", file=sys.stderr)

                # ── Print header before basic extract_text
                print("→ extract_text() output:", file=sys.stderr)
                try:
                    text = extract_text(path, i)
                    # --- 🆕 Detect duplicate pages across all PDFs ---
                    import hashlib

                    page_hash = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
                    file_key = (fname, page_hash)   # <-- isolate by filename
                    if file_key in seen_hashes:
                        print(f"[DUPLICATE PAGE] {fname} p{i+1} matches ...", file=sys.stderr)
                        others.append((path, i, "Duplicate"))
                        continue  # ⬅️ SKIPS classification and appending
                    else:
                        seen_hashes[file_key] = (path, i)


                    print(text or "[NO TEXT]", file=sys.stderr)
                except Exception as e:
                    print(f"[ERROR] extract_text failed: {e}", file=sys.stderr)

                print("=" * 400, file=sys.stderr)

                # Multi-method extraction
                extracts = {}

                print("→ PDFMiner:", file=sys.stderr)
                try:
                    extracts['PDFMiner'] = pdfminer_extract(path, page_numbers=[i], laparams=PDFMINER_LA_PARAMS) or ""
                    print(extracts['PDFMiner'], file=sys.stderr)
                except Exception as e:
                    extracts['PDFMiner'] = ""
                    print(f"[ERROR] PDFMiner failed: {e}", file=sys.stderr)

               

                print("→ Tesseract OCR:", file=sys.stderr)
                try:
                    img = pdf_page_to_image(path, i, dpi=150)  # ✅ use your PyMuPDF helper
                    extracts['Tesseract'] = pytesseract.image_to_string(img, config="--psm 6") or ""
                    print(extracts['Tesseract'], file=sys.stderr)
                except Exception as e:
                    extracts['Tesseract'] = ""
                    print(f"[ERROR] Tesseract failed: {e}", file=sys.stderr)

                print("→ pdfplumber:", file=sys.stderr)
                try:
                    with pdfplumber.open(path) as pdf:
                        extracts['pdfplumber'] = pdf.pages[i].extract_text() or ""
                        print(extracts['pdfplumber'], file=sys.stderr)
                except Exception as e:
                    extracts['pdfplumber'] = ""
                    print(f"[ERROR] pdfplumber failed: {e}", file=sys.stderr)

                print("→ PyMuPDF (fitz):", file=sys.stderr)
                try:
                    doc = fitz.open(path)
                    extracts['PyMuPDF'] = doc.load_page(i).get_text()
                    doc.close()
                    print(extracts['PyMuPDF'], file=sys.stderr)
                except Exception as e:
                    extracts['PyMuPDF'] = ""
                    print(f"[ERROR] PyMuPDF failed: {e}", file=sys.stderr)

                print("=" * 400, file=sys.stderr)
             

                # Collect W-2 employer names across methods
                info_by_method, names = {}, []
                for method, txt in extracts.items():
                    cat, ft = classify_text(txt)
                    if cat == 'Income' and ft == 'W-2':
                        info = parse_w2(txt)
                        if info['employer_name'] != 'N/A':
                            info_by_method[method] = info
                            names.append(info['employer_name'])
                    # --- 1099-INT bookmark extraction ---
                    if cat == 'Income' and ft == '1099-INT':
                        title = extract_1099int_bookmark(txt)
                        if title and title != '1099-INT':
                            int_titles[(path, i)] = title
                    # <<< new DIV logic
                    if cat == 'Income' and ft == '1099-DIV':
                        title = extract_1099div_bookmark(txt)
                        if title and title != '1099-DIV':
                            div_titles[(path, i)] = title
                    if cat == 'Income' and ft == '1099-SA':
                        title = extract_1099sa_bookmark(txt)
                        if title and title != '1099-SA':
                            sa_titles[(path, i)] = title

                    if cat == 'Expenses' and ft == '1098-Mortgage':
                        title = extract_1098mortgage_bookmark(txt)
                        if title and title != '1098-Mortgage':
                            mort_titles[(path, i)] = title
                    if cat == 'Expenses' and ft == '5498-SA':
                        title = extract_5498sa_bookmark(txt)
                        if title and title != '5498-SA':
                            sa5498_titles[(path, i)] = title
                           
                    if cat == 'Expenses' and ft == '1098-T':
                        title = extract_1098t_bookmark(txt)
                        if title and title != '1098-T':
                            t1098_titles[(path, i)] = title
                    if cat == 'Expenses' and ft == '529-Plan':
                        title = extract_529_bookmark(txt)
                        if title and title != '529-Plan':
                            t529_titles[(path, i)] = title

                if names:
                    common = Counter(names).most_common(1)[0][0]
                    chosen = next(m for m,i in info_by_method.items() if i['employer_name'] == common)
                    print(f"--- Chosen employer ({chosen}): {common} ---", file=sys.stderr)
                    print_w2_summary(info_by_method[chosen])
                    w2_titles[(path, i)] = common

                # Classification & grouping
                    # … after you’ve extracted text …
                   # NEW: {acct: "Issuer Name"}

                tiered = extract_text(path, i)
                acct_num = extract_account_number(tiered)

                # --- Detect Schedule K-1 / Form 1065 / Statement A / QBI pages ---
                # --- Detect Schedule K-1 / Form 1065 / and related pages (QBI, Statement A, etc.) ---
                text_lower = tiered.lower()
                ein_num = extract_ein_number(tiered)

                # --- Detect Schedule K-1 / Form 1065 / Statement A / QBI / worksheet pages ---
                text_lower = tiered.lower()
                ein_num = extract_ein_number(tiered)

                # ✅ If EIN missing (common on continuation or basis pages), reuse the previous one
                # Track EIN across pages
                if not ein_num:
                    ein_num = last_ein_seen
                else:
                    last_ein_seen = ein_num


                # ✅ Detect any K-1-related page (expanded keyword list)
                k1_keywords = [
                    "schedule k-1", "form 1065", "form 1120-s", "form 1041",
                    "statement a", "qualified business income", "section 199a",
                    "qbi pass-through", "qbi pass through entity reporting",
                    # new worksheet / continuation phrases
                    "basis worksheet", "partner's basis worksheet",
                    "allocation of losses and deductions",
                    "partner's share of income", "at-risk basis", "section 704(c)",
                    "k-1 supplemental information", "continuation statement",
                    "additional information from schedule k-1", "passive activity purposes",
                    
                ]
                is_k1_page = any(k in text_lower for k in k1_keywords)

                # NEW — Detect TRUE form type
                is_1120s = bool(re.search(r"schedule\s*k-1.*form\s*1120[-\s]*s", text_lower))
                is_1065 = bool(re.search(r"schedule\s*k-1.*form\s*1065", text_lower))
                
                is_1041 = bool(re.search(r"schedule\s*k-1.*form\s*1041", text_lower))

                if (is_1065 or is_1120s or is_1041 or is_k1_page) and ein_num:

                # Assign correct form type
                    k1_form_type[ein_num] = (
                        "1065" if is_1065 else
                        "1120S" if is_1120s else
                        "1041" if is_1041 else
                        k1_form_type.get(ein_num, "1065")
                    )

                    # Extract company / partnership name
                    company = extract_k1_company(tiered)



                    k1_pages.setdefault(ein_num, []).append((path, i, "K-1"))
                    if company:
                        company = clean_k1_company_name(company)
                        k1_names[ein_num] = company





                    print(
                        f"[DEBUG] Unified K-1 page: {os.path.basename(path)} p{i+1} "
                        f"EIN={ein_num}, Company={company}, Form={k1_form_type[ein_num]}",
                        file=sys.stderr
                    )


                lower_text = tiered.lower()

                # --- Only add to Consolidated-1099 if it's truly a 1099 form ---
                if acct_num:
                    # Check if it's a 1099 (INT, DIV, B, MISC, etc.)
                    if "1099" in lower_text and not re.search(r"1098[-\s]*t", lower_text, re.IGNORECASE):
                        account_pages.setdefault(acct_num, []).append((path, i, "Consolidated-1099"))

        # Capture issuer name if present
                        issuer = extract_consolidated_issuer(tiered)
                        if issuer:
                            account_names.setdefault(acct_num, issuer)

                        print(f"[DEBUG] {os.path.basename(path)} p{i+1}: Added to Consolidated-1099 (acct={acct_num})", file=sys.stderr)
                    else:
                        # Skip non-1099 forms (e.g., 1098-T, 1098-Mortgage, etc.)
                        print(f"[DEBUG] {os.path.basename(path)} p{i+1}: Skipped Consolidated-1099 (acct={acct_num}, form=non-1099)", file=sys.stderr)

# Always classify after account checks
                cat, ft = classify_text(tiered)

               
                # NEW: log every classification
                print(
                    f"[Classification] {os.path.basename(path)} p{i+1} → "
                    f"Category='{cat}', Form='{ft}', "
                    f"snippet='{tiered[:150].strip().replace(chr(80),' ')}…'",
                    file=sys.stderr
                )

                entry = (path, i, ft)
                if cat == 'Income':
                    income.append(entry)
                elif cat == 'Expenses':
                    expenses.append(entry)
                else:
                    others.append(entry)

        else:
            # Image handling
            print(f"\n=== Image {fname} ===", file=sys.stderr)
            oi = extract_text_from_image(path)
            print("--- OCR Image ---", file=sys.stderr)
            print(oi, file=sys.stderr)
            cat, ft = classify_text(oi)
            entry = (path, 0, ft)
            if cat == 'Income':
                income.append(entry)
            elif cat == 'Expenses':
                expenses.append(entry)
            else:
                others.append(entry)

        # ---- K-1 (Form 1065) grouping synthesis ----
    k1_payload = {}      # key -> list of (path, page_index, 'K-1')
    k1_pages_seen = set()  # to mark pages already grouped
    # --- 🩹 Move all raw K-1 entries into k1_pages before grouping ---
    for (path, idx, form) in list(income):
        if form == "K-1":
            text = extract_text(path, idx).lower()
            ein = extract_ein_number(text)
            k1_type = k1_form_type.get(ein, "1065")  

            if not ein and k1_pages:
                ein = list(k1_pages.keys())[-1]   # reuse last EIN if missing
            if ein:
                k1_pages.setdefault(ein, []).append((path, idx, "K-1"))
                # remove from income so it doesn't create a stray K-1 later
                income.remove((path, idx, form))

    for ein, pages in k1_pages.items():
        if pages:
            key = f"K1::{k1_type}::{ein}"
            # store real pages
            k1_payload[key] = [(p, i, "K-1") for (p, i, _) in pages]
            # remember them so we don’t process twice
            for (p, i, _) in pages:
                k1_pages_seen.add((p, i))
            # add single synthetic entry
            income.append((key, -1, "K-1"))
           # remove synthetic K-1 entries before sorting
           #income = [e for e in income if not str(e[0]).startswith("K1::")]

            print(f"[DEBUG] Added unified K-1 package for EIN={ein} with {len(pages)} page(s)", file=sys.stderr)

    # 🧹 remove any individual K-1 pages that were grouped
    income = [
        e for e in income
        if not (e[2] == "K-1" and not e[0].startswith("K1::") and (e[0], e[1]) in k1_pages_seen)
    ]
    print(f"[DEBUG] Cleaned raw K-1 pages; synthetic K1 groups = {[x[0] for x in income if 'K1::' in x[0]]}", file=sys.stderr)


            

    # ---- Consolidated-1099 synthesis (insert this BEFORE income.sort(...)) ----
    consolidated_payload = {}        # key -> list of real page entries
    consolidated_pages = set()       # pages already placed under Consolidated-1099
    # Track pages we already decided are "Unused" so we don't touch them again
    unused_pages: set[tuple[str, int]] = set()


    for acct, pages in account_pages.items():
        if len(pages) <= 1:
            continue  # only group repeated accounts
        key = f"CONSOLIDATED::{acct}"
        consolidated_payload[key] = [(p, i, "Consolidated-1099") for (p, i, _) in pages]
        for (p, i, _) in pages:
            consolidated_pages.add((p, i))
    # add a synthetic income row that will sort using priority of 'Consolidated-1099'
        income.append((key, -1, "Consolidated-1099"))
# --------------------------------------------------------------------------
    # REMOVE synthetic K-1 entries ONLY FOR SORTING
    clean_income = [e for e in income if not str(e[0]).startswith("K1::")]

    # Now sort only REAL pages
    clean_income.sort(key=lambda e:(get_form_priority(e[2],'Income'), e[0], e[1]))

        # Restore synthetic K-1 groups after sorting
    # (They must be present so K-1 bookmark block runs)
    clean_income += [e for e in income if str(e[0]).startswith("K1::")]

    income = clean_income

    # Sort
    income.sort(key=lambda e:(get_form_priority(e[2],'Income'), e[0], e[1]))
    expenses.sort(key=lambda e:(get_form_priority(e[2],'Expenses'), e[0], e[1]))
    # merge & bookmarks
    merger = PdfMerger()
    page_num = 0
    stop_after_na = False
    import mimetypes
    #seen_pages = set()
    #def append_and_bookmark(entry, parent, title, with_bookmark=True):
    
    def append_and_bookmark(entry, parent, title, with_bookmark=True):
        nonlocal page_num, seen_pages
        global k1_page_map

        p, idx, _ = entry

        # Synthetic K-1 key expansion -------------------------
        if p.startswith("K1::"):
            key = p
            real_entries = k1_payload.get(key, [])
            if not real_entries:
                return

            for i, real_entry in enumerate(real_entries):
                append_and_bookmark(
                    real_entry,
                    parent,
                    title if i == 0 and with_bookmark else "",
                    with_bookmark=(i == 0 and with_bookmark),
                )
            return

        # Skip duplicates -------------------------------------
        sig = (p, idx)
        if sig in seen_pages:
            return
        seen_pages[sig] = True

        # Write single page temp PDF --------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            w = PdfWriter()
            w.add_page(PdfReader(p).pages[idx])
            w.write(tmp)
            tmp.flush()
            tmp_path = tmp.name

        with open(tmp_path, 'rb') as fh:
            merger.append(fileobj=fh)
        os.unlink(tmp_path)

        # ⭐ Record the merged PDF page number for K-1 mapping
        # Record merged position for every REAL page (ignore synthetic)
        if not p.startswith("K1::"):
            k1_page_map[(p, idx)] = page_num


        # Add bookmark
        if with_bookmark and title:
            merger.add_outline_item(title, page_num, parent=parent)

        page_num += 1

    # --- Unified K-1 bookmark builder ----------------------------------------
    def build_k1_bookmarks(merger, root, k1_pages, k1_names,
                       extract_text, append_and_bookmark, is_unused_page):
        """
        CORRECT K-1 BUILDER
    Ensures:
    - Pages for each EIN are SORTED BEFORE merging
    - Form 1065 / 1120-S / 1041 folders appear in correct order
    - No pages are appended before sorting
    """
        global k1_page_map

        # 1. Create top-level K-1 folder
        k1_root = merger.add_outline_item("K-1", 0, parent=root)

    # 2. Determine first-page anchor for each EIN (based on file order)
        ein_first_order = {
            ein: min(idx for (_, idx, _) in pages)
            for ein, pages in k1_pages.items()
        }

        # 3. Detect form type order (1065,1120-S,1041)
        form_first_page = {}
        for ein, pages in k1_pages.items():
            form = k1_form_type.get(ein, "1065")
            first_idx = min(idx for (_, idx, _) in pages)
            if form not in form_first_page or first_idx < form_first_page[form]:
                form_first_page[form] = first_idx

        # 4. Sort form folders in true order
        sorted_forms = sorted(form_first_page, key=lambda f: form_first_page[f])

        # 5. Create folders
        form_roots = {}
        for form in sorted_forms:
            # find the first merged page for this form
            ein_for_form = [
                ein for ein in k1_pages
                if k1_form_type.get(ein, "1065") == form
            ]

            if ein_for_form:
                first_ein = ein_for_form[0]
                first_page = min(
                    k1_page_map.get((p, idx), 999999)
                    for (p, idx, _) in k1_pages[first_ein]
                )
            else:
                first_page = 0  # fallback

            form_roots[form] = merger.add_outline_item(
                f"Form {form}",
                first_page,
                parent=k1_root
            )


        # 6. NOW handle each EIN group
        for ein in sorted(k1_pages, key=lambda e: ein_first_order[e]):
            pages = k1_pages[ein]
            company = k1_names.get(ein, f"EIN {ein}")
            form_type = k1_form_type.get(ein, "1065")

            # ---- SORT PAGES FIRST (THE MOST IMPORTANT PART) ----
            sorted_pages = sorted(
                pages,
                key=lambda x: k1_page_priority(extract_text(x[0], x[1]))
            )

        # Find anchor = main K-1 form page
            anchor_path, anchor_idx = None, None
            for (p, idx, _) in sorted_pages:
                text = extract_text(p, idx).lower()
                if "schedule k-1" in text and ("form 1065" in text or "form 1120-s" in text or "form 1041" in text):
                    anchor_path, anchor_idx = p, idx
                    break
            # fallback
            if anchor_path is None:
                anchor_path, anchor_idx = sorted_pages[0][0], sorted_pages[0][1]


            # Create the EIN bookmark
            # convert original anchor index → merged page index
            anchor_page = k1_page_map.get((anchor_path, anchor_idx), 0)

            clean = extract_k1_company(company)
            comp_node = merger.add_outline_item(
                f"{clean} (EIN {ein})",
                anchor_page,
                parent=form_roots[form_type]
            )


            # ---- NOW append pages IN SORTED ORDER ----
            for (p, idx, _) in sorted_pages:
                txt = extract_text(p, idx)
                if is_unused_page(txt.lower()):
                    continue
                append_and_bookmark((p, idx, "K-1"), comp_node, "", with_bookmark=False)


    # ── Bookmarks
   
    if income:
        root = merger.add_outline_item('Income', page_num)
            # --- Custom hierarchical K-1 bookmark structure ---
        processed_pages = set()  # ✅ Track K-1/QBI pages already appended



        groups = group_by_type(income)
        for form, grp in sorted(groups.items(), key=lambda kv: get_form_priority(kv[0], 'Income')):
                # ✅ Run the unified K-1 block FIRST
            if form == 'K-1':

                # STEP 1 — Append ALL real K-1 pages FIRST (no bookmarks yet)
                for ein, pages in k1_pages.items():
                    sorted_pages = sorted(
                        pages,
                        key=lambda x: k1_page_priority(extract_text(x[0], x[1]))
                    )
                    for (p, idx, _) in sorted_pages:
                        append_and_bookmark((p, idx, "K-1"), None, "", with_bookmark=False)

                # STEP 2 — NOW build bookmarks (page numbers now exist)
                build_k1_bookmarks(
                    merger, root,
                    k1_pages, k1_names,
                    extract_text, append_and_bookmark, is_unused_page
                )

                continue


            if stop_after_na:
                break
            if form == 'Consolidated-1099':
                cons_root = merger.add_outline_item('Consolidated-1099', page_num, parent=root)

                for entry in filtered_grp:
                    key, _, _ = entry
                    acct = key.split("::", 1)[1]

                    issuer = account_names.get(acct)
                    issuer = alias_issuer(issuer) if issuer else None
                    forms_label = issuer or f"Account {acct}"
                    forms_node = merger.add_outline_item(forms_label, page_num, parent=cons_root)

                    real_entries = consolidated_payload.get(key, [])

        # (optional context labels — does NOT skip appends)
             

        # ALWAYS append the real pages
                    for real_entry in real_entries:
                        page_text = extract_text(real_entry[0], real_entry[1])
                        if is_unused_page(page_text):
                            print(f"[DROP?] {os.path.basename(real_entry[0])} page {real_entry[1]+1} "
                                    f"marked as UNUSED", file=sys.stderr)
                            others.append((real_entry[0], real_entry[1], "Unused"))
                            #append_and_bookmark(real_entry, forms_node, "Unused")
                            continue

    # 1️⃣ First, check strong classifier
                        form_type = classify_div_int(page_text)

                        if form_type == "1099-DIV":
                            append_and_bookmark(real_entry, forms_node, "1099-DIV Description")

        # 2️⃣ Also check for other forms on same page
                            extra_forms = [ft for ft in (classify_text_multi(page_text) or [])
                                           if ft != "1099-DIV"]
                            for ft in extra_forms:
                                merger.add_outline_item(ft, page_num - 1, parent=forms_node)

                        elif form_type == "1099-INT":
                            if has_nonzero_int(page_text):
                                append_and_bookmark(real_entry, forms_node, "1099-INT Description")
                            else:
        # Still append the page, but give it a neutral label
                                append_and_bookmark(real_entry, forms_node, "1099-INT (all zero)")
                                print(f"[NOTE] {os.path.basename(real_entry[0])} page {real_entry[1]+1} "
                                  f"→ 1099-INT detected but all zero; kept page with neutral bookmark", file=sys.stderr)

                        # 2️⃣ Also check for other forms on same page
                            extra_forms = [ft for ft in (classify_text_multi(page_text) or [])
                                           if ft != "1099-INT"]
                            for ft in extra_forms:
                                merger.add_outline_item(ft, page_num - 1, parent=forms_node)

                        else:
        # 3️⃣ Fallback: pure multi-form logic
                            form_matches = classify_text_multi(page_text)

                            title = None
                            extra_forms = []

                            if form_matches:
    # Special rule: drop 1099-INT if all zero
                                if "1099-INT" in form_matches and not has_nonzero_int(page_text):
                                    form_matches = [f for f in form_matches if f != "1099-INT"]

                                if form_matches:
                                    title = form_matches[0]
                                    extra_forms = form_matches[1:]

# Append once, with or without bookmark
                            if title:
                                append_and_bookmark(real_entry, forms_node, title)
                                for ft in extra_forms:
                                    merger.add_outline_item(ft, page_num - 1, parent=forms_node)
                            else:
    # Only zero INT → keep page, no bookmark
                                append_and_bookmark(real_entry, forms_node, "", with_bookmark=False)

                continue
            # 4️⃣ ── Normal single-form handling (W-2, 1099s, etc.) ─────────────
            filtered_grp = [e for e in grp if (e[0], e[1]) not in consolidated_pages]
            if not filtered_grp:
                continue

            #k1 bookmrk

  # done with this form; go to next
            #Normal Forms
            node = merger.add_outline_item(form, page_num, parent=root)
            for j, entry in enumerate(filtered_grp, 1):
                path, idx, _ = entry
                # ✅ Skip if page already processed in K-1 grouping
                if (path, idx) in processed_pages:
                    continue

                # 🚫 Skip if already appended under Consolidated-1099
                if (path, idx) in consolidated_pages:
                    continue

                # build the label
                lbl = form if len(grp) == 1 else f"{form}#{j}"
                if form == 'W-2':
                    emp = w2_titles.get((path, idx))
                    if emp:
                        lbl = emp
                elif form == '1099-INT':
                    payer = int_titles.get((path, idx))
                    if payer:
                        lbl = payer
                elif form == '1099-DIV':                  # <<< new
                    payer = div_titles.get((path, idx))
                    if payer:
                        lbl = payer
                elif form == '1099-SA':
                    payer = sa_titles.get((path, idx))
                    if payer:
                        lbl = payer

                # NEW: strip ", N.A" and stop after this bookmark
                if ", N.A" in lbl:
                    lbl = lbl.replace(", N.A", "")
                print(f"[Bookmark] {os.path.basename(path)} p{idx+1} → Category='Income', Form='{form}', Title='{lbl}'", file=sys.stderr)
                   
                # normal case
                print(f"[Bookmark] {os.path.basename(path)} p{idx+1} → Category='Income', Form='{form}', Title='{lbl}'", file=sys.stderr)
                page_text = extract_text(path, idx)
                owner = detect_ssn_owner(page_text, tp_ssn, sp_ssn)
                if owner:
                    lbl = f"{lbl} – {owner}"
 
                print(f"[SSN Tag] {os.path.basename(path)} p{idx+1} → {owner}", file=sys.stderr)
    
                append_and_bookmark(entry, node, lbl)
            
            if stop_after_na:
                break

    if expenses:
        root = merger.add_outline_item('Expenses', page_num)
        for form, grp in group_by_type(expenses).items():
            if stop_after_na:
                break
            node = merger.add_outline_item(form, page_num, parent=root)
            for j, entry in enumerate(grp, 1):
                path, idx, _ = entry
                lbl = form if len(grp) == 1 else f"{form}#{j}"
                if form == '1098-Mortgage':
                    m = mort_titles.get((path, idx))
                    if m:
                      lbl = m
                elif form == '5498-SA':
                    trustee = sa5498_titles.get((path, idx))
                    if trustee:
                        lbl = trustee
                    else:
                        lbl = extract_5498sa_bookmark(text)

                elif form == '1098-T':
                    trustee = t1098_titles.get((path, idx))
                    if trustee:
                        lbl = trustee
                    else:
                        page_text = extract_text(path, idx)  # ✅ get text for this page
                        lbl = extract_1098t_bookmark(page_text)
                elif form == "Child Care Expenses":
                    page_text = extract_text(path, idx)  # ✅ get the actual text for this page
                    provider = extract_daycare_bookmark(page_text)
                    lbl = provider if provider else "Child Care Provider"
                    append_and_bookmark(entry, node, lbl)
                elif form == '529-Plan':
                    title = t529_titles.get((path, idx))
                    if title:
                        lbl = title

               
                # NEW: strip ", N.A" and stop
                if ", N.A" in lbl:
                    lbl = lbl.replace(", N.A", "")
                print(f"[Bookmark] {os.path.basename(path)} p{idx+1} → Category='Expenses', Form='{form}', Title='{lbl}'", file=sys.stderr)
                   
                # normal case
                print(f"[Bookmark] {os.path.basename(path)} p{idx+1} → Category='Expenses', Form='{form}', Title='{lbl}'", file=sys.stderr)
                # 🆕 Detect SSN owner for this page and label it
                page_text = extract_text(path, idx)
                owner = detect_ssn_owner(page_text, tp_ssn, sp_ssn)
                if owner:
                    lbl = f"{lbl} – {owner}"
 
                print(f"[SSN Tag] {os.path.basename(path)} p{idx+1} → {owner}", file=sys.stderr)
    
                append_and_bookmark(entry, node, lbl)
            if stop_after_na:
                break

# --- Add Others section with Unused and Duplicate pages ---
    if (others and any(e[2] != "Unused" for e in others)) or duplicate_files:
        root = merger.add_outline_item('Others', page_num)

        # Unused pages
        unused_pages = [e for e in others if e[2] == 'Unused']
        if unused_pages:
            node_unused = merger.add_outline_item('Unused', page_num, parent=root)
            for entry in unused_pages:
                append_and_bookmark(entry, node_unused, "", with_bookmark=False)
                print(f"[Bookmark] {os.path.basename(entry[0])} p{entry[1]+1} → Category='Others', Form='Unused'", file=sys.stderr)

    # 🆕 Duplicate pages or duplicate files
        # 🆕 Duplicate pages or duplicate files
        dup_pages = [e for e in others if e[2] == 'Duplicate']
        if dup_pages or duplicate_files:
            node_dupe = merger.add_outline_item('Duplicate', page_num, parent=root)

            # Page-level duplicates (append without bookmarks)
            for entry in dup_pages:
                append_and_bookmark(entry, node_dupe, "", with_bookmark=False)

            # File-level duplicates (append without bookmarks)
            for f in duplicate_files:
                dup_path = os.path.join(abs_input, f)
                try:
                    reader = PdfReader(dup_path)
                    for i in range(len(reader.pages)):
                        append_and_bookmark((dup_path, i, "Duplicate"), node_dupe, "", with_bookmark=False)
                    print(f"[Duplicate] Added file {f} under 'Others → Duplicate'", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️ Failed to append duplicate file {f}: {e}", file=sys.stderr)




            #append_and_bookmark(entry, node, lbl)

    input_count = sum(
    len(PdfReader(os.path.join(input_dir, f)).pages)
    for f in files if f.lower().endswith(".pdf")
    )
    print(f"[SUMMARY] Input pages={input_count}, Output pages={page_num}", file=sys.stderr)

    # Write merged output
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)
    with open(abs_output,'wb') as f:
        merger.write(f)
    merger.close()
    print(f"Merged PDF created at {abs_output}", file=sys.stderr)

    # Cleanup uploads
    # Cleanup uploads
    # Cleanup uploads (originals + converted PDFs)
    to_delete = set(files) | set(os.path.basename(f) for f in converted_files)

    for fname in list(to_delete):
        fpath = os.path.join(input_dir, fname)
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"🧹 Deleted {fname}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ Failed to delete {fname}: {e}", file=sys.stderr)

    # Also remove any leftover images (JPG, PNG, etc.)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            try:
                os.remove(os.path.join(input_dir, fname))
                print(f"🧹 Deleted leftover image {fname}", file=sys.stderr)
            except Exception as e:
                print(f"⚠️ Failed to delete leftover image {fname}: {e}", file=sys.stderr)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Merge PDFs with robust text extraction and TP/SP labeling")
    p.add_argument('input_dir', help="Folder containing PDFs to merge")
    p.add_argument('output_pdf', help="Path for the merged PDF (outside input_dir)")
    p.add_argument('tp_ssn', nargs='?', default='', help="Taxpayer SSN last 4 digits")
    p.add_argument('sp_ssn', nargs='?', default='', help="Spouse SSN last 4 digits")
    args = p.parse_args()
    merge_with_bookmarks(args.input_dir, args.output_pdf, args.tp_ssn, args.sp_ssn)

