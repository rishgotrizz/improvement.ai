import os
from playwright.sync_api import sync_playwright

DOCS_DIR = "/Users/macbook/Documents/pbl/docs"
HTML_PATH = os.path.join(DOCS_DIR, "EduPredict_AI_Review_II_Documentation.html")
PDF_PATH_DOCS = os.path.join(DOCS_DIR, "EduPredict_AI_Review_II_Documentation.pdf")
PDF_PATH_ROOT = "/Users/macbook/Documents/pbl/EduPredict_AI_Review_II_Documentation.pdf"

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox", "--disable-setuid-sandbox"])
    page = browser.new_page()
    page.goto(f"file://{HTML_PATH}", wait_until="networkidle")
    
    # Generate PDF in docs/
    page.pdf(
        path=PDF_PATH_DOCS,
        format="A4",
        print_background=True,
        margin={
            "top": "14mm",
            "bottom": "16mm",
            "left": "12mm",
            "right": "12mm"
        }
    )
    
    # Generate PDF in root/
    page.pdf(
        path=PDF_PATH_ROOT,
        format="A4",
        print_background=True,
        margin={
            "top": "14mm",
            "bottom": "16mm",
            "left": "12mm",
            "right": "12mm"
        }
    )
    browser.close()

print(f"✅ Generated PDF at {PDF_PATH_DOCS} and {PDF_PATH_ROOT}")
