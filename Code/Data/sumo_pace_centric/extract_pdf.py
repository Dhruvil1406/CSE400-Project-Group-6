import subprocess
import sys

def install_and_extract():
    try:
        import fitz
    except ImportError:
        print("Installing PyMuPDF to read the PDF...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF"])
        import fitz

    print("Extracting text from Research Paper.pdf...")
    doc = fitz.open('Research Paper.pdf')
    text = ""
    for page in doc:
        text += page.get_text()
    
    with open('extracted_paper.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Extraction complete. Text saved to extracted_paper.txt.")

if __name__ == "__main__":
    install_and_extract()
