from pypdf import PdfReader
from docx import Document

def read_resume_from_upload(file):
    name = file.filename.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file.file)
        return " ".join([p.extract_text() or "" for p in reader.pages])

    elif name.endswith(".docx"):
        doc = Document(file.file)
        return "\n".join(p.text for p in doc.paragraphs)

    else:
        return file.file.read().decode("utf-8", errors="ignore")