from PyPDF2 import PdfReader


def read_pdf(file):
    pdf = PdfReader(file)
    return ' '.join([pdf.pages[i].extract_text() for i in range(len(pdf.pages))])

def read_docx(file):
    doc = Document(file)
    return ' '.join([p.text for p in doc.paragraphs])


