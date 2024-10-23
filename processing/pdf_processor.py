# processing/pdf_processor.py

import PyPDF2

def extract_text_from_pdf(pdf_path, max_pages=None):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            num_pages = len(reader.pages)
            pages_to_read = num_pages if not max_pages else min(num_pages, max_pages)
            for page_num in range(pages_to_read):
                page = reader.pages[page_num]
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return "Error processing the PDF file."