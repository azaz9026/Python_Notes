from fpdf import FPDF
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from bs4 import BeautifulSoup

# Read Python code from file
with open('Statistics .py', 'r') as file:
    code = file.read()

# Highlight the code
formatter = HtmlFormatter()
highlighted_code = highlight(code, PythonLexer(), formatter)

# Convert HTML to plain text
soup = BeautifulSoup(highlighted_code, 'html.parser')
plain_text = soup.get_text()

# Create PDF
pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, plain_text)

# Save the PDF
pdf.output("your_code.pdf")
