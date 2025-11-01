#!/usr/bin/env python3
"""Convert TECHNICAL_REPORT.md to PDF"""

import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

# Read the markdown file
with open('TECHNICAL_REPORT.md', 'r') as f:
    md_content = f.read()

# Convert markdown to HTML
html_content = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code', 'codehilite']
)

# Add HTML wrapper with styling
html_full = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page {{
            size: letter;
            margin: 1in;
        }}
        body {{
            font-family: 'Georgia', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            font-size: 24pt;
            margin-top: 24pt;
            margin-bottom: 12pt;
            page-break-after: avoid;
            color: #1a1a1a;
        }}
        h2 {{
            font-size: 18pt;
            margin-top: 18pt;
            margin-bottom: 10pt;
            page-break-after: avoid;
            color: #1a1a1a;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 4pt;
        }}
        h3 {{
            font-size: 14pt;
            margin-top: 14pt;
            margin-bottom: 8pt;
            page-break-after: avoid;
            color: #1a1a1a;
        }}
        h4 {{
            font-size: 12pt;
            margin-top: 12pt;
            margin-bottom: 6pt;
            page-break-after: avoid;
            color: #333;
        }}
        p {{
            margin-bottom: 10pt;
            text-align: justify;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10pt;
            border-left: 3px solid #ccc;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            line-height: 1.4;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 12pt 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8pt;
            text-align: left;
        }}
        th {{
            background-color: #f5f5f5;
            font-weight: bold;
        }}
        blockquote {{
            border-left: 4px solid #ddd;
            padding-left: 12pt;
            margin-left: 0;
            font-style: italic;
            color: #666;
        }}
        ul, ol {{
            margin-bottom: 10pt;
        }}
        li {{
            margin-bottom: 4pt;
        }}
        hr {{
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 24pt 0;
        }}
        strong {{
            font-weight: bold;
            color: #1a1a1a;
        }}
        em {{
            font-style: italic;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Convert HTML to PDF
font_config = FontConfiguration()
HTML(string=html_full).write_pdf(
    'TECHNICAL_REPORT.pdf',
    font_config=font_config
)

print("✓ PDF created: TECHNICAL_REPORT.pdf")
