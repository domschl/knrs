from __future__ import annotations

import zipfile
from pathlib import Path


def generate_minimal_pdf(file_path: Path, text: str) -> None:
    """Generate a minimal valid PDF file with custom text."""
    # Escape parentheses in PDF text
    escaped_text = text.replace("(", "\\(").replace(")", "\\)")
    stream_content = f"BT\n/F1 12 Tf\n72 712 Td\n({escaped_text}) Tj\nET\n"
    stream_len = len(stream_content)

    pdf_header = b"%PDF-1.4\n"
    
    obj1 = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    obj4 = f"4 0 obj\n<< /Length {stream_len} >>\nstream\n{stream_content}endstream\nendobj\n"
    obj5 = "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    body = b""
    offsets = []
    for obj_str in [obj1, obj2, obj3, obj4, obj5]:
        offsets.append(len(pdf_header) + len(body))
        body += obj_str.encode("utf-8")
        
    xref_pos = len(pdf_header) + len(body)
    
    xref = "xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n"
        
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    
    with open(file_path, "wb") as f:
        f.write(pdf_header)
        f.write(body)
        f.write(xref.encode("utf-8"))
        f.write(trailer.encode("utf-8"))


def generate_minimal_epub(file_path: Path, title: str, text: str) -> None:
    """Generate a minimal valid EPUB file using zipfile."""
    container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    content_opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{title}</dc:title>
    <dc:creator>Benchmark Generator</dc:creator>
    <dc:identifier id="bookid">urn:uuid:12345678-1234-1234-1234-123456789012</dc:identifier>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="chapter1"/>
  </spine>
</package>"""

    toc_ncx = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE ncx PUBLIC "-//NISO//DTD NCX 2005-1//EN" "http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content="urn:uuid:12345678-1234-1234-1234-123456789012"/>
    <meta name="dtb:depth" content="1"/>
    <meta name="dtb:totalPageCount" content="0"/>
    <meta name="dtb:maxPageNumber" content="0"/>
  </head>
  <docTitle>
    <text>{title}</text>
  </docTitle>
  <navMap>
    <navPoint id="navPoint-1" playOrder="1">
      <navLabel>
        <text>Chapter 1</text>
      </navLabel>
      <content src="chapter1.xhtml"/>
    </navPoint>
  </navMap>
</ncx>"""

    chapter_xhtml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <title>Chapter 1</title>
</head>
<body>
  <h1>{title}</h1>
  <p>{text}</p>
</body>
</html>"""

    # EPUB requires 'mimetype' to be the first file in the ZIP, uncompressed.
    with zipfile.ZipFile(file_path, "w") as z:
        # Write mimetype uncompressed
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        # Write XML container and content
        z.writestr("META-INF/container.xml", container_xml, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/content.opf", content_opf, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/toc.ncx", toc_ncx, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/chapter1.xhtml", chapter_xhtml, compress_type=zipfile.ZIP_DEFLATED)


def generate_minimal_md(file_path: Path, title: str, text: str) -> None:
    """Generate a minimal markdown file with frontmatter."""
    content = f"""---
title: {title}
uuid: 12345678-1234-1234-1234-123456789012
authors: ["Benchmark Generator"]
creation_date: "2026-05-23"
context: "AINotes/Books/Benchmark"
---

# {title}

{text}
"""
    file_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    # Test generation
    test_dir = Path("test_benchmark_docs")
    test_dir.mkdir(exist_ok=True)
    
    generate_minimal_pdf(test_dir / "test.pdf", "Minimal test PDF document for benchmark.")
    generate_minimal_epub(test_dir / "test.epub", "Test EPUB", "Minimal test EPUB chapter content.")
    generate_minimal_md(test_dir / "test.md", "Test Markdown", "Minimal test Markdown content.")
    print("Test documents generated in:", test_dir.resolve())
