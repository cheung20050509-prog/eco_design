import pymupdf4llm
import pathlib

pdf_path = "/root/eco_design/《金融数据分析与智能量化交易应用设计课程》论文撰写说明.pdf"
md_path = "/root/eco_design/《金融数据分析与智能量化交易应用设计课程》论文撰写说明.md"

print(f"Converting {pdf_path} to Markdown...")
md_text = pymupdf4llm.to_markdown(pdf_path)

pathlib.Path(md_path).write_bytes(md_text.encode('utf-8'))
print(f"Successfully saved to {md_path}")
