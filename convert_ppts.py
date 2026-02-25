import os
import subprocess
import pymupdf4llm
import pathlib

ppt_dir = "/root/eco_design/ecofinal/ecofinal/金融数据分析与智能量化交易应用PPT_2025"
output_dir = "/root/eco_design/ecofinal/ecofinal/PPT_Markdown"

os.makedirs(output_dir, exist_ok=True)

ppt_files = [f for f in os.listdir(ppt_dir) if f.endswith('.pptx')]

for ppt_file in ppt_files:
    ppt_path = os.path.join(ppt_dir, ppt_file)
    pdf_path = os.path.join(output_dir, ppt_file.replace('.pptx', '.pdf'))
    md_path = os.path.join(output_dir, ppt_file.replace('.pptx', '.md'))
    
    print(f"Processing {ppt_file}...")
    
    # 1. Convert PPTX to PDF using LibreOffice
    print(f"  Converting to PDF...")
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf", 
        "--outdir", output_dir, ppt_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 2. Convert PDF to Markdown using pymupdf4llm
    if os.path.exists(pdf_path):
        print(f"  Converting PDF to Markdown...")
        try:
            md_text = pymupdf4llm.to_markdown(pdf_path)
            pathlib.Path(md_path).write_bytes(md_text.encode('utf-8'))
            print(f"  Successfully saved to {md_path}")
            
            # Clean up the intermediate PDF file
            os.remove(pdf_path)
        except Exception as e:
            print(f"  Error converting PDF to Markdown: {e}")
    else:
        print(f"  Failed to generate PDF for {ppt_file}")

print("\nAll conversions completed!")
