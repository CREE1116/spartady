from __future__ import annotations
import glob,hashlib,io,json,os,re
from collections import Counter,defaultdict
from dataclasses import dataclass,field
from pathlib import Path
from typing import Dict,List,Optional,Set,Tuple
import numpy as np
import pymupdf
from PIL import Image

@dataclass
class SlideChunk:
    chunk_id: str
    doc_name: str
    lecture_order: int
    page_num: int
    chapter_title: str
    slide_title: str
    base_header: str
    section_num: str
    sub_headers: List[str]
    body_lines: List[str]
    raw_text: str
    num_images: int = 0
    image_paths: List[str] = field(default_factory=list)
    pil_images: List[Image.Image] = field(default_factory=list, repr=False)
    is_image_heavy: bool = False
    is_admin_or_toc: bool = False
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    text_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    mm_embedding: Optional[np.ndarray] = field(default=None, repr=False)



def _extract_lecture_order(filename: str) -> float:
    m = re.search(r"강의(\d+(?:\.\d+)?)", filename)
    if m:
        return float(m.group(1))
    return 99.0


def _clean_slide_header(header: str) -> Tuple[str, str, str]:
    """
    Automatically splits a raw slide header into:
      - `section_num`: e.g. '1.1', '2.1', '3-1', '4', '[별첨1]'
      - `base_header`: header with trailing '(1/2)', '(2/4)', '(1)', '(2)' removed
      - `core_topic`: core noun phrase after removing section numbers and prefixes
    """
    h = re.sub(r"\s+", " ", header).strip()
    # Remove trailing pagination like (1/2), (2/4), (1), (2)
    base = re.sub(r"\s*\(\d+(?:/\d+)?\)\s*$", "", h).strip()
    base = re.sub(r"\*+$", "", base).strip()

    sec_num = ""
    core = base
    m_sec = re.match(r"^(\[별첨\d*\]|[IVX]+\.|[0-9]+(?:[\.\-][0-9]+)*\.?)\s*(.+)$", base)
    if m_sec:
        sec_num = m_sec.group(1).rstrip(".")
        core = m_sec.group(2).strip()

    # If core has a prefix like "정보보안의역사/ 1950년대이전" or "대칭암호화방식– DES 알고리즘"
    # preserve both in base_header, and extract concise core_topic
    core = re.sub(r"\s*\((?:개요|작성지침|전반)\)\s*$", "", core).strip()
    return sec_num, base, core


def extract_multimodal_chunks_from_pdfs(
    pdf_dir: str,
    save_images: bool = True,
) -> List[SlideChunk]:
    """
    Programmatically parses all PDFs in `pdf_dir` into `SlideChunk` objects.
    Extracts typography hierarchy and non-template visual figures (`PIL.Image`).
    """
    import unicodedata

    pdf_files = sorted(
        glob.glob(os.path.join(pdf_dir, "*.pdf")),
        key=lambda p: _extract_lecture_order(unicodedata.normalize("NFC", os.path.basename(p))),
    )
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    if save_images:
        EXTRACTED_IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: List[SlideChunk] = []

    for pdf_path in pdf_files:
        doc_name = unicodedata.normalize("NFC", os.path.basename(pdf_path))
        lec_order = int(round(_extract_lecture_order(doc_name) * 10))
        doc = pymupdf.open(pdf_path)

        # Count image xrefs across the document to filter out repeated slide-master logos
        xref_counts = Counter(
            img[0] for page in doc for img in page.get_images(full=True)
        )

        current_chapter = doc_name.replace(".pdf", "")
        doc_chunks: List[SlideChunk] = []

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            chunk_id = f"L{lec_order:03d}_P{page_num:02d}"

            blocks = page.get_text("dict").get("blocks", [])
            header_spans: List[str] = []
            sub_spans: List[str] = []
            body_spans: List[str] = []

            for b in blocks:
                for line in b.get("lines", []):
                    txt = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    txt = unicodedata.normalize("NFC", txt)
                    if not txt:
                        continue
                    # Ignore pure page number footers
                    if re.fullmatch(r"\d{1,3}", txt):
                        continue
                    max_sz = max((s.get("size", 0.0) for s in line.get("spans", [])), default=0.0)
                    if max_sz >= 32.0:
                        # Chapter cover slide title
                        current_chapter = txt
                        header_spans.append(txt)
                    elif max_sz >= 22.0:
                        header_spans.append(txt)
                    elif max_sz >= 15.0:
                        sub_spans.append(txt)
                    else:
                        body_spans.append(txt)

            slide_title = " / ".join(header_spans).strip()
            if not slide_title and sub_spans:
                # Handle slides where the title was formatted at 16-20pt (e.g. 강의5 P11: 4.2 인증및세션관리취약점)
                slide_title = sub_spans[0]
                sub_spans = sub_spans[1:]

            # Clean bullets
            clean_subs = [re.sub(r"^[\uf0a7\uf076•▪\-]+\s*", "", s).strip() for s in sub_spans if s.strip()]
            clean_body = [re.sub(r"^[\uf0a7\uf076•▪\-]+\s*", "", s).strip() for s in body_spans if s.strip()]

            sec_num, base_header, _ = _clean_slide_header(slide_title)

            # Extract unique non-template content images from the slide
            pil_imgs: List[Image.Image] = []
            img_paths: List[str] = []
            for img_info in page.get_images(full=True):
                xref, width, height = img_info[0], img_info[2], img_info[3]
                if xref_counts[xref] <= 2 and width >= 120 and height >= 80:
                    try:
                        extracted = doc.extract_image(xref)
                        raw_bytes = extracted["image"]
                        im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                        im.thumbnail((512, 512))
                        pil_imgs.append(im)
                        if save_images:
                            out_p = EXTRACTED_IMG_DIR / f"{chunk_id}_img{len(pil_imgs)}.png"
                            if not out_p.exists():
                                im.save(out_p)
                            img_paths.append(str(out_p))
                    except Exception:
                        pass
                if len(pil_imgs) >= 2:
                    break

            # Fallback for vector-diagram slides or low-text slides with drawings
            drawings_count = len(page.get_drawings())
            if not pil_imgs and (drawings_count > 8 or (len(clean_body) <= 2 and len(clean_subs) <= 2)):
                try:
                    pix = page.get_pixmap(dpi=110)
                    im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    im.thumbnail((512, 512))
                    pil_imgs.append(im)
                    if save_images:
                        out_p = EXTRACTED_IMG_DIR / f"{chunk_id}_render.png"
                        if not out_p.exists():
                            im.save(out_p)
                        img_paths.append(str(out_p))
                except Exception:
                    pass

            # Determine if administrative / cover / TOC slide
            admin_keywords = (
                "수업오리엔테이션", "강의목차", "강의세부계획", "강의시간운용",
                "강의교재", "평가", "강의방향", "수업관련", "목차와학습목표",
                "수고하셨습니다", "예제",
            )
            no_space_title = re.sub(r"\s+", "", slide_title)
            is_admin = (
                lec_order == 10  # 강의1 오리엔테이션
                or any(k in no_space_title for k in admin_keywords)
                or (len(clean_subs) <= 2 and len(clean_body) == 0 and any(s.startswith("Ch ") for s in header_spans + clean_subs))
            )

            is_img_heavy = (len(clean_body) <= 3 and len(pil_imgs) > 0 and not is_admin)
            raw_text = "\n".join([slide_title] + clean_subs + clean_body)

            chunk = SlideChunk(
                chunk_id=chunk_id,
                doc_name=doc_name,
                lecture_order=lec_order,
                page_num=page_num,
                chapter_title=current_chapter,
                slide_title=slide_title,
                base_header=base_header,
                section_num=sec_num,
                sub_headers=clean_subs,
                body_lines=clean_body,
                raw_text=raw_text,
                num_images=len(pil_imgs),
                image_paths=img_paths,
                pil_images=pil_imgs,
                is_image_heavy=is_img_heavy,
                is_admin_or_toc=is_admin,
            )
            doc_chunks.append(chunk)

        for i, ch in enumerate(doc_chunks):
            if i > 0:
                ch.prev_chunk_id = doc_chunks[i - 1].chunk_id
            if i < len(doc_chunks) - 1:
                ch.next_chunk_id = doc_chunks[i + 1].chunk_id

        all_chunks.extend(doc_chunks)

    return all_chunks


# =====================================================================
