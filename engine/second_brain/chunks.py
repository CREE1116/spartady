"""Paragraph-aware chunks with exact source offsets; capture links stay on page anchors."""
import re
from .core import digest


def estimated_tokens(text):
    # Conservative reproducible estimate, explicitly not a provider tokenizer count.
    return len(re.findall(r'[가-힣]|[A-Za-z0-9_]+|[^\s\w]', text))


def chunk_bundle(bundle, limit=450):
    if limit < 50:
        raise ValueError('Chunk token estimate limit must be at least 50')
    anchor = next(s for s in bundle['sources'] if s['kind']=='courseware')
    text = anchor['text']
    # Preserve line/paragraph boundaries. A long formula/line is split only at safe
    # whitespace when possible; every emitted range remains an exact source slice.
    boundaries = [m.end() for m in re.finditer(r'\n', text)] + [len(text)]
    ranges=[];start=0;end=0
    for boundary in sorted(set(boundaries)):
        if estimated_tokens(text[start:boundary]) <= limit:
            end=boundary;continue
        if end>start:
            ranges.append((start,end));start=end
        while estimated_tokens(text[start:boundary])>limit:
            lo,hi=start+1,boundary
            while lo<hi:
                mid=(lo+hi+1)//2
                if estimated_tokens(text[start:mid])<=limit:lo=mid
                else:hi=mid-1
            cut=lo
            space=max(text.rfind(' ',start,cut),text.rfind('\t',start,cut))
            if space>start+(cut-start)//2:cut=space+1
            ranges.append((start,cut));start=cut
        end=boundary
    if end>start:ranges.append((start,end))
    result=[]
    for index,(start,end) in enumerate(ranges):
        if not text[start:end].strip():continue
        id=f"{bundle['id']}:C{index+1:03d}"
        source=dict(anchor,id=id,text=text[start:end],parent_source_id=anchor['id'],
                    char_start=start,char_end=end)
        # Context supplements belong to the page, not inferred to each concept.
        context=[dict(s,scope='page_context') for s in bundle['sources'] if s['kind']!='courseware']
        result.append(dict(bundle,id=id,anchor_id=bundle['id'],sources=[source]+context,
            char_start=start,char_end=end,estimated_tokens=estimated_tokens(source['text']),
            chunking={'policy':'paragraph_source_offsets_v1','limit':limit},
            input_hash=digest({'source':source,'context':context,'title':bundle['title']})))
    return result
