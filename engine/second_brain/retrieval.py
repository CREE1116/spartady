"""Local lexical retrieval plus typed graph expansion with page/source citations."""
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from .knowledge import key


def walk(graph,concept_id,direction='both',depth=2,kinds=None):
    ids={n['id'] for n in graph['concepts']}
    if concept_id not in ids:raise ValueError('Unknown concept ID')
    if direction not in {'both','prerequisites','dependents'} or not 0<=depth<=10:
        raise ValueError('Invalid graph traversal request')
    allowed=set(kinds or ['REQUIRES','SUPPORTS','RELATED','PART_OF','EXAMPLE_OF'])
    visited={concept_id};frontier={concept_id};selected={}
    for _ in range(depth):
        next_ids=set()
        for e in graph['relations']:
            if e['kind'] not in allowed or e['status']=='REJECTED':continue
            a,b=e['source_id'],e['target_id']
            if direction in {'both','dependents'} and a in frontier:
                selected[e['id']]=e;next_ids.add(b)
            if direction in {'both','prerequisites'} and b in frontier:
                selected[e['id']]=e;next_ids.add(a)
        frontier=next_ids-visited;visited.update(frontier)
        if not frontier:break
    return {'concepts':[n for n in graph['concepts'] if n['id'] in visited],
            'relations':list(selected.values()),'status':graph['status']}


def retrieve(graph,query,limit=5,max_chars=12000,concept_id=None,unit_id=None):
    if not query.strip() or not 1<=limit<=50 or max_chars<200:
        raise ValueError('Query, positive retrieval limit and context budget required')
    nodes={n['id']:n for n in graph['concepts']}
    by_chunk=defaultdict(list)
    for o in graph['occurrences']:by_chunk[o['bundle_id']].append(o)
    chunks=graph['chunks']
    if unit_id and unit_id not in {c['id'] for c in chunks}:raise ValueError('Unknown learning unit')
    if not chunks:return {'query':query,'hits':[],'sources':[],'relations':[],'context_text':''}
    texts=[]
    for chunk in chunks:
        names=' '.join(nodes[o['concept_id']]['name']+' '+' '.join(nodes[o['concept_id']]['aliases'])
                       for o in by_chunk[chunk['id']])
        texts.append(chunk['title']+' '+names+' '+' '.join(s['text'] for s in chunk['sources']))
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,4),lowercase=True)
    matrix=vectorizer.fit_transform(texts)
    scores=(matrix@vectorizer.transform([query]).T).toarray().ravel()
    matched={id for id,n in nodes.items() if any(len(t)>=2 and key(t) in key(query)
                  for t in [n['name']]+n['aliases'])}
    if concept_id:
        if concept_id not in nodes:raise ValueError('Unknown concept ID')
        matched.add(concept_id)
    expanded=set(matched);relations={}
    for id in sorted(matched):
        neighborhood=walk(graph,id,depth=1)
        expanded.update(n['id'] for n in neighborhood['concepts'])
        relations.update((e['id'],e) for e in neighborhood['relations'])
    ranking=[]
    for index,chunk in enumerate(chunks):
        local={o['concept_id'] for o in by_chunk[chunk['id']]}
        direct=bool(local&matched);neighbor=bool(local&expanded)
        score=float(scores[index])+0.4*direct+0.1*(neighbor and not direct)+2.0*(chunk['id']==unit_id)
        if score>0:ranking.append((score,chunk['id'],chunk))
    ranking.sort(key=lambda r:(-r[0],r[1]))
    sources={};hits=[];blocks=[];remaining=max_chars
    def excerpt(text,budget):
        if len(text)<=budget:return text,0,len(text),False
        terms=[t for t in re.findall(r'[가-힣A-Za-z0-9]+',query) if len(t)>=2]
        positions=[text.find(t) for t in terms if t in text]
        start=max(0,(min(positions) if positions else 0)-budget//3)
        end=min(len(text),start+budget)
        return text[start:end],start,end,True
    for score,id,chunk in ranking[:limit]:
        selected=[]
        for source in chunk['sources']:
            if source['id'] in sources:
                selected.append(source['id']);continue
            header=f"[{source['id']}] {source.get('document',chunk['document'])} p.{source.get('page',chunk['page'])} {source['kind']}\n"
            budget=min(3000,remaining-len(header)-2)
            if budget<80:break
            text,start,end,truncated=excerpt(source['text'],budget)
            record=dict(source,text=text,excerpt_start=start,excerpt_end=end,truncated=truncated)
            sources[source['id']]=record;selected.append(source['id'])
            block=header+text+'\n';blocks.append(block);remaining-=len(block)
        if selected:
            hits.append({'chunk_id':id,'anchor_id':chunk['anchor_id'],'document':chunk['document'],
                'page':chunk['page'],'title':chunk['title'],'score':score,
                'source_ids':selected,'concept_ids':sorted({o['concept_id'] for o in by_chunk[id]})})
        if remaining<100:break
    available=set(sources)
    # Only return edges whose actual evidence has been included; graph expansion
    # should not smuggle an unsupported claim into the tutor's bounded context.
    grounded=[e for e in relations.values() if all(
        ev['source_id'] in available and ev['quote'] in sources[ev['source_id']]['text']
        for ev in e['evidence'])]
    return {'query':query,'hits':hits,'concepts':[nodes[id] for id in sorted(expanded)],
        'sources':list(sources.values()),'relations':grounded,'context_text':'\n'.join(blocks),
        'retrieval':'character_tfidf_plus_typed_graph','context_budget_chars':max_chars,
        'graph_status':graph['status']}
