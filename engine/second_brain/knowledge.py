"""Persistent concept knowledge, source-proven identity links, and asymmetric chunk evidence."""
import math
import re
from collections import Counter, defaultdict
from .core import canonical, cid, digest


def key(name):
    # Korean spacing variation is lexical normalization, never an embedding merge.
    name=canonical(name)
    return re.sub(r'(?<=[가-힣])\s+(?=[가-힣])','',name)


def heading_concepts(bundle):
    source=next(s for s in bundle['sources'] if s['kind']=='courseware')
    raw=bundle.get('title') or source['text'].splitlines()[0]
    names=[]
    for part in re.split(r'[/·:：]|\s+-\s+',raw):
        name=re.sub(r'^\s*[\d.\s]+','',part).strip()
        name=re.sub(r'\([^)]*\)','',name).strip()
        if not 2<=len(name)<=60 or key(name) in {key(n) for n in names}:continue
        # Require a literal source span, not invented metadata terminology.
        if name not in source['text']:continue
        names.append(name)
    return [{'name':name,'aliases':[],'role':'teaches','difficulty':1,
             'objective':name+'의 설명 확인','evidence':[{'source_id':source['id'],'quote':name}],
             'origin':'HEADING_BASELINE','role_strength':0.4} for name in names[:4]]


def assemble(store):
    # Snapshot raw chunks, outputs and the derived graph in one short transaction.
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        return _assemble(store)


def _assemble(store):
    chunks=store.chunks()
    observations=[];assertions=[];errors=[];calls=[];coverage=[]
    sources={}
    for bundle in chunks:
        for source in bundle['sources']:
            if source['id'] in sources and sources[source['id']]!=source:
                raise ValueError('Conflicting source identities across chunks')
            sources[source['id']]=source
        extraction=store.extraction(bundle['id'])
        if extraction:
            g=extraction['graph'];names={n['id']:n['name'] for n in g['concepts']}
            for o in g['occurrences']:
                observations.append(dict(o,name=names[o['concept_id']],
                    origin='MODEL_EXTRACTION',role_strength=1.0))
            for e in g['relations']:
                assertions.append(dict(e,source_name=names[e['source_id']],target_name=names[e['target_id']],
                    origin='MODEL_EXTRACTION',model=g['manifest']['model']))
            errors.extend(g['errors']);calls.extend(g['calls'])
            if not g['concepts']:
                for c in heading_concepts(bundle):
                    observations.append(dict(c,bundle_id=bundle['id']))
            coverage.append({'chunk_id':bundle['id'],'status':'MODEL_EXTRACTED',
                             'backend_hash':extraction['backend_hash'],
                             'valid_concepts':len(g['concepts']),'heading_fallback':not bool(g['concepts'])})
        else:
            for c in heading_concepts(bundle):
                observations.append(dict(c,bundle_id=bundle['id']))
            coverage.append({'chunk_id':bundle['id'],'status':'HEADING_BASELINE'})
    # Resolve only exact lexical forms and bilingual equivalences printed by the source.
    parent={key(o['name']):key(o['name']) for o in observations}
    display={key(o['name']):o['name'] for o in observations}
    def root(x):
        while parent[x]!=x:x=parent[x]
        return x
    merges=[]
    for source in sources.values():
        for match in re.finditer(r'([가-힣A-Za-z][가-힣A-Za-z0-9 _-]{1,50})\(([A-Za-z][A-Za-z0-9 _/-]{1,70})\)',source['text']):
            a,b=key(match[1].strip()),key(match[2].strip())
            if a not in parent or b not in parent:continue
            ra,rb=root(a),root(b)
            if ra==rb:continue
            representative,other=sorted((ra,rb),key=lambda s:(not bool(re.search('[가-힣]',s)),s))
            parent[other]=representative
            merges.append({'forms':[display[a],display[b]],'rule':'SOURCE_PRINTED_EQUIVALENCE',
                'evidence':{'source_id':source['id'],'quote':match[0]}})
    nodes={};occurrences=[];seen={}
    for o in observations:
        identity=root(key(o['name']));id=cid(identity)
        node=nodes.setdefault(id,{'id':id,'name':display[identity],'aliases':[],
            'candidate_aliases':[],'occurrence_ids':[],'status':'SOURCE_GROUNDED_CANDIDATE'})
        # Unproven model aliases are retained for investigation, not indexed as fact.
        aliases=[o['name']]
        for alias in o.get('aliases',[]):
            printed=any(key(o['name'])+'('+key(alias)+')' in key(sources[ev['source_id']]['text']) or
                        key(alias)+'('+key(o['name'])+')' in key(sources[ev['source_id']]['text'])
                        for ev in o['evidence'])
            if printed:aliases.append(alias)
            elif alias not in node['candidate_aliases']:node['candidate_aliases'].append(alias)
        node['aliases']=sorted(set(node['aliases']+aliases)-{node['name']})
        oid=f"{o['bundle_id']}:{id}:{o['role']}"
        if oid in seen:
            previous=seen[oid]
            for ev in o['evidence']:
                if ev not in previous['evidence']:previous['evidence'].append(ev)
            previous['objectives']=sorted(set(previous['objectives']+[o['objective']]))
            previous['difficulty']=max(previous['difficulty'],o['difficulty'])
            previous['role_strength']=max(previous['role_strength'],o['role_strength'])
            continue
        record=dict(o,id=oid,concept_id=id,objectives=[o['objective']])
        seen[oid]=record
        node['occurrence_ids'].append(oid);occurrences.append(record)
    # Searchable mention edges cover vocabulary across pages without asserting that
    # a shared term implies a prerequisite. Only explicit use cues create weak uses.
    occupied={(o['bundle_id'],o['concept_id']) for o in occurrences}
    vocabulary={id:[n['name']]+n['aliases'] for id,n in nodes.items()}
    for bundle in chunks:
        source=next(s for s in bundle['sources'] if s['kind']=='courseware')
        for id,terms in vocabulary.items():
            if (bundle['id'],id) in occupied:continue
            literal=next((term for term in sorted(terms,key=len,reverse=True)
                if len(term)>=2 and term in source['text']),None)
            if literal is None:continue
            # English substrings must be complete terms (e.g. MAC must not match MACsec).
            if re.fullmatch(r'[A-Za-z0-9 _-]+',literal) and not re.search(
                r'(?<![A-Za-z0-9])'+re.escape(literal)+r'(?![A-Za-z0-9])',source['text']):continue
            pos=source['text'].find(literal)
            window=source['text'][max(0,pos-30):pos+len(literal)+30]
            uses=bool(re.search(r'이용|사용|활용|기반|앞서|가정|필요',window))
            role='uses' if uses else 'mentions'
            oid=f"{bundle['id']}:{id}:{role}"
            occurrences.append({'id':oid,'concept_id':id,'name':nodes[id]['name'],
                'bundle_id':bundle['id'],'role':role,'role_strength':0.25 if uses else 0,
                'difficulty':1,'objective':'','aliases':[],
                'evidence':[{'source_id':source['id'],'quote':literal}],
                'origin':'LEXICAL_USE_CUE' if uses else 'LEXICAL_MENTION'})
            nodes[id]['occurrence_ids'].append(oid)
    relations={}
    decisions=store.decisions()
    for e in assertions:
        a,b=cid(root(key(e['source_name']))),cid(root(key(e['target_name'])))
        if a==b:
            errors.append({'type':'relation','reason':'Self edge after source-proven identity resolution'});continue
        id='E_'+digest({'source':a,'target':b,'kind':e['kind']})[:20]
        edge=relations.setdefault(id,{'id':id,'source_id':a,'target_id':b,'kind':e['kind'],
            'status':'CANDIDATE','evidence':[],'reasons':[],'bundle_ids':[],'models':[],
            'weight':None,'weight_semantics':'not_calibrated'})
        for field in ['evidence','reasons','bundle_ids']:
            for value in e.get(field,[]):
                if value not in edge[field]:edge[field].append(value)
        if e['model'] not in edge['models']:edge['models'].append(e['model'])
    for id,edge in relations.items():
        cited={ev['source_id'] for ev in edge['evidence']}
        edge['evidence_hash']=digest({'evidence':edge['evidence'],
            'sources':{s:digest(sources[s]) for s in sorted(cited)}})
        if id in decisions and decisions[id]['evidence_hash']==edge['evidence_hash']:
            edge.update(decisions[id])
    # O -> P is a directional evidence score, NOT a calibrated probability.
    outcome=defaultdict(dict);prereq=defaultdict(dict);evidence=defaultdict(list)
    for o in occurrences:
        role=o['role'];unit=o['bundle_id'];concept=o['concept_id']
        if role=='teaches':outcome[unit][concept]=max(outcome[unit].get(concept,0),o['role_strength'])
        elif role=='uses':prereq[unit][concept]=max(prereq[unit].get(concept,0),o['role_strength'])
        evidence[(unit,concept)].extend(o['evidence'])
    df=Counter()
    for unit in chunks:
        df.update({o['concept_id'] for o in occurrences if o['bundle_id']==unit['id']})
    idf={id:1+math.log((1+len(chunks))/(1+df[id])) for id in nodes}
    def score(a,b):
        left,right=outcome[a],prereq[b]
        shared=left.keys()&right.keys()
        numerator=sum(idf[z]*left[z]*right[z] for z in shared)
        norm=math.sqrt(sum(idf[z]*v*v for z,v in left.items()))*math.sqrt(sum(idf[z]*v*v for z,v in right.items()))
        return numerator/(norm+1e-12),shared
    chunk_edges=[]
    for a in sorted(outcome):
        for b in sorted(prereq):
            if a==b:continue
            forward,shared=score(a,b)
            if not shared:continue
            reverse,_=score(b,a)
            reliability=min(max(outcome[a][z] for z in shared),max(prereq[b][z] for z in shared))
            chunk_edges.append({'id':'CE_'+digest([a,b,sorted(shared)])[:20],
                'source_id':a,'target_id':b,'kind':'PREREQUISITE_EVIDENCE','status':'CANDIDATE',
                'forward_score':forward,'reverse_score':reverse,'direction_margin':forward-reverse,
                'weight':forward*reliability,'role_reliability':reliability,
                'weight_semantics':'role_overlap_evidence_not_probability',
                'concept_ids':sorted(shared),'evidence_terms':[nodes[z]['name'] for z in sorted(shared)],
                'source_evidence':[e for z in sorted(shared) for e in evidence[(a,z)]],
                'target_evidence':[e for z in sorted(shared) for e in evidence[(b,z)]]})
    graph={'version':'second-brain-v1','status':'UNVALIDATED_RESEARCH_GRAPH',
        'concepts':list(nodes.values()),'occurrences':occurrences,'relations':list(relations.values()),
        'chunk_edges':chunk_edges,'chunks':chunks,'sources':list(sources.values()),
        'identity_evidence':merges,'errors':errors,'coverage':coverage,
        'unresolved_alignment':store.get_metadata('unresolved_alignment',[])}
    grouped=defaultdict(list)
    for o in occurrences:grouped[o['concept_id']].append(o)
    for id,node in nodes.items():
        teaching=[o for o in grouped[id] if o['role']=='teaches']
        node['learning_objectives']=sorted({goal for o in teaching
            for goal in o.get('objectives',[o.get('objective','')]) if goal})
        node['teaching_chunk_ids']=sorted({o['bundle_id'] for o in teaching})
        node['difficulty_estimate']=max((o['difficulty'] for o in teaching),default=None)
        node['difficulty_semantics']='uncalibrated relative estimate, not measured time'
    graph['manifest']={'anchors':len(store.anchors()),'chunks':len(chunks),'concepts':len(nodes),
        'occurrences':len(occurrences),'concept_relations':len(relations),'chunk_relations':len(chunk_edges),
        'model_extracted_chunks':sum(c['status']=='MODEL_EXTRACTED' for c in coverage),
        'heading_baseline_chunks':sum(c['status']=='HEADING_BASELINE' for c in coverage),
        'source_printed_merges':len(merges),'rejected_records':len(errors),
        'model_empty_chunks':sum(c.get('heading_fallback',False) for c in coverage),
        'whitespace_repaired_quotes':sum(bool(ev.get('quote_repair')) for o in occurrences for ev in o['evidence']),
        'unproven_aliases':sum(len(n['candidate_aliases']) for n in nodes.values()),
        'identity_policy':'lexical normalization plus source-printed bilingual equivalence',
        'difficulty':'relative estimate, not measured minutes','weights':'evidence, not probabilities'}
    graph['manifest']['raw_pages_archived']=store.db.execute('SELECT COUNT(*) FROM raw_pages').fetchone()[0]
    store.set_metadata('graph',graph)
    return graph
