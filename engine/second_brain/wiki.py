"""Readable, source-grounded concept cards; no generated narrative is treated as fact."""
from collections import defaultdict
from pathlib import Path


def export_wiki(graph,directory):
    directory=Path(directory);cards=directory/'concepts';cards.mkdir(parents=True,exist_ok=True)
    sources={s['id']:s for s in graph['sources']}
    names={n['id']:n['name'] for n in graph['concepts']}
    occurrences=defaultdict(list)
    for o in graph['occurrences']:occurrences[o['concept_id']].append(o)
    index=['# 교안 지식 저장소','',
           '개념·근거·관련 자료를 저장한다. 후보 관계는 검증된 선수 관계와 구분한다.','',
           f"청크 {len(graph['chunks'])}개 / 개념 {len(graph['concepts'])}개",'']
    for node in sorted(graph['concepts'],key=lambda n:(n['name'],n['id'])):
        id=node['id'];local=occurrences[id]
        lines=['# '+node['name'],'',f"개념 ID: {id}",'']
        if node['difficulty_estimate'] is not None:
            lines += [f"난이도 추정: {node['difficulty_estimate']}/3 (학습 시간으로 보정되지 않음)",'']
        if node['aliases']:lines+=['원문에 확인된 표기: '+', '.join(node['aliases']),'']
        goals=sorted({goal for o in local if o['role']=='teaches'
                      for goal in o.get('objectives',[o.get('objective','')]) if goal})
        if goals:lines+=['## 학습 목표','']+['- '+goal for goal in goals]+['']
        lines+=['## 원문 근거와 출현','']
        seen=set()
        for o in sorted(local,key=lambda o:(o['bundle_id'],o['role'])):
            for ev in o['evidence']:
                marker=(ev['source_id'],ev['quote'],o['role'])
                if marker in seen:continue
                seen.add(marker);source=sources[ev['source_id']]
                location=f"{source.get('document',source.get('linked_document','강의 연결 자료'))} p.{source.get('page',source.get('linked_page','연결 교안'))}"
                if source['kind']!='courseware':
                    location+=f" · {source.get('lecture_id','')} {source.get('start_ms',0)}~{source.get('end_ms',0)}ms"
                lines.append(f"- {location} · {o['role']} · {o['origin']} · `{ev['source_id']}`")
                lines.append('  > '+ev['quote'].replace('\n','\n  > '))
        lines+=['','## 개념 관계','']
        for e in graph['relations']:
            if id not in {e['source_id'],e['target_id']}:continue
            a,b=e['source_id'],e['target_id']
            lines.append(f"- [{names[a]}]({a}.md) → [{names[b]}]({b}.md) · {e['kind']} · {e['status']}")
            lines.extend('  - '+reason for reason in e['reasons'])
        lines+=['','## 자료 위치','']
        for unit in sorted({o['bundle_id'] for o in local}):lines.append('- 청크 `'+unit+'`')
        lines+=['','난이도는 상대 추정치이며, 메모로 숙달 여부를 추론하지 않는다.','']
        (cards/(id+'.md')).write_text('\n'.join(lines))
        index.append(f"- [{node['name']}](concepts/{id}.md) — 근거 출현 {len(local)}개")
    (directory/'INDEX.md').write_text('\n'.join(index)+'\n')
    return {'index':str((directory/'INDEX.md').resolve()),'concept_cards':len(graph['concepts'])}
