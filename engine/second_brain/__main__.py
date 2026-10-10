import argparse
import json
import sys
from pathlib import Path
from .bundles import make_bundles
from .core import Brain
from .providers import create_backend
from .service import SecondBrain

def legacy_extract():
    parser = argparse.ArgumentParser(description='Source-grounded concept graph extraction')
    parser.add_argument('--config', required=True)
    parser.add_argument('--input', required=True, help='JSON with slides, displays, supplements')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    bundles, unresolved = make_bundles(data['slides'], data.get('displays',[]), data.get('supplements',[]))
    backend = create_backend(json.loads(Path(args.config).read_text()))
    try:
        graph = Brain(backend, args.output).build(bundles, unresolved)
        print(json.dumps(graph['manifest'], ensure_ascii=False, indent=2))
        if graph['errors']:
            raise SystemExit(1)
    finally:
        if hasattr(backend, 'close'): backend.close()

def main():
    # Keep the previous extraction-only entry point usable.
    if '--config' in sys.argv and len(sys.argv)>1 and sys.argv[1].startswith('--'):
        return legacy_extract()
    parser=argparse.ArgumentParser(description='Persistent source-grounded second brain')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['ingest','capture','enrich','graph','wiki','search','traverse','plan','quiz','decide','tutor']:
        p=sub.add_parser(command);p.add_argument('--db',required=True);p.add_argument('--output')
        if command=='ingest':
            group=p.add_mutually_exclusive_group(required=True)
            group.add_argument('--input');group.add_argument('--pdf-dir')
            p.add_argument('--chunk-tokens',type=int,default=450)
        if command=='capture':p.add_argument('--input',required=True)
        if command in {'enrich','tutor'}:p.add_argument('--config',required=True)
        if command=='enrich':
            p.add_argument('--limit',type=int);p.add_argument('--chunk-id',action='append')
        if command in {'search','tutor'}:
            p.add_argument('--query',required=True);p.add_argument('--concept-id')
            p.add_argument('--unit-id')
        if command=='tutor':p.add_argument('--history',help='JSON file with recent user/assistant messages')
        if command=='search':
            p.add_argument('--limit',type=int,default=5);p.add_argument('--max-chars',type=int,default=12000)
        if command=='traverse':
            p.add_argument('--concept-id',required=True);p.add_argument('--direction',default='both')
            p.add_argument('--depth',type=int,default=2)
        if command=='plan':
            p.add_argument('--daily-budget',required=True,help='JSON list of remaining daily learning slots')
            p.add_argument('--targets',help='JSON list of chunk IDs');p.add_argument('--weights',help='JSON with costs and values')
            p.add_argument('--include-candidates',action='store_true')
            p.add_argument('--min-weight',type=float,default=0.2)
        if command=='quiz':
            p.add_argument('--unit-id',required=True);p.add_argument('--correct',type=int,required=True)
            p.add_argument('--total',type=int,required=True);p.add_argument('--reference',required=True)
        if command=='decide':
            p.add_argument('--edge-id',required=True);p.add_argument('--status',required=True)
            p.add_argument('--reason',required=True)
    args=parser.parse_args();brain=SecondBrain(args.db);backend=None
    try:
        if args.command=='ingest':
            report=brain.ingest_pdfs(args.pdf_dir,args.chunk_tokens) if args.pdf_dir else \
                brain.ingest(json.loads(Path(args.input).read_text()),args.chunk_tokens)
            result=dict(report,manifest=brain.rebuild()['manifest'])
        elif args.command=='capture':
            result=brain.capture(json.loads(Path(args.input).read_text()))
            result['manifest']=brain.rebuild()['manifest']
        elif args.command=='enrich':
            backend=create_backend(json.loads(Path(args.config).read_text()))
            result=brain.enrich(backend,args.limit,args.chunk_id,
                progress=lambda item:print(json.dumps(item,ensure_ascii=False),file=sys.stderr,flush=True))
            result['manifest']=brain.rebuild()['manifest']
        elif args.command=='graph':result=brain.rebuild()
        elif args.command=='wiki':
            if not args.output:parser.error('wiki requires --output DIRECTORY')
            result=brain.export_wiki(args.output)
        elif args.command=='search':result=brain.search(args.query,args.limit,args.max_chars,args.concept_id,args.unit_id)
        elif args.command=='traverse':result=brain.traverse(args.concept_id,args.direction,args.depth)
        elif args.command=='plan':
            weights=json.loads(Path(args.weights).read_text()) if args.weights else {}
            result=brain.plan(json.loads(args.daily_budget),targets=json.loads(args.targets) if args.targets else None,
                include_candidates=args.include_candidates,min_weight=args.min_weight,
                costs=weights.get('costs'),values=weights.get('values'))
        elif args.command=='quiz':result=brain.record_quiz(args.unit_id,args.correct,args.total,args.reference)
        elif args.command=='decide':
            brain.graph();brain.store.decide_edge(args.edge_id,args.status,args.reason)
            result=brain.rebuild()['manifest']
        elif args.command=='tutor':
            backend=create_backend(json.loads(Path(args.config).read_text()))
            history=json.loads(Path(args.history).read_text()) if args.history else []
            result=brain.tutor(args.query,backend,args.concept_id,args.unit_id,history)
        text=json.dumps(result,ensure_ascii=False,indent=2)
        if args.output and args.command!='wiki':Path(args.output).write_text(text+'\n')
        else:print(text)
    finally:
        if backend is not None and hasattr(backend,'close'):backend.close()
        brain.close()

if __name__ == '__main__': main()
