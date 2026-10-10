"""Transactional, persistent raw -> chunk -> extraction warehouse (SQLite, no daemon)."""
import json
import sqlite3
from pathlib import Path
from .core import digest


class KnowledgeStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS anchors(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, anchor_id TEXT NOT NULL,
            input_hash TEXT NOT NULL, payload TEXT NOT NULL,
            FOREIGN KEY(anchor_id) REFERENCES anchors(id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS extractions(chunk_id TEXT PRIMARY KEY, input_hash TEXT NOT NULL,
            backend_hash TEXT NOT NULL, payload TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS extraction_attempts(id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL,input_hash TEXT NOT NULL,backend_hash TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS raw_pages(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS edge_decisions(id TEXT PRIMARY KEY, status TEXT NOT NULL,
            reason TEXT NOT NULL, evidence_hash TEXT);
        CREATE TABLE IF NOT EXISTS completions(unit_id TEXT PRIMARY KEY, quiz_reference TEXT NOT NULL);
        ''')
        if 'evidence_hash' not in {r[1] for r in self.db.execute('PRAGMA table_info(edge_decisions)')}:
            self.db.execute('ALTER TABLE edge_decisions ADD COLUMN evidence_hash TEXT')
            self.db.commit()

    @staticmethod
    def encode(value):return json.dumps(value,ensure_ascii=False,sort_keys=True)

    def ingest(self,bundles,chunks,unresolved=()):
        ids=[b['id'] for b in bundles]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate anchor IDs')
        chunk_ids=[c['id'] for c in chunks]
        if len(chunk_ids)!=len(set(chunk_ids)):raise ValueError('Duplicate chunk IDs')
        if any(c['anchor_id'] not in ids for c in chunks):raise ValueError('Chunk has unknown anchor')
        existing={b['id']:b for b in self.anchors()}
        if any(b['id'] in existing and existing[b['id']]['document']!=b['document'] for b in bundles):
            raise ValueError('Anchor ID belongs to another document; use a unique source ID')
        source_ids={}
        for b in [old for old in self.anchors() if old['id'] not in ids]+list(bundles):
            for source in b['sources']:
                if source['id'] in source_ids and source_ids[source['id']]!=source:
                    raise ValueError('Conflicting source ID across imports')
                source_ids[source['id']]=source
        changed=[];removed=[]
        with self.db:
            for b in bundles:
                self.db.execute('INSERT INTO anchors VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                                (b['id'],self.encode(b)))
            for aid in ids:
                old={r[0] for r in self.db.execute('SELECT id FROM chunks WHERE anchor_id=?',(aid,))}
                new={c['id'] for c in chunks if c['anchor_id']==aid}
                for id in old-new:
                    self.db.execute('DELETE FROM completions WHERE unit_id=?',(id,))
                    self.db.execute('DELETE FROM chunks WHERE id=?',(id,));removed.append(id)
            for c in chunks:
                hash=c.get('input_hash',digest(c))
                old=self.db.execute('SELECT input_hash,payload FROM chunks WHERE id=?',(c['id'],)).fetchone()
                if old and old[0]==hash:continue
                self.db.execute('DELETE FROM extractions WHERE chunk_id=?',(c['id'],))
                # Capture context changes invalidate extraction, not quiz completion.
                # Only changing the actual courseware learning unit resets its quiz.
                def learning_identity(value):
                    main=next(s for s in value['sources'] if s['kind']=='courseware')
                    return digest({'text':main['text'],'document':value['document'],'title':value['title']})
                if old and learning_identity(json.loads(old[1]))!=learning_identity(c):
                    self.db.execute('DELETE FROM completions WHERE unit_id=?',(c['id'],))
                self.db.execute('INSERT INTO chunks VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
                    'anchor_id=excluded.anchor_id,input_hash=excluded.input_hash,payload=excluded.payload',
                    (c['id'],c['anchor_id'],hash,self.encode(c)))
                changed.append(c['id'])
            previous=self.get_metadata('unresolved_alignment',[])
            updated={s['id'] for b in bundles for s in b['sources'] if s['kind']!='courseware'}
            updated.update(s['id'] for s in unresolved)
            previous=[s for s in previous if s['id'] not in updated]
            merged={s['id']:s for s in previous+list(unresolved)}
            self.set_metadata('unresolved_alignment',list(merged.values()))
            self.db.execute('DELETE FROM metadata WHERE key="graph"')
        return {'changed_chunks':changed,'removed_chunks':removed,'input_anchors':len(ids)}

    def chunks(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM chunks ORDER BY id')]

    def anchors(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM anchors ORDER BY id')]

    def extraction(self,id):
        row=self.db.execute('SELECT input_hash,backend_hash,payload FROM extractions WHERE chunk_id=?',(id,)).fetchone()
        return {'input_hash':row[0],'backend_hash':row[1],'graph':json.loads(row[2])} if row else None

    def put_extraction(self,chunk,identity,graph):
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row=self.db.execute('SELECT input_hash FROM chunks WHERE id=?',(chunk['id'],)).fetchone()
            stale=not row or row[0]!=chunk['input_hash']
            payload=dict(graph,storage_status='STALE_INPUT_NOT_PUBLISHED') if stale else graph
            self.db.execute('INSERT INTO extraction_attempts(chunk_id,input_hash,backend_hash,payload) VALUES (?,?,?,?)',
                            (chunk['id'],chunk['input_hash'],digest(identity),self.encode(payload)))
            if stale:return False
            if len(graph['calls'])!=1 or graph['calls'][0]['status']!='VALIDATED':return False
            self.db.execute('INSERT OR REPLACE INTO extractions VALUES (?,?,?,?)',
                (chunk['id'],row[0],digest(identity),self.encode(graph)))
            self.db.execute('DELETE FROM metadata WHERE key="graph"')
        return True

    def archive_pages(self,pages):
        with self.db:
            for page in pages:
                self.db.execute('INSERT INTO raw_pages VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                                (page['id'],self.encode(page)))
        return len(pages)

    def set_metadata(self,key,value):
        self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,self.encode(value)))

    def get_metadata(self,key,default=None):
        row=self.db.execute('SELECT payload FROM metadata WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else default

    def decide_edge(self,id,status,reason):
        if status not in {'ACCEPTED','REJECTED','CANDIDATE'} or not reason.strip():
            raise ValueError('Relation status and reason required')
        graph=self.get_metadata('graph')
        if not graph or id not in {e['id'] for e in graph['relations']}:
            raise ValueError('Unknown relation')
        edge=next(e for e in graph['relations'] if e['id']==id)
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO edge_decisions VALUES (?,?,?,?)',
                            (id,status,reason,edge['evidence_hash']))
            self.db.execute('DELETE FROM metadata WHERE key="graph"')

    def decisions(self):
        return {r[0]:{'status':r[1],'reason':r[2],'evidence_hash':r[3]}
                for r in self.db.execute('SELECT id,status,reason,evidence_hash FROM edge_decisions')}

    def close(self):self.db.close()
