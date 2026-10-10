"""Persistent learning sessions; raw messages and quiz outcomes remain server-owned."""
import datetime,json,uuid

def timestamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
class LearningSessions:
 def init_sessions(self):
  self.db.execute('CREATE TABLE IF NOT EXISTS app_sessions(id TEXT PRIMARY KEY,unit_id TEXT NOT NULL,payload TEXT NOT NULL)')
  columns={r[1] for r in self.db.execute('PRAGMA table_info(app_messages)')}
  for name,kind in [('session_id','TEXT'),('created_at','TEXT')]:
   if name not in columns:self.db.execute('ALTER TABLE app_messages ADD COLUMN '+name+' '+kind)
  self.db.commit()
  for (unit_id,) in self.db.execute('SELECT DISTINCT unit_id FROM app_messages WHERE session_id IS NULL').fetchall():
   try:session=self.create_session(unit_id,migrated=True)
   except ValueError:continue
   with self.db:
    self.db.execute('UPDATE app_messages SET session_id=? WHERE unit_id=? AND session_id IS NULL',(session['id'],unit_id))
    for id,payload in self.db.execute('SELECT id,payload FROM app_quizzes').fetchall():
     q=json.loads(payload)
     if q['unit_id']==unit_id and not q.get('session_id'):
      q['session_id']=session['id'];self.db.execute('UPDATE app_quizzes SET payload=? WHERE id=?',(json.dumps(q,ensure_ascii=False),id))
 def create_session(self,unit_id,migrated=False):
  unit=self.chunk(unit_id);now=timestamp()
  value={'id':uuid.uuid4().hex,'unit_id':unit_id,'title':unit['title'],'document':unit['document'],'page':unit['page'],
         'started_at':None if migrated else now,'updated_at':now,'status':'in_progress','model':self.config()['learning_model'],'migrated':migrated}
  with self.db:self.db.execute('INSERT INTO app_sessions VALUES (?,?,?)',(value['id'],unit_id,json.dumps(value,ensure_ascii=False)))
  return value
 def get_session(self,args):
  unit_id=args['unit_id'];self.chunk(unit_id)
  if args.get('session_id'):
   row=self.db.execute('SELECT payload FROM app_sessions WHERE id=? AND unit_id=?',(args['session_id'],unit_id)).fetchone()
   if not row:raise ValueError('학습 세션을 찾지 못했습니다.')
   return json.loads(row[0])
  if not args.get('new'):
   row=self.db.execute('SELECT payload FROM app_sessions WHERE unit_id=? ORDER BY rowid DESC LIMIT 1',(unit_id,)).fetchone()
   if row:return json.loads(row[0])
  return self.create_session(unit_id)
 def touch_session(self,session,**changes):
  session=dict(session,updated_at=timestamp(),**changes)
  with self.db:self.db.execute('UPDATE app_sessions SET payload=? WHERE id=?',(json.dumps(session,ensure_ascii=False),session['id']))
  return session
 def add_message(self,session,role,content,sources=()):
  with self.db:self.db.execute('INSERT INTO app_messages(unit_id,role,content,sources,session_id,created_at) VALUES (?,?,?,?,?,?)',
                              (session['unit_id'],role,content,json.dumps(sources,ensure_ascii=False),session['id'],timestamp()))
  self.touch_session(session,model=self.config()['learning_model'])
 def session_messages(self,id):
  return [{'role':r[0],'content':r[1],'sources':json.loads(r[2]),'created_at':r[3]} for r in self.db.execute(
      'SELECT role,content,sources,created_at FROM app_messages WHERE session_id=? ORDER BY id',(id,))]
 def session_record(self,args):
  session=self.get_session(args);quizzes=[]
  for payload,result in self.db.execute('SELECT payload,result FROM app_quizzes ORDER BY rowid'):
   q=json.loads(payload)
   if q.get('session_id')==session['id']:
    grade=json.loads(result) if result else None
    quizzes.append({'id':q['id'],'correct':grade.get('correct') if grade else None,'total':3,'passed':grade.get('quiz_passed') if grade else None})
  return dict(session,messages=self.session_messages(session['id']),quizzes=quizzes)
 def session_list(self):
  records=[]
  for (payload,) in self.db.execute('SELECT payload FROM app_sessions ORDER BY rowid DESC'):
   session=json.loads(payload);session['message_count']=self.db.execute('SELECT COUNT(*) FROM app_messages WHERE session_id=?',(session['id'],)).fetchone()[0]
   records.append(session)
  return records
