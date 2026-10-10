import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from desktop.backend.server import DesktopEngine,today
import datetime
import pymupdf

class ProductTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.engine=DesktopEngine(self.root/'data')
 def tearDown(self):self.engine.close();self.temp.cleanup()
 def import_pdf(self):
  file=self.root/'lecture.pdf';pdf=pymupdf.open();p=pdf.new_page();p.insert_text((50,60),'Encryption and decryption');p.insert_text((50,90),'Encryption converts plaintext into ciphertext. Decryption reverses it.');pdf.save(file);pdf.close()
  return self.engine.dispatch('import',{'paths':[str(file)]})
 def test_clean_start_and_import(self):
  b=self.engine.bootstrap();self.assertEqual(b['documents'],[]);self.assertEqual(b['units'],[])
  b=self.import_pdf();self.assertEqual(len(b['documents']),1);self.assertTrue(b['units'])
 def test_separate_models_do_not_change_graph(self):
  b=self.import_pdf();knowledge=b['settings']['knowledge_model']
  self.engine.save_settings({'learning_model':{'provider':'openai_api','model':'test-model'}})
  a=self.engine.bootstrap();self.assertEqual(a['settings']['knowledge_model'],knowledge);self.assertEqual(a['units'],b['units']);self.assertEqual(a['plan'],b['plan'])
 def test_budget_planning_no_llm(self):
  b=self.import_pdf();ids=[u['id'] for u in b['units']]
  self.engine.get_backend=lambda *a: (_ for _ in ()).throw(AssertionError('LLM planner'))
  self.engine.save_exam({'title':'Test','date':(today()+datetime.timedelta(days=3)).isoformat(),'targets':ids})
  p=self.engine.make_plan();self.assertEqual(p['llm_calls'],0)
  self.engine.dispatch('budget',{'date':today().isoformat(),'minutes':0});self.assertEqual(self.engine.make_plan()['days'][0]['units'],[])
 def test_lecture_notebook(self):
  b=self.import_pdf();doc=b['documents'][0]
  n=self.engine.save_notebook({'document_id':doc['id'],'page':1,'text':'# My lecture\n\nEncryption','revision':0})
  self.assertEqual(n['revision'],1);self.assertEqual(self.engine.notebook({'document_id':doc['id']})['text'],n['text'])
 def test_unicode_filename_matches_exam_scope(self):
  import unicodedata,shutil
  original=self.import_pdf()['documents'][0]
  source=self.root/unicodedata.normalize('NFD','한글교안.pdf');shutil.copy2(self.root/'lecture.pdf',source)
  b=self.engine.dispatch('import',{'paths':[str(source)]});d=b['documents'][-1]
  targets=[u['id'] for u in b['units'] if u['document']==d['name']]
  self.assertTrue(targets);self.assertEqual(d['name'],unicodedata.normalize('NFC',d['name']))
  self.engine.save_exam({'title':'Unicode exam','date':(today()+datetime.timedelta(days=2)).isoformat(),'targets':targets})
 def test_session_survives_restart_and_new_session_is_separate(self):
  b=self.import_pdf();unit=b['units'][0];session=self.engine.dispatch('learning_session',{'unit_id':unit['id']})
  source=self.engine.chunk(unit['id'])['sources'][0]
  class Tutor:
   def complete(self,*a,**kw):return {'text':json.dumps({'answer':'Saved answer','source_ids':[source['id']]})}
  self.engine.get_backend=lambda:Tutor()
  self.engine.dispatch('tutor',{'unit_id':unit['id'],'session_id':session['id'],'question':'Explain encryption'})
  directory=self.engine.directory;self.engine.close();self.engine=DesktopEngine(directory)
  resumed=self.engine.dispatch('learning_session',{'unit_id':unit['id']})
  self.assertEqual(resumed['id'],session['id']);self.assertEqual(len(resumed['messages']),2)
  fresh=self.engine.dispatch('learning_session',{'unit_id':unit['id'],'new':True})
  self.assertNotEqual(fresh['id'],resumed['id']);self.assertEqual(fresh['messages'],[])
  previous=self.engine.dispatch('learning_session',{'unit_id':unit['id'],'session_id':resumed['id']})
  self.assertEqual(previous['messages'],resumed['messages'])
 def test_failed_response_keeps_question_in_session(self):
  unit=self.import_pdf()['units'][0]
  class Failed:
   def complete(self,*a,**kw):raise RuntimeError('offline')
  self.engine.get_backend=lambda:Failed()
  with self.assertRaises(RuntimeError):self.engine.dispatch('tutor',{'unit_id':unit['id'],'question':'Keep this question'})
  session=self.engine.dispatch('learning_session',{'unit_id':unit['id']})
  self.assertEqual([m['content'] for m in session['messages']],['Keep this question'])
 def test_legacy_messages_migrate_without_loss(self):
  unit=self.import_pdf()['units'][0]
  with self.engine.db:self.engine.db.execute('INSERT INTO app_messages(unit_id,role,content) VALUES (?,?,?)',(unit['id'],'user','Earlier conversation'))
  directory=self.engine.directory;self.engine.close();self.engine=DesktopEngine(directory)
  session=self.engine.dispatch('learning_session',{'unit_id':unit['id']})
  self.assertTrue(session['migrated']);self.assertEqual(session['messages'][0]['content'],'Earlier conversation')
 def test_quiz_outcome_is_attached_to_its_session(self):
  unit=self.import_pdf()['units'][0];source=self.engine.chunk(unit['id'])['sources'][0]
  class Quiz:
   def complete(self,*a,**kw):return {'text':json.dumps({'questions':[{'question':f'Question {i}','choices':['correct','wrong1','wrong2','wrong3'],'correct':0,'explanation':'Evidence','source_id':source['id'],'quote':source['text'][:25]} for i in range(3)]})}
  self.engine.get_backend=lambda:Quiz();session=self.engine.dispatch('learning_session',{'unit_id':unit['id']})
  q=self.engine.quiz({'unit_id':unit['id'],'session_id':session['id']})
  self.assertNotIn('correct',q['questions'][0])
  private=json.loads(self.engine.db.execute('SELECT payload FROM app_quizzes WHERE id=?',(q['id'],)).fetchone()[0])
  self.engine.grade({'quiz_id':q['id'],'answers':[x['correct'] for x in private['questions']]})
  record=self.engine.dispatch('learning_session',{'unit_id':unit['id'],'session_id':session['id']})
  self.assertEqual(record['status'],'completed');self.assertTrue(record['quizzes'][0]['passed']);self.assertEqual(record['last_quiz']['correct'],3)
 def test_first_model_download_configures_empty_roles_and_preserves_existing_selection(self):
  from unittest.mock import patch
  model=self.root/'downloaded.lmx';model.write_bytes(b'fixture')
  result={'name':'Qwen3.5 0.8B','path':str(model)}
  with patch('desktop.backend.server.install_model',return_value=result):
   self.engine.dispatch('install_model',{'repo':'fixture/model','filename':'fixture.gguf'})
  conf=self.engine.config()
  self.assertEqual(conf['knowledge_model']['model_path'],str(model.resolve()));self.assertEqual(conf['learning_model']['model_path'],str(model.resolve()))
  self.engine.save_settings({'learning_model':{'provider':'openai_api','model':'custom-learning'}})
  other=self.root/'other.lmx';other.write_bytes(b'fixture')
  with patch('desktop.backend.server.install_model',return_value={'path':str(other)}):self.engine.dispatch('install_model',{})
  conf=self.engine.config();self.assertEqual(conf['knowledge_model']['model_path'],str(model.resolve()));self.assertEqual(conf['learning_model']['model'],'custom-learning')
 def test_installed_model_is_not_duplicated_by_path_alias(self):
  folder=self.engine.directory/'models';folder.mkdir();model=folder/'Qwen3.5-0.8B-Q4_K_M.lmx';model.write_bytes(b'fixture')
  alias=folder/'..'/'models'/model.name
  self.engine.save_settings({'knowledge_model':{'provider':'local_lmx','model_path':str(alias)}})
  models=self.engine.local_models()
  self.assertEqual(len(models),1);self.assertEqual(models[0]['path'],self.engine.config()['knowledge_model']['model_path']);self.assertEqual(models[0]['name'],'Qwen3.5 0.8B · Q4_K_M')
if __name__=='__main__':unittest.main()
