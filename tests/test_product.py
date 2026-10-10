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
if __name__=='__main__':unittest.main()
