import sys,unicodedata
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pymupdf
from desktop.backend.server import DesktopEngine
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
pdf=pymupdf.open();page=pdf.new_page();page.insert_text((50,60),'Encryption converts plaintext into ciphertext.');file=root/unicodedata.normalize('NFD','한글교안.pdf');pdf.save(file);pdf.close()
e=DesktopEngine(root/'data');b=e.dispatch('import',{'paths':[str(file)]});unit=b['units'][0]
with e.db:
 e.documents[0]['name']=unicodedata.normalize('NFD',e.documents[0]['name']);e.brain.store.set_metadata('app_documents',e.documents)
 for role,text in [('user','My earlier question'),('assistant','## Persisted explanation\n\n**Encryption** protects text.\n\n- Plaintext\n- Ciphertext\n\n```python\nencrypt(text)\n```\n\n<script>window.unsafe=true</script>')]:e.db.execute('INSERT INTO app_messages(unit_id,role,content) VALUES (?,?,?)',(unit['id'],role,text))
e.save_notebook({'document_id':e.documents[0]['id'],'page':1,'text':'# Existing notes\n\n**Keep this text**\n\n| Concept | Meaning |\n| --- | --- |\n| Encryption | Protection |\n\n- [ ] Review quiz','revision':0});e.close()
