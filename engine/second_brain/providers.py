"""Explicit extraction backends. No routing, fallback, or LOOM installation dependency."""
import hashlib
import json
import os
import queue
import threading
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


class OpenAIAPI:
    provider = 'openai_api'

    def __init__(self, model, api_key_env='OPENAI_API_KEY', timeout=180):
        self.model, self.api_key_env, self.timeout = model, api_key_env, timeout
        self.cache_identity = {'provider': self.provider, 'model': model,
                               'endpoint': 'https://api.openai.com/v1/responses'}

    def complete(self, system, prompt, max_output_tokens=2048):
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f'{self.api_key_env} is not configured')
        payload = {'model': self.model, 'instructions': system, 'input': prompt,
                   'max_output_tokens': max_output_tokens, 'store': False,
                   'text': {'format': {'type': 'json_object'}}}
        request = urllib.request.Request(self.cache_identity['endpoint'],
            data=json.dumps(payload).encode(), headers={
                'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            # Do not echo response bodies or authentication headers into experiment logs.
            raise RuntimeError(f'OpenAI API HTTP {error.code}') from None
        except urllib.error.URLError:
            raise RuntimeError('OpenAI API connection failed') from None
        if result.get('status') != 'completed':
            raise RuntimeError('OpenAI response incomplete or failed')
        text = ''.join(part['text'] for item in result.get('output', [])
            if item.get('type') == 'message' for part in item.get('content', [])
            if part.get('type') == 'output_text')
        if not text:
            raise RuntimeError('OpenAI response contained no text')
        return {'text': text, 'provider': self.provider, 'model': result.get('model', self.model),
                'usage': result.get('usage'), 'latency_ms': int((time.monotonic()-start)*1000)}


class OpenAIOAuth:
    """Official Codex ChatGPT login adapter; OAuth tokens are managed by Codex.

    This is an optional CLI dependency for the OAuth route only, not a generic
    Responses API bearer-token implementation. No user credential files are read.
    """
    provider = 'openai_oauth'

    def __init__(self, model, binary='codex', timeout=180):
        self.model, self.binary, self.timeout = model, str(binary), timeout
        self.cache_identity = {'provider': self.provider, 'model': model,
                               'transport': 'codex-exec-chatgpt-v1'}

    def complete(self, system, prompt, max_output_tokens=2048):
        # Isolate extraction from project instructions, plugins, and user tool config.
        env = {k: v for k, v in os.environ.items()
               if k not in {'OPENAI_API_KEY', 'CODEX_API_KEY', 'CODEX_ACCESS_TOKEN'}}
        with tempfile.TemporaryDirectory(prefix='spartady-oauth-') as directory:
            output = Path(directory)/'response.json'
            command = [self.binary, 'exec', '--ignore-user-config', '--ignore-rules',
                '--ephemeral', '--skip-git-repo-check', '--sandbox', 'read-only',
                '-c', 'forced_login_method="chatgpt"',
                '-c', 'features.shell_tool=false', '-c', 'features.apply_patch_freeform=false',
                '-c', 'web_search="disabled"', '--model', self.model,
                '--json', '--output-last-message', str(output), '-']
            start = time.monotonic()
            try:
                result = subprocess.run(command, input=system+'\n\n'+prompt,
                    cwd=directory, env=env, text=True, encoding='utf-8', capture_output=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                raise TimeoutError('OpenAI OAuth extraction timeout') from None
            except OSError:
                raise RuntimeError('Codex CLI is required for the OAuth option') from None
            if result.returncode:
                raise RuntimeError('OpenAI OAuth extraction failed; check codex login status')
            usage = None
            for line in result.stdout.splitlines():
                event = json.loads(line)
                item = event.get('item', {})
                if item.get('type') not in (None, 'agent_message', 'reasoning'):
                    raise RuntimeError('Unexpected tool activity in OAuth extraction')
                if event.get('type') == 'turn.failed':
                    raise RuntimeError('OpenAI OAuth turn failed')
                if event.get('type') == 'turn.completed':
                    usage = event.get('usage')
            if not output.is_file() or not output.read_text().strip():
                raise RuntimeError('OpenAI OAuth returned no final response')
            # CLI has no matching output-token cap; record this limitation explicitly.
            return {'text': output.read_text(), 'provider': self.provider, 'model': self.model,
                'usage': usage, 'latency_ms': int((time.monotonic()-start)*1000),
                'output_token_limit_enforced': False}


class LocalLMX:
    """Project-owned local subprocess; weights and inference never leave the device."""
    provider = 'local_lmx'

    def __init__(self, model_path, binary=None, timeout=300):
        path = Path(model_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError('Local model must be a file')
        # Native LMX header contains the manifest checksum; no full weight scan needed.
        with path.open('rb') as stream:
            header = stream.read(128)
        if header[:8] != b'LOOMLM2\0':
            raise ValueError('Local backend requires a native LMX v2 model')
        self.model = 'lmx/'+header[40:72].hex()
        self.timeout = timeout
        binary = Path(binary or Path(__file__).resolve().parents[2]/
            ('integrations/local_lmx/target/release/spartady-local-lmx.exe' if os.name=='nt' else 'integrations/local_lmx/target/release/spartady-local-lmx')).resolve(strict=True)
        self.cache_identity = {'provider': self.provider, 'model': self.model,
            'runtime_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'model_bytes': path.stat().st_size, 'model_mtime_ns': path.stat().st_mtime_ns,
            'options': {k:v for k,v in os.environ.items() if k.startswith('SPARTADY_LMX_')}}
        self.process = subprocess.Popen([str(binary), str(path)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)

        self.responses=queue.Queue()
        def read_responses():
            for line in self.process.stdout:self.responses.put(line)
            self.responses.put(None)
        threading.Thread(target=read_responses,daemon=True).start()

    def complete(self, system, prompt, max_output_tokens=2048):
        self.process.stdin.write(json.dumps({'system':system,'prompt':prompt,
            'max_output_tokens':max_output_tokens}, ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        try:line=self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close();raise TimeoutError('Local LMX inference timeout') from None
        if not line:
            raise RuntimeError('Local LMX runtime exited without a response')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        if result.get('provider') != self.provider:
            raise RuntimeError('Invalid local inference provider')
        if result.get('stop') == 'length':
            raise RuntimeError('Local LMX output token limit reached')
        # Stable identity used by the Python cache, native base identity retained separately.
        result['native_model_identity'] = result['model']
        result['model'] = self.model
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe: pipe.close()


def create_backend(config):
    provider = config['provider']
    if provider == 'openai_api':
        return OpenAIAPI(config['model'], config.get('api_key_env','OPENAI_API_KEY'))
    if provider == 'openai_oauth':
        return OpenAIOAuth(config['model'], config.get('binary','codex'))
    if provider == 'local_lmx':
        return LocalLMX(config['model_path'], config.get('binary'))
    raise ValueError(f'Unknown provider: {provider}')
