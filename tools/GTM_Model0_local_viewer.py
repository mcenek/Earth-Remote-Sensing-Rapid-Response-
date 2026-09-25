"""Local, read-only GTM experiment viewer. No model inference or acquisition on requests."""
from __future__ import annotations
import argparse
import json
import mimetypes
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
import webbrowser

ROOT=Path(__file__).resolve().parents[1]
UI=ROOT/'GTM_Viewer'
BUNDLES=ROOT/'outputs/GTM_viewer_bundles'
PORTABLE=UI/'portable_bundles'
DOCS=ROOT/'docs'

def inside(root, relative):
    target=(root/relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Path outside viewer directory')
    return target

def registry():
    result=json.loads((UI/'GTM_Model0_registry.json').read_text(encoding='utf-8'))
    indexed={e['id']:e for e in result['experiments']}
    warnings=[]
    bundle_ids=set()
    manifests=sorted(PORTABLE.glob('*/experiment.json'))+sorted(BUNDLES.glob('*/experiment.json'))
    for manifest in manifests:
        try:
            if manifest.stat().st_size>50*1024*1024:
                raise ValueError('Manifest exceeds 50 MiB')
            data=json.loads(manifest.read_text(encoding='utf-8-sig'))
            e=data.get('experiment',data)
            if not isinstance(e.get('id'),str) or not isinstance(e.get('scenes'),list):
                raise ValueError('Need experiment id and scenes')
            if e['id'] in bundle_ids:
                raise ValueError('Duplicate bundle experiment id: '+e['id'])
            def asset(value):
                if not value or value.startswith('data:image/'):
                    return value
                if not inside(manifest.parent,value).is_file():
                    raise ValueError('Missing asset: '+value)
                prefix='portable-bundles/' if manifest.parent.parent==PORTABLE else 'bundle-assets/'
                return prefix+manifest.parent.name+'/'+value.replace('\\','/')
            for s in e['scenes']:
                for k in ['rgb','emit','truthImage','predictionImage','probabilityImage','disagreementImage','observedImage','coarseImage','residualImage','reference90Rgb','reference365Rgb']:
                    if s.get(k): s[k]=asset(s[k])
                if isinstance(s.get('grid'),str): s['grid']=asset(s['grid'])
            if e.get('reportUrl'): e['reportUrl']=asset(e['reportUrl'])
            indexed[e['id']]=e
            bundle_ids.add(e['id'])
        except (ValueError,TypeError,KeyError,OSError) as error:
            warnings.append(manifest.parent.name+': '+str(error))
    # Archived bundle metadata must not override the lifecycle declared by the reset.
    for identifier, experiment in indexed.items():
        if identifier in {'GTM_Model0','GTM_Model1','GTM_Model2','GTM_Model3','GTM_Model4','GTM_Model5'}:
            experiment['lifecycle']='archived'
            experiment['priority']=False
            if not experiment.get('status','').startswith('ARCHIVED'):
                experiment['status']='ARCHIVED · '+experiment.get('status','Historical evidence')
    preferred=result.get('activeExperimentId','GTM_Model6')
    result['activeExperimentId']=preferred if preferred in indexed else 'GTM_Model6_mapper_r5_portable' if 'GTM_Model6_mapper_r5_portable' in indexed else 'GTM_Model6'
    result['experiments']=list(indexed.values())
    result['local']={'bundleDirectory':str(BUNDLES),'warnings':warnings,'mode':'Local read-only viewer'}
    return result

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route=unquote(urlsplit(self.path).path)
        if route=='/api/experiments':
            try: return self.send_bytes(json.dumps(registry()).encode(), 'application/json; charset=utf-8')
            except Exception as error: return self.send_error(500,str(error))
        if route=='/api/health':
            return self.send_bytes(json.dumps({'app':'GTM local experiment viewer','version':1,'local_only':True}).encode(),'application/json')
        try:
            if route.startswith('/bundle-assets/'):
                path=inside(BUNDLES,route[len('/bundle-assets/'):])
            elif route.startswith('/portable-bundles/'):
                path=inside(PORTABLE,route[len('/portable-bundles/'):])
            elif route.startswith('/research-docs/'):
                path=inside(DOCS,route[len('/research-docs/'):])
            else:
                path=inside(UI,route.lstrip('/') or 'index.html')
            if not path.is_file(): return self.send_error(404)
            content_type=mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
            return self.send_bytes(path.read_bytes(),content_type)
        except (ValueError,OSError): return self.send_error(404)
    def send_bytes(self,content,content_type):
        self.send_response(200)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(content)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        try: self.wfile.write(content)
        except (BrokenPipeError,ConnectionResetError): pass
    def log_message(self,fmt,*args):
        if args and str(args[0]).startswith('GET /api/health'): return
        super().log_message(fmt,*args)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args()
    BUNDLES.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    url=f'http://127.0.0.1:{args.port}/'
    print('GTM local experiment viewer: '+url,flush=True)
    print('Experiment folder: '+str(BUNDLES),flush=True)
    if args.open: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__': main()
