import sys, time, io, base64
sys.path.insert(0, '/Users/slian/Desktop/3D/Worldedge/fastfillout' if False else '/Volumes/harddisk/fastfillout/unknownsize-20261013/probes-20261021/scripts')
from judge_lib import *
from PIL import Image
t0 = time.time()
while time.time() - t0 < 3 * 3600:
    st, text, u = post([{'type': 'text', 'text': 'Reply with the word ok.'}], 'probe-text')
    print(time.strftime('%H:%M:%S'), st, flush=True)
    if st == 'ok': break
    time.sleep(300)
else: raise SystemExit('NEVER AVAILABLE')
im = Image.new('RGB', (64, 64), (255, 0, 0)); b = io.BytesIO(); im.save(b, 'PNG')
st, text, u = post([{'type': 'text', 'text': 'What single colour is this image? One word.'},
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()}}], 'probe-image')
print('image probe', st, text, flush=True)
