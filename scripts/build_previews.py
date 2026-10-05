"""Make original-speed 0..6s GIF previews and 2s thumbnails from actual MP4s."""
from pathlib import Path
import cv2
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
for path in sorted((ROOT/'artifacts/media').rglob('*.mp4')):
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS)
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count < fps*6:raise RuntimeError(f'Incomplete video: {path}')
    frames=[]
    for i in range(60):
        cap.set(cv2.CAP_PROP_POS_FRAMES,round(i*fps/10))
        ok,frame=cap.read()
        if not ok:raise RuntimeError(f'Cannot decode {path} frame {i}')
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        im.thumbnail((480,270))
        frames.append(im)
    frames[0].save(path.with_name(path.stem+'_preview.gif'),save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES,round(2*fps));ok,frame=cap.read()
    if not ok:raise RuntimeError('Cannot decode thumbnail')
    Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(path.with_suffix('.png'))
    print(path.name,f'{count/fps:.1f}s, {fps:g}fps, GIF 6s/10fps')
    cap.release()
