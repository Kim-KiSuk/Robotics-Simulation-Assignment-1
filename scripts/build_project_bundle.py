"""Package pinned course source + overlay + fixed policy; no simulator installation."""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
BASE='e83a5d2f11ca1b5f03b690e1978479e620c500e2'

def build(source):
    archive=subprocess.check_output(['git','-C',str(source),'archive',BASE])
    entries={}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for info in tar:
            p=Path(info.name)
            if p.parts[0] in ('docs','.github') or '__pycache__' in p.parts or p.suffix in ('.pyc','.pyo'):
                continue
            if info.isfile():entries[info.name]=(tar.extractfile(info).read(),info.mode)
            elif info.issym():
                # The bundle includes source files; external installation links are not shipped.
                continue
    for p in (ROOT/'overlay').rglob('*'):
        if p.is_file():entries[str(p.relative_to(ROOT/'overlay'))]=(p.read_bytes(),0o755 if p.suffix=='.sh' else 0o644)
    for folder in ('evaluation/team_v21','artifacts/checkpoints/final'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file():entries[str(p.relative_to(ROOT))]=(p.read_bytes(),0o644)
    entries['EVALUATION_COMMAND.txt']=((ROOT/'submission/evaluation_only.txt').read_bytes(),0o644)
    entries['SUBMISSION_README.md']=(
        ('# unseen 지형 보행 학습 — 제출 프로젝트\n\n'
         f'원본 commit: {BASE}\n\n'
         '수업 원본 소스 + 과제 overlay + 최종 ant_final.pt + 팀 평가 배포본을 포함한다.\n'
         'Isaac Sim, conda 의존성, 다운로드되는 USD는 별도 설치가 필요하다.\n'
         '이 폴더는 이미 압축 해제된 실행 프로젝트다. conda activate lerobot-arena 후 이 폴더에서 ./isaaclab.sh -i rsl_rl을 실행한다.\n'
         '그 다음 EVALUATION_COMMAND.txt의 cd 경로를 이 폴더의 실제 위치로 맞추고 평가한다.\n'
         '처음 다운로드부터의 안내: https://github.com/Kim-KiSuk/Robotics-Simulation-Assignment-1/blob/main/docs/REPRODUCE.md\n'
         '팀 v2.1 평가 시 TEAM_ANT_EVAL_V21_ROOT를 이 프로젝트의 evaluation/team_v21 절대경로로 지정한다.\n'
         '코드/가중치 식별과 원자료: https://github.com/Kim-KiSuk/Robotics-Simulation-Assignment-1\n').encode(),0o644)
    out=ROOT/'artifacts/project';out.mkdir(parents=True,exist_ok=True)
    target=out/'IsaacLab_RS_final.tar.gz'
    with target.open('wb') as f,gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for name,(data,mode) in sorted(entries.items()):
            info=tarfile.TarInfo('IsaacLab_RS_final/'+name);info.size=len(data);info.mode=mode;info.mtime=0
            tar.addfile(info,io.BytesIO(data))
    metadata={'base_commit':BASE,'archive':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'file_count':len(entries),'excludes':['base docs','base .github','bytecode caches','external symlinks','Isaac Sim binaries','downloaded robot assets'],
              'files':{name:hashlib.sha256(data).hexdigest() for name,(data,mode) in sorted(entries.items())}}
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(target,len(entries),'files',target.stat().st_size,'bytes')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--course-repo',type=Path,required=True)
    build(p.parse_args().course_repo)
