#!/usr/bin/env python3
"""Reassemble pinned archive parts and extract only declared, new evidence files."""
import hashlib,json,shutil,tarfile,tempfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parent
m=json.loads((ROOT/'TRANSPORT.json').read_text())
def require(ok,message):
    if not ok:raise ValueError(message)
with tempfile.TemporaryDirectory() as tmp:
    archive=Path(tmp)/'evidence.tar.gz';h=hashlib.sha256()
    with archive.open('wb') as out:
        for item in m['parts']:
            name=item['path'];require(Path(name).name==name,'Unsafe part path')
            data=(ROOT/name).read_bytes();require(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'Archive part mismatch: '+name)
            out.write(data);h.update(data)
    require(h.hexdigest()==m['archive_sha256'],'Combined archive mismatch')
    inventory=json.loads((ROOT/'PUBLIC-FILES.json').read_text())
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[v.name for v in members]
        require(len(names)==len(set(names)) and set(names)==set(m['members']),'Archive member list mismatch')
        for member in members:
            pp=PurePosixPath(member.name);target=ROOT/member.name
            require(member.isfile() and not pp.is_absolute() and '..' not in pp.parts and member.name in inventory,'Unsafe or undeclared archive member')
            require(not target.exists() and not target.is_symlink(),'Refusing to replace existing evidence: '+member.name)
            require(not any(p.is_symlink() for p in target.parents),'Symlink parent path')
            require(member.size==m['members'][member.name],'Member size mismatch')
        for member in members:
            target=ROOT/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
            require(hashlib.sha256(target.read_bytes()).hexdigest()==inventory[member.name],'Extracted hash mismatch')
print('Evidence extracted and hash-verified. Run python verify_public.py next.')
