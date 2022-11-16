from pathlib import Path

name = 'train'
a = Path('./test_folder') / f'{name}'
print(a)
if not a.exists():
    a.mkdir(exist_ok=True,parents=True)
    print('create folder')