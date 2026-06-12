from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import launcher

scripts = launcher.get_scripts()

candidates = [s for s in scripts if s.get('uses_modal') and not s.get('entrypoints')]
if not candidates:
    print('No candidate files found for adding entrypoints.')
    sys.exit(0)

print(f'Found {len(candidates)} modal scripts without detected entrypoints. Processing...')
modified = []
for s in candidates:
    p = s['path']
    rel = s['rel']
    try:
        txt = p.read_text(errors='ignore')
    except Exception as e:
        print(f'{rel} -> could not read: {e}')
        continue
    # skip if decorator already exists anywhere
    if '@app.local_entrypoint' in txt:
        print(f'{rel} -> already has @app.local_entrypoint, skipping')
        continue
    # ensure 'app' exists in file
    if 'app = modal.App' not in txt and 'modal.App(' not in txt:
        print(f'{rel} -> no Modal App instance found, skipping')
        continue
    # find candidate function names
    pattern = re.compile(r"^\s*def\s+(main|entrypoint|run|cli)\s*\(", re.MULTILINE)
    m = pattern.search(txt)
    if not m:
        print(f'{rel} -> no candidate function (main/entrypoint/run/cli) found, skipping')
        continue
    start = m.start()
    # find start of the line
    line_start = txt.rfind('\n', 0, start) + 1
    # move above any decorators
    insert_at = line_start
    # scan backwards from line_start for decorator lines
    cur = line_start
    while True:
        prev_nl = txt.rfind('\n', 0, cur - 1)
        if prev_nl == -1:
            prev_line_start = 0
        else:
            prev_line_start = prev_nl + 1
        prev_line = txt[prev_line_start:cur].strip()
        if prev_line.startswith('@'):
            insert_at = prev_line_start
            cur = prev_line_start
            if prev_line_start == 0:
                break
            continue
        break
    new_txt = txt[:insert_at] + '@app.local_entrypoint()\n' + txt[insert_at:]
    # backup
    bak = p.with_suffix(p.suffix + '.bak')
    try:
        p.write_text(txt, encoding='utf-8')
        bak.write_text(txt, encoding='utf-8')
    except Exception:
        # Fall back to original encoding write
        p.write_text(txt, errors='ignore')
        bak.write_text(txt, errors='ignore')
    # write modified
    try:
        p.write_text(new_txt, encoding='utf-8')
    except Exception:
        p.write_text(new_txt, errors='ignore')
    modified.append(str(rel))
    print(f'{rel} -> inserted @app.local_entrypoint() above function at offset {insert_at}')

print('\nModified files:')
for m in modified:
    print(m)
print('Done.')
