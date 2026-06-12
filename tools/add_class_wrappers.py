import ast
from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import launcher

scripts = launcher.get_scripts()

candidates = [s for s in scripts if s.get('uses_modal') and not s.get('entrypoints')]
if not candidates:
    print('No candidate files found for class wrapper generation.')
    sys.exit(0)

print(f'Found {len(candidates)} modal scripts without entrypoints. Scanning for @app.cls classes...')

modified = []
failed = []
for s in candidates:
    p = s['path']
    rel = s['rel']
    try:
        src = p.read_text(errors='ignore')
    except Exception as e:
        print(f'{rel} -> read error: {e}')
        continue
    try:
        tree = ast.parse(src)
    except Exception as e:
        print(f'{rel} -> parse error: {e}')
        continue

    app_name_literal = None
    # find app assignment like: app = modal.App("example-name")
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'app':
                    val = node.value
                    if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute):
                        if getattr(val.func.value, 'id', None) == 'modal' and val.func.attr == 'App':
                            if val.args:
                                first = val.args[0]
                                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                                    app_name_literal = first.value
                                    break
        if app_name_literal:
            break

    class_names = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                # handle @app.cls or @modal.cls
                if isinstance(dec, ast.Attribute):
                    if isinstance(dec.value, ast.Name) and dec.attr == 'cls':
                        if dec.value.id in ('app', 'modal'):
                            class_names.append(node.name)
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                    if isinstance(dec.func.value, ast.Name) and dec.func.attr == 'cls':
                        if dec.func.value.id in ('app', 'modal'):
                            class_names.append(node.name)

    if not class_names:
        print(f'{rel} -> no @app.cls-decorated classes found, skipping')
        continue

    wrappers = []
    for cname in class_names:
        wrapper_name = f'entrypoint_{cname}'
        # build code
        if app_name_literal:
            code = (
                f"@app.local_entrypoint()\n"
                f"def {wrapper_name}():\n"
                f"    \"\"\"Auto-generated wrapper to instantiate {cname}\"\"\"\n"
                f"    try:\n"
                f"        Cls = modal.Cls.from_name({repr(app_name_literal)}, {repr(cname)})\n"
                f"    except Exception:\n"
                f"        try:\n"
                f"            Cls = modal.Cls.from_name(app.name, {repr(cname)})\n"
                f"        except Exception as e:\n"
                f"            raise RuntimeError('Could not resolve class {cname} for local entrypoint: ' + str(e))\n"
                f"    inst = Cls()\n"
                f"    return 'instantiated {cname}'\n"
            )
        else:
            code = (
                f"@app.local_entrypoint()\n"
                f"def {wrapper_name}():\n"
                f"    \"\"\"Auto-generated wrapper to instantiate {cname} (app name not literal in file).\"\"\"\n"
                f"    try:\n"
                f"        Cls = modal.Cls.from_name(app.name, {repr(cname)})\n"
                f"    except Exception as e:\n"
                f"        raise RuntimeError('Could not resolve class {cname} for local entrypoint: ' + str(e))\n"
                f"    inst = Cls()\n"
                f"    return 'instantiated {cname}'\n"
            )
        wrappers.append(code)

    if not wrappers:
        continue

    new_src = src + '\n\n# Auto-generated class local_entrypoint wrappers\n' + '\n\n'.join(wrappers)
    # backup
    bak = p.with_suffix(p.suffix + '.bak')
    try:
        bak.write_text(src, encoding='utf-8')
    except Exception:
        bak.write_text(src, errors='ignore')
    try:
        p.write_text(new_src, encoding='utf-8')
    except Exception:
        p.write_text(new_src, errors='ignore')
    modified.append(str(rel))
    print(f'{rel} -> appended {len(wrappers)} class wrapper(s)')

# compile modified files
if modified:
    print('\nVerifying modified files compile...')
    import subprocess
    failed = []
    for m in modified:
        path = ROOT / m
        cmd = [sys.executable, '-m', 'py_compile', str(path)]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            failed.append(m)
    if failed:
        print('\nCompilation failed for:')
        for f in failed:
            print(f)
        print('Reverting changes for failed files...')
        for f in failed:
            p = ROOT / f
            bak = p.with_suffix(p.suffix + '.bak')
            if bak.exists():
                p.write_text(bak.read_text(errors='ignore'), encoding='utf-8')
                print(f'{f} reverted from backup')
    else:
        print('All modified files compiled cleanly.')

print('\nDone.')
