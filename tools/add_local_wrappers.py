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
    print('No candidate files found for adding local_entrypoint wrappers.')
    sys.exit(0)

print(f'Found {len(candidates)} modal scripts without local_entrypoint decorators. Scanning for @app.function/@modal.function...')

modified = []
errors = []
for s in candidates:
    p = s['path']
    rel = s['rel']
    try:
        src = p.read_text(errors='ignore')
    except Exception as e:
        errors.append((rel, f'read error: {e}'))
        continue
    try:
        tree = ast.parse(src)
    except Exception as e:
        errors.append((rel, f'parse error: {e}'))
        continue

    funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # inspect decorators
            has_modal_deco = False
            for dec in node.decorator_list:
                decf = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(decf, ast.Attribute) and isinstance(decf.value, ast.Name):
                    if decf.value.id in ('app', 'modal') and decf.attr in ('function', 'method'):
                        has_modal_deco = True
                        break
                # also allow decorators like @modal.function
                if isinstance(decf, ast.Name) and decf.id in ('function', 'modal'):
                    # best-effort
                    has_modal_deco = True
                    break
            if has_modal_deco:
                funcs.append(node)

    if not funcs:
        print(f'{rel} -> no @app.function/@modal.function decorated top-level functions found, skipping')
        continue

    # prepare wrapper code to append
    wrappers = []
    for fn in funcs:
        name = fn.name
        # build parameter list string
        args = fn.args
        parts = []
        call_args = []
        # posonlyargs (py3.8+)
        posonly = getattr(args, 'posonlyargs', [])
        for a in posonly:
            parts.append(a.arg)
            call_args.append(a.arg)
        # regular args with defaults
        total_args = list(args.args)
        defaults = [None] * (len(total_args) - len(args.defaults)) + list(args.defaults)
        for a, d in zip(total_args, defaults):
            if d is not None:
                try:
                    default_repr = ast.unparse(d)
                except Exception:
                    default_repr = '<default>'
                parts.append(f"{a.arg}={default_repr}")
            else:
                parts.append(a.arg)
            call_args.append(a.arg)
        # vararg
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")
            call_args.append(f"*{args.vararg.arg}")
        # kwonly args
        for a, d in zip(args.kwonlyargs, args.kw_defaults):
            if d is not None:
                try:
                    default_repr = ast.unparse(d)
                except Exception:
                    default_repr = '<default>'
                parts.append(f"{a.arg}={default_repr}")
            else:
                parts.append(a.arg)
            call_args.append(f"{a.arg}={a.arg}")
        # kwarg
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")
            call_args.append(f"**{args.kwarg.arg}")

        param_sig = ', '.join(parts)
        call_sig = ', '.join(call_args)
        # create wrapper name
        wrapper_name = 'entrypoint_' + name
        wrapper = []
        wrapper.append('\n')
        wrapper.append('@app.local_entrypoint()')
        wrapper.append(f"\ndef {wrapper_name}({param_sig}):")
        # body: try to call .local if available, fallback to direct call
        call_line = f"    try:\n        return {name}.local({call_sig})\n    except Exception:\n        return {name}({call_sig})"
        wrapper.append(call_line)
        wrappers.append('\n'.join(wrapper))

    if not wrappers:
        continue

    new_src = src + '\n\n# Auto-generated local_entrypoint wrappers\n' + '\n\n'.join(wrappers)
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
    print(f'{rel} -> appended {len(wrappers)} wrapper(s)')

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
