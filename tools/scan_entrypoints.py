import ast
from pathlib import Path
import sys
# ensure workspace root is on sys.path so we can import launcher
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import launcher

scripts = launcher.get_scripts()

print('Scanning for modal entrypoints and function signatures...')
for s in scripts:
    rel = s['rel']
    entrypoints = s.get('entrypoints', [])
    uses_modal = s.get('uses_modal', False)
    if not entrypoints:
        continue
    p = s['path']
    try:
        src = p.read_text(errors='ignore')
    except Exception as e:
        print(f"{rel} -> unable to read file: {e}")
        continue
    try:
        tree = ast.parse(src)
    except Exception as e:
        print(f"{rel} -> parse error: {e}")
        continue
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in entrypoints:
            found.add(node.name)
            args = node.args
            params = []
            # posonlyargs for newer Python
            posonly = getattr(args, 'posonlyargs', [])
            for a in posonly:
                params.append(a.arg)
            # regular args with defaults
            defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
            for a, d in zip(args.args, defaults):
                if d is not None:
                    try:
                        default_repr = ast.unparse(d)
                    except Exception:
                        default_repr = '<default>'
                    params.append(f"{a.arg}={default_repr}")
                else:
                    params.append(a.arg)
            if args.vararg:
                params.append(f"*{args.vararg.arg}")
            for a, d in zip(args.kwonlyargs, args.kw_defaults):
                if d is not None:
                    try:
                        default_repr = ast.unparse(d)
                    except Exception:
                        default_repr = '<default>'
                    params.append(f"{a.arg}={default_repr}")
                else:
                    params.append(a.arg)
            if args.kwarg:
                params.append(f"**{args.kwarg.arg}")
            print(f"{rel} -> {node.name}({', '.join(params)})")
    missing = set(entrypoints) - found
    for m in missing:
        print(f"{rel} -> declared entrypoint '{m}' not found in AST")

print('Scan complete.')
