"""Static check that every local import (including imports inside functions) and every `-m` module resolves."""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ("modules", "geometry", "experiments", "results", "tests")


def module_file(mod):
    p = ROOT.joinpath(*mod.split("."))
    for f in (p.with_suffix(".py"), p / "__init__.py"):
        if f.is_file():
            return f
    return None


def defined_names(f):
    names = set()
    for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                names |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return names


def python_problems():
    out = []
    for f in sorted(ROOT.rglob("*.py")):
        rel = f.relative_to(ROOT)
        if rel.parts[0] not in LOCAL and rel.name != "reproduce.py":
            continue
        package = list(rel.parts[:-1])
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                pairs = [(a.name, None) for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = package[:len(package) - n.level + 1] if n.level else []
                mod = ".".join(base + ([n.module] if n.module else []))
                pairs = [(mod, a.name) for a in n.names]
            else:
                continue
            for mod, name in pairs:
                if mod.split(".")[0] not in LOCAL:
                    continue
                target = module_file(mod)
                if target is None:
                    out.append(f"{rel}:{n.lineno} missing module {mod}")
                elif name not in (None, "*") and not module_file(f"{mod}.{name}") and name not in defined_names(target):
                    out.append(f"{rel}:{n.lineno} {mod} does not define {name}")
    return out


def module_flag_problems():
    out = []
    for f in sorted([*ROOT.rglob("*.sh"), *ROOT.rglob("*.py"), *ROOT.rglob("*.yml")]):
        if f.relative_to(ROOT).parts[0] in (".venv", "data"):
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for mod in re.findall(r"-m\s+([A-Za-z_][\w.]*)", line):
                if mod.split(".")[0] in LOCAL and module_file(mod) is None:
                    out.append(f"{f.relative_to(ROOT)}:{i} -m {mod} does not exist")
    return out


class ImportTest(unittest.TestCase):
    def test_python_imports_resolve(self):
        self.assertEqual(python_problems(), [])

    def test_module_flags_resolve(self):
        self.assertEqual(module_flag_problems(), [])


if __name__ == "__main__":
    unittest.main()
