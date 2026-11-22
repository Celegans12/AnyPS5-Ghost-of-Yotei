import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_emutls_imports

IMPORTS = """
The Import Tables (interpreted .idata section contents):
  DLL Name: KERNEL32.dll
  vma:  Hint/Ord Member-Name Bound-To
  00000000   1  ExitProcess
  DLL Name: libgcc_s_seh-1.dll
  vma:  Hint/Ord Member-Name Bound-To
  00176fd0  45  __emutls_get_address
"""

OTHER_DLL = """
The Import Tables (interpreted .idata section contents):
  DLL Name: other.dll
  vma:  Hint/Ord Member-Name Bound-To
  00176fd0  45  __emutls_get_address
"""

OUTSIDE_IMPORTS = """
Export Address Table -- Ordinal Base 1
  00176fd0  45  __emutls_get_address
"""


class ImportedFromTests(unittest.TestCase):
    def test_libgcc_import(self):
        self.assertEqual(check_emutls_imports.imported_from(IMPORTS), "libgcc_s_seh-1.dll")

    def test_other_dll(self):
        self.assertIsNone(check_emutls_imports.imported_from(OTHER_DLL))

    def test_symbol_outside_import_table(self):
        self.assertIsNone(check_emutls_imports.imported_from(OUTSIDE_IMPORTS))


class PeModulesTests(unittest.TestCase):
    def test_selects_pe_modules_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.prx").write_bytes(b"MZ")
            (root / "b.prx").write_bytes(b"\x7fELF")
            (root / "c.txt").write_bytes(b"MZ")
            self.assertEqual([path.name for path in check_emutls_imports.pe_modules(root)], ["a.prx"])

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                check_emutls_imports.pe_modules(Path(directory))


if __name__ == "__main__":
    unittest.main()
