"""PEP 517 setuptools CMake build driver for M1Triton (`_C` Metal extension + bundled `m1triton-opt`)."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Sentinel extension built via CMake + Ninja."""

    def __init__(self, name: str, source_dir: str = ".") -> None:
        super().__init__(name, sources=[])
        self.source_dir = str(Path(source_dir).resolve())


class CMakeBuildExt(build_ext):
    """Build `_C` and `m1triton-opt` with Apple Clang + Ninja and stage them into the package."""

    def _find_cxx_compiler(self) -> str:
        apple_clang = Path("/usr/bin/clang++")
        if apple_clang.is_file() and os.access(apple_clang, os.X_OK):
            return str(apple_clang)
        found = shutil.which("clang++")
        if found:
            return found
        raise RuntimeError("Apple Clang compiler (`/usr/bin/clang++`) is required to build m1triton.")

    def _stage_opt_binary(self, opt_src: Path, pkg_dir: Path) -> Path:
        opt_dst = pkg_dir / "bin" / "m1triton-opt"
        opt_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(opt_src, opt_dst)
        mode = opt_dst.stat().st_mode
        opt_dst.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return opt_dst

    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        import pybind11

        cmake_bin = shutil.which("cmake")
        if not cmake_bin:
            raise RuntimeError("`cmake` was not found on PATH.")

        cxx_bin = self._find_cxx_compiler()
        build_temp = Path(self.build_temp).resolve() / ext.name.replace(".", "_")
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"
        cmake_args = [
            cmake_bin,
            "-S",
            ext.source_dir,
            "-B",
            str(build_temp),
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_CXX_COMPILER={cxx_bin}",
            f"-DCMAKE_OBJCXX_COMPILER={cxx_bin}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        ]
        if shutil.which("ninja"):
            cmake_args.extend(["-G", "Ninja"])

        subprocess.run(cmake_args, check=True)

        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL") or str(os.cpu_count() or 4)
        subprocess.run(
            [cmake_bin, "--build", str(build_temp), "--target", "_C", "m1triton-opt", "-j", jobs],
            check=True,
        )

        built_sos = sorted((build_temp / "m1triton").glob("_C*.so"))
        if not built_sos:
            raise RuntimeError(f"CMake build succeeded but `_C*.so` was not found under {build_temp / 'm1triton'}")
        built_so = built_sos[0]

        opt_src = build_temp / "m1triton-opt"
        if not opt_src.is_file():
            raise RuntimeError(f"CMake build succeeded but `m1triton-opt` was not found at {opt_src}")

        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        ext_fullpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_so, ext_fullpath)
        self._stage_opt_binary(opt_src, ext_fullpath.parent)

        if self.inplace:
            src_pkg_dir = Path(ext.source_dir) / "python" / "m1triton"
            src_so = src_pkg_dir / ext_fullpath.name
            if src_so.resolve() != ext_fullpath:
                src_pkg_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built_so, src_so)
            if src_pkg_dir.resolve() != ext_fullpath.parent.resolve():
                self._stage_opt_binary(opt_src, src_pkg_dir)

    def get_outputs(self) -> list[str]:
        outputs = list(super().get_outputs())
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                ext_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
                opt_path = ext_dir / "bin" / "m1triton-opt"
                if opt_path.is_file():
                    outputs.append(str(opt_path))
        return outputs

    def copy_extensions_to_source(self) -> None:
        super().copy_extensions_to_source()
        build_py = self.get_finalized_command("build_py")
        for ext in self.extensions:
            if not isinstance(ext, CMakeExtension):
                continue
            inplace_file, regular_file = self._get_inplace_equivalent(build_py, ext)
            built_opt = Path(regular_file).resolve().parent / "bin" / "m1triton-opt"
            inplace_pkg_dir = Path(inplace_file).resolve().parent
            if built_opt.is_file() and (inplace_pkg_dir / "bin" / "m1triton-opt").resolve() != built_opt:
                self._stage_opt_binary(built_opt, inplace_pkg_dir)


setup(
    ext_modules=[CMakeExtension("m1triton._C")],
    cmdclass={"build_ext": CMakeBuildExt},
)
