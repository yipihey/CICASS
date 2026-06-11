# CICASS — EnzoNG.jl wrapper fork

This is `yipihey/CICASS`, a fork of [`astromcquinn/CICASS`](https://github.com/astromcquinn/CICASS)
(O'Leary & McQuinn 2012; McQuinn & O'Leary 2012) used as the streaming-velocity
initial-conditions generator in the EnzoNG.jl unified multi-code framework.

`origin` is this fork; `upstream` is McQuinn's repo (fetch-only, push disabled) —
so we can pull upstream updates but never push to it.

## What this fork adds (on top of upstream CICASS)

- **`makeCosICs/capi_out.c`** — `WriteCapiSnapshot`, an HDF5-free raw writer
  (`-DOUTPUT_CAPI`) that dumps `{DM particles + gas grid + streaming offset}` as a
  flat little-endian `.cicass` file (magic `CICASS01`). Dodges the
  `libhdf5 ∥ libenzo` collision; read by Julia's `CICASSLib.read_snapshot`.
- **`makeCosICs/cicass_capi.cc`** — a C-ABI shim (`cicass_generate`,
  `cicass_last_error`, `cicass_version`, `cicass_real_bytes`) so the makeCosICs
  realizer can be driven in-process / from a CodeBridge worker.
- **`makeCosICs/main.c`** — `main` made dual-purpose: `#ifdef CICASS_LIB` exposes
  `extern "C" int cicass_run_genics(...)` (library) else the standalone `main`
  (executable); plus the `-DOUTPUT_CAPI` output branch.
- **`deps/build_cicass_darwin.sh`** — builds `transfer.x` (GSL) and
  `libcicass_capi.dylib` (FFTW3 + GSL, no HDF5) via Homebrew on macOS.
- **`vbc_transfer/main.cc`** — one modern-clang fix (`(size_t)NUMEQNS` narrowing).

## Build

```
bash deps/build_cicass_darwin.sh        # → build/{transfer.x, libcicass_capi.dylib}
```

Needs Homebrew `gsl` + `fftw` and Apple clang `g++`. The Julia side
(`CICASS.jl/lib/CICASSLib`) calls `cicass_generate` and reads the `.cicass` dump;
`MultiCodeCICASSExt` injects the streaming ICs into live Enzo and RAMSES.

NB CICASS is a **post-recombination** model — valid for `zstart ≲ 1000`
(it aborts above recombination). Its value is enabling correct z ≈ 100–200 starts
with the baryon–dark-matter streaming velocity.
