#!/usr/bin/env bash
# Build CICASS for the EnzoNG wrapper (CICASSLib) on macOS.
#
#   bash deps/build_cicass_darwin.sh [CICASS_ROOT]
#
# Produces, under $CICASS_ROOT/build:
#   transfer.x              -- vbc_transfer TF-grid generator (GSL only)
#   libcicass_capi.dylib    -- makeCosICs realizer as a C-ABI library
#                              (-DCICASS_LIB -DOUTPUT_CAPI, HDF5-free raw dump)
#
# Dependencies (Homebrew): gsl, fftw. Apple clang g++.
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
GSL="$(brew --prefix gsl)"
FFTW="$(brew --prefix fftw)"
OUT="$ROOT/build"
mkdir -p "$OUT"

CXX="${CXX:-g++}"
COMMON="-O3 -Wno-deprecated-declarations -Wno-narrowing -DGSL_INTERP"
GSL_INC="-I$GSL/include -I$GSL/include/gsl"
FFTW_INC="-I$FFTW/include"
GSL_LIB="-L$GSL/lib -lgsl -lgslcblas"
FFTW_LIB="-L$FFTW/lib -lfftw3"

echo "== [1/2] vbc_transfer/transfer.x =="
( cd "$ROOT/vbc_transfer"
  $CXX $COMMON -DRECFAST -DPRINT_PK -DGSL $GSL_INC \
       -o "$OUT/transfer.x" main.cc -lm $GSL_LIB )

echo "== [2/2] makeCosICs -> libcicass_capi.dylib =="
( cd "$ROOT/makeCosICs"
  # Particle layout: DEFAULT = regular LATTICE (exact Lagrangian q = grid; enables
  # AHK phase-space-sheet diagnostics + a trivial/exact displacement field).  Set
  # CICASS_GLASS=1 to restore the pre-relaxed glass (GLASS_DM/GLASS_GAS/LARGE_SPACING).
  GLASSOPT="${CICASS_GLASS:+-DGLASS_DM -DGLASS_GAS -DLARGE_SPACING}"
  OPT="-DCICASS_LIB -DOUTPUT_CAPI $GLASSOPT -DGSL_INTERP"
  echo "   particle layout: ${CICASS_GLASS:+GLASS}${CICASS_GLASS:-LATTICE}"
  CF="-O3 -fPIC -Wno-deprecated-declarations -Wno-narrowing $OPT $GSL_INC $FFTW_INC"
  OBJ=""
  for f in main io read_glass getPk enzo_out capi_out; do
    $CXX $CF -c "$f.c" -o "$OUT/$f.lo"
    OBJ="$OBJ $OUT/$f.lo"
  done
  $CXX $CF -c cicass_capi.cc -o "$OUT/cicass_capi.lo"
  OBJ="$OBJ $OUT/cicass_capi.lo"
  $CXX -dynamiclib -o "$OUT/libcicass_capi.dylib" $OBJ \
       $GSL_LIB $FFTW_LIB -lm
  rm -f $OUT/*.lo )

echo "== done =="
ls -la "$OUT"/transfer.x "$OUT"/libcicass_capi.dylib
