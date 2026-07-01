#!/usr/bin/env bash
# Build CICASS for the EnzoNG wrapper (CICASSLib) on Linux (Rocky8/RHEL8).
#
#   bash deps/build_cicass_linux.sh [CICASS_ROOT]
#
# Produces, under $CICASS_ROOT/build:
#   transfer.x              -- vbc_transfer TF-grid generator (GSL only)
#   libcicass_capi.so       -- makeCosICs realizer as a C-ABI library
#                              (-DCICASS_LIB -DOUTPUT_CAPI, HDF5-free raw dump)
#
# Dependencies: gsl, fftw3 (system packages), g++.
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="$ROOT/build"
mkdir -p "$OUT"

CXX="${CXX:-g++}"
COMMON="-O3 -Wno-deprecated-declarations -Wno-narrowing -DGSL_INTERP"
GSL_INC="-I/usr/include"
FFTW_INC="-I/usr/include"
GSL_LIB="-lgsl -lgslcblas"
FFTW_LIB="-lfftw3_threads -lfftw3 -lpthread"   # threaded FFTW: parallelize the SIZE^3 FFTs across cores

echo "== [1/2] vbc_transfer/transfer.x =="
( cd "$ROOT/vbc_transfer"
  $CXX $COMMON -DRECFAST -DPRINT_PK -DGSL $GSL_INC \
       -o "$OUT/transfer.x" main.cc -lm $GSL_LIB )

echo "== [2/2] makeCosICs -> libcicass_capi.so =="
( cd "$ROOT/makeCosICs"
  GLASSOPT="${CICASS_GLASS:+-DGLASS_DM -DGLASS_GAS -DLARGE_SPACING}"
  OPT="-DCICASS_LIB -DOUTPUT_CAPI $GLASSOPT -DGSL_INTERP -DOPENMP -fopenmp"   # -fopenmp: k-space loop + threaded FFTW
  echo "   particle layout: ${CICASS_GLASS:+GLASS}${CICASS_GLASS:-LATTICE}"
  CF="-O3 -fPIC -Wno-deprecated-declarations -Wno-narrowing $OPT $GSL_INC $FFTW_INC"
  OBJ=""
  for f in main io read_glass getPk enzo_out capi_out; do
    $CXX $CF -c "$f.c" -o "$OUT/$f.lo"
    OBJ="$OBJ $OUT/$f.lo"
  done
  $CXX $CF -c cicass_capi.cc -o "$OUT/cicass_capi.lo"
  OBJ="$OBJ $OUT/cicass_capi.lo"
  $CXX -shared -fPIC -fopenmp -o "$OUT/libcicass_capi.so" $OBJ \
       $GSL_LIB $FFTW_LIB -lm
  rm -f $OUT/*.lo )

echo "== done =="
ls -la "$OUT"/transfer.x "$OUT"/libcicass_capi.so
