/*********************************************************************
capi_out.c -- HDF5-free raw-binary writer for the EnzoNG CICASS wrapper.

Dumps exactly the arrays SaveEnzoSnapshot consumes, but as a flat,
self-describing little-endian binary file (no HDF5, no Gadget records),
so the Julia CICASSLib can read it and inject {DM particles + gas grid}
into a live Enzo or RAMSES without any libhdf5 <-> libenzo collision.

Layout (all little-endian; doubles unless noted):
  char   magic[8]   = "CICASS01"
  int32  SIZE                     # grid cells per axis (N); Ndm = N^3
  int32  NUMSPECIES               # 1 (DM only) or 2 (DM+gas)
  double hdr[10] = { BOXSIZE[Mpc/h], ZINIT, OmegaM, OmegaB, OmegaL,
                     hconst, Mpart_dm, Mpart_gas, VSTREAM[km/s @z=1000],
                     TAVG[K] }
  # --- DM particles (N^3), axis-contiguous ---
  double dm_pos_x[N^3], dm_pos_y[N^3], dm_pos_z[N^3]   # box fraction [0,1)
  double dm_vel_x[N^3], dm_vel_y[N^3], dm_vel_z[N^3]   # physical peculiar km/s
  # --- gas on the regular N^3 grid (CICASS c-order i + j*N + k*N*N) ---
  double gas_delta[N^3]                                # overdensity delta_b
  double gas_vel_x[N^3], gas_vel_y[N^3], gas_vel_z[N^3]# physical peculiar km/s
  double gas_temp[N^3]                                 # T = (temp+1)*TAVG

NB velocities are CICASS-native physical peculiar km/s (the same arrays
SaveEnzoSnapshot scales by 1/(VelocityUnits/1e5)); they are NOT in the
Gadget v/sqrt(a) convention -- the Julia injector applies each code's
convention. The gas-DM mean of gas_vel - dm_vel is the streaming velocity.
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>

extern double BOXSIZE, ZINIT, OmegaM, OmegaB, hconst, TAVG, VSTREAM;
extern int SIZE, NUMSPECIES;
extern double Mpart[];
double periodic_wrap(double x);

extern "C" int WriteCapiSnapshot(double **disdm, double **veldm, double *deltab,
                                 double **velb, double *temp, const char *path)
{
  long N = (long)SIZE * (long)SIZE * (long)SIZE;
  FILE *f = fopen(path, "wb");
  if (f == NULL) {
    fprintf(stderr, "#capi_out: could not open %s for writing\n", path);
    return -1;
  }

  const char magic[8] = {'C','I','C','A','S','S','0','1'};
  fwrite(magic, 1, 8, f);
  int hdr_i[2] = { SIZE, NUMSPECIES };
  fwrite(hdr_i, sizeof(int), 2, f);
  double hdr_d[10] = { BOXSIZE, ZINIT, OmegaM, OmegaB, 1.0 - OmegaM,
                       hconst, Mpart[0], Mpart[1], VSTREAM, TAVG };
  fwrite(hdr_d, sizeof(double), 10, f);

  /* DM particles: absolute positions (grid.dis[0] is Lagrangian + displacement
     in Mpc/h) wrapped to [0,BOXSIZE) then normalized to box fraction [0,1). */
  double *buf = (double *) malloc(sizeof(double) * N);
  if (buf == NULL) { fprintf(stderr, "#capi_out: malloc failed\n"); fclose(f); return -2; }
  for (int dim = 0; dim < 3; dim++) {
    for (long i = 0; i < N; i++)
      buf[i] = periodic_wrap(disdm[dim][i]) / BOXSIZE;
    fwrite(buf, sizeof(double), N, f);
  }
  free(buf);
  for (int dim = 0; dim < 3; dim++)
    fwrite(veldm[dim], sizeof(double), N, f);   /* physical peculiar km/s */

  /* Gas on the regular grid. */
  fwrite(deltab, sizeof(double), N, f);          /* overdensity delta_b */
  for (int dim = 0; dim < 3; dim++)
    fwrite(velb[dim], sizeof(double), N, f);      /* physical peculiar km/s */
  fwrite(temp, sizeof(double), N, f);             /* relative temperature */

  fclose(f);
  fprintf(stderr, "#capi_out: wrote %s (N=%d, Ndm=%ld)\n", path, SIZE, N);
  return 0;
}
