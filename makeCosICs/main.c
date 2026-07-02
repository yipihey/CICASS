/***************************************************************************************
Generates proper cosmological intial conditions as discussed in O'Leary & McQuinn (2011; 
ApJ, accepted) and McQuinn & O'Leary.  
........................................................................................
See README for the details of the code.  There are many compile time flags that need
to be set depending on the application
***************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <unistd.h>
#include <dirent.h>
#include "gsl/gsl_rng.h"
#include <gsl/gsl_spline.h>
#include "init_cond.h"
#ifdef OPENMP
#include "omp.h"
#endif

#ifdef OUTPUT_CAPI
extern "C" int WriteCapiSnapshot(double **disdm, double **veldm, double *deltab,
                                 double **velb, double *temp, const char *path);
#endif



/**************************Globals********************************* 
These can be set at run time, but here many are set to fiducial values
*******************************************************************/
#ifdef TWO_ARRAY_SIZES  //CURRENTLY THIS NEEDS TO BE DEBUGGED
int DMSIZE = 512;
int BARSIZE = 256;
#endif
int SIZE = -1;              //number cells, set at input
double BOXSIZE = -1;        //in Mpc/h
int NUMSPECIES = 2;         //generally 2 species, baryons and dark matter.  

int SEEDRAND = 113334;      //random number seed for calculations
int FIXAMPL = 0;            //Angulo & Pontzen (2016) FIXED amplitude: |delta_k| = sqrt(P(k)) exactly
                            //(random phases only) -> realized P(k) matches the input mode-by-mode,
                            //killing box-to-box amplitude scatter.  0 = standard Rayleigh (default).
int FLIPPHASE = 0;          //phase inversion (delta_k -> -delta_k) for the PAIRED run; average the
                            //(FIXAMPL,FLIPPHASE=0) + (FIXAMPL,FLIPPHASE=1) pair to cancel leading
                            //non-Gaussian variance.  0 = no flip (default).
double ZINIT =200.;
double AINIT, TAVG, NEAVG;  //globals that do not need to be set


/**************Streaming velocity globals****************************/
double VSTREAM = 0.;
double ZSTREAM = 1000.;     //The redshift VSTREAM is valid at

/*************Ouput-related globals*********************************/
char OutputDir[200], BASEOUT[200], GlassFile[200];
int GlassTileFac;
int PRINT_INPUT_POWER = 1;   //prints the input power spectrum to file *BASEOUT*.pk

/******************Cosmological parameters*********************
Either set here or at execution
**************************************************************/
double hconst = 0.71; 
double OmegaM = 0.27;        //assumes flat universe
double OmegaB = 0.046;
double Mpart[2];            

//globals related to interpolation array
double DIM_ARRAY, NUMK_PAR_ARRAY;

#ifdef CICASS_LIB
extern "C" int cicass_run_genics(int argc, char *argv[])
#else
int main(int argc, char *argv[])
#endif
{

#if defined(OUTPUT_GADGET) && defined(OUTPUT_ENZO)
  fprintf(stdout, "Cannot output both for gadget and enzo\n");
  return -1;
#endif
  setParams(argc, argv);

#ifdef OPENMP
  /* Parallelize the SIZE^3 FFTs across cores (the r2c/c2r plans in doFFT/doInverseFFT
     pick up the thread count set here).  Thread count = OMP_NUM_THREADS. */
  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());
  fprintf(stdout, "#cicass: OpenMP + threaded FFTW, %d threads\n", omp_get_max_threads());
  fflush(stdout);
#endif

  int i, Nf =  (SIZE/2 +1)*SIZE*SIZE;
  struct dispGrid grid;
  double *K = NULL, *Pk = NULL;
  int *Count = NULL;
  char fname[200];
  void save_gadget_snap(double ***dis, double ***vel, double *delta, 
			double *temp, int Ndm, int Ngas, int count);

  AINIT = 1./(1.+ZINIT);

  if(PRINT_INPUT_POWER == 1)
    printInputPower();


  Mpart[0] = (OmegaM - OmegaB)*CRITDENSITYMPC*pow(BOXSIZE/SIZE, 3.)*CONVERTTOMASSUNITS_MSUNPH;  //in 10^10 Msun/h (standard for Gadget
  Mpart[1] = OmegaB*CRITDENSITYMPC*pow(BOXSIZE/SIZE, 3.)*CONVERTTOMASSUNITS_MSUNPH;

#ifdef TWO_ARRAY_SIZES
  Mpart[0] *= pow(SIZE/DMSIZE, 3.);
  Mpart[1] *= pow(SIZE/BARSIZE, 3.);  
#endif

  //does dark matter and baryons (w temperature) separately
  for(i=1; i<= NUMSPECIES; i++)
  {

#ifdef TWO_ARRAY_SIZES
    if(i == 1) SIZE = DMSIZE;
    else  SIZE = BARSIZE;

    Nf =  (SIZE/2 +1)*SIZE*SIZE;
    GlassTileFac = (int) SIZE/128;
#endif

    grid.dis[i-1] = (double **) malloc(sizeof(double*)*3);
    grid.v[i-1] = (double **)malloc(sizeof(double*)*3);
    fprintf(stdout, "Generating ICs for species %d\n", i);
    generateDisplacements(&grid, i);
    fprintf(stdout, "Done generating ICs for species %d\n", i);
    gridAndPrintPowerOfDisplacements(&grid, SIZE*SIZE*SIZE, Nf, i-1); 
#ifdef TWO_GADGET_OUTFILES
    //print each individually
    #if !defined(NO_TEMPERATURE) || !defined(NO_DELTA)
	if(i == 2)
	  gridDeltabAndTemperatureCIC(&grid, 0);
#endif
	
    save_gadget_snap(grid.dis, grid.v, grid.delta[1], grid.T, 
		     SIZE*SIZE*SIZE*(1- (i-1)), SIZE*SIZE*SIZE*(i-1), i-1);

    for(int j=0; j<3; j++)
      {
	fftw_free(grid.dis[i-1][j]); 
	fftw_free(grid.v[i-1][j]);
      }
#endif
  }

#if defined(TWO_GADGET_FILES) || defined(TWO_ARRAY_SIZES)
  return 1;
#endif


int cic_Td = 0;
#ifdef OUTPUT_ENZO
    SaveEnzoSnapshot(grid.dis[0], grid.v[0], grid.delta[1],  grid.v[1], 
		     grid.T);
#elif OUTPUT_GADGET

#if !defined(NO_TEMPERATURE) || !defined(NO_DELTA)
  gridDeltabAndTemperatureCIC(&grid, 0);
  //NOT CORRECT FOR NGP

  cic_Td = 1;
#endif
  save_gadget_snap(grid.dis, grid.v, grid.delta[1], grid.T,
		   SIZE*SIZE*SIZE, SIZE*SIZE*SIZE, -1);
#endif

#ifdef OUTPUT_CAPI
  /* EnzoNG wrapper: HDF5-free raw dump of {DM particles + gas grid}. */
  {
#if !defined(NO_TEMPERATURE) || !defined(NO_DELTA)
    /* CICASS_SMOOTH_BARYON: write the RAW smooth Fourier-realized baryon density grid
       (grid->delta[1] = IFFT of T_b(k)*noise, set in generateDisplacements) WITHOUT the
       CIC interpolation to displaced-glass positions.  This is the actual CAMB/CLASS
       baryon density field on the regular Eulerian grid, free of particle shot noise. */
    if (getenv("CICASS_SMOOTH_BARYON") != NULL) {
      fprintf(stderr, "#capi: SMOOTH baryon grid (raw Fourier delta_b, no glass/CIC)\n");
    } else {
      gridDeltabAndTemperatureCIC(&grid, 0);
      cic_Td = 1;
    }
#endif
    char capifn[300];
    sprintf(capifn, "%s/%s.cicass", OutputDir, BASEOUT);
    WriteCapiSnapshot(grid.dis[0], grid.v[0], grid.delta[1], grid.v[1],
                      grid.T, capifn);
  }
#endif


  /******************************************************************
Prints power spectrum of gridded density for component=X.
Currently it prints the power spectrum of the baryons in the output directory
Can generalize this to print power spectrum of any quantity that is desired.
This is very helpful for debugging
  **************************************************************/
#ifndef NO_DELTA
  if(cic_Td == 1)
      gridDeltabAndTemperatureCIC(&grid, 1); //projects back onto grid
  
  int component = 1; //currenty baryons
  fftw_complex *df= (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nf);
  initialize(&K, &Pk, &Count, 0);
  doFFT(grid.delta[component], df, Nf);
  makePk(K, Pk, Count, df, df);
  sprintf(fname, "%s/deltabPk_%s_%.0lf.pk", OutputDir, BASEOUT, ZINIT);
  printStatistics(K, Pk, fname);
  fftw_free(df);
  initialize(&K, &Pk, &Count, 2);
#endif


  fprintf(stderr,"#success main\n");
  return 0;
}



/*************************************************************************
Generates displacment field for particle ICs
//flag == 1 is dark matter, 2 is baryons
**************************************************************************/
void generateDisplacements(struct dispGrid *grid, int flag)
{
  double *dphi[3], *dphidot[3];
  long N, Nf;

  int s2, i;
  fftw_complex *phif, *dphif[3], *Tf = NULL, *deltaf = NULL;    
  fftw_complex *phifdot, *dphifdot[3];

  // initGlobals(argc, argv);

  s2 = SIZE*SIZE;
  //size of arrays
  N = SIZE*s2;
  Nf = (SIZE/2 +1)*s2;

  phif = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nf);
  phifdot = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nf);

#ifndef NO_TEMPERATURE 
  if(flag == 2)
    Tf = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * Nf); //will eventually use this
#endif
#ifndef NO_DELTA 
  if(flag == 2)
    deltaf = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * Nf); //will eventually use this
#endif

  fprintf(stdout, "Generating Gaussian Field.\n");
  addGaussianField(phif, phifdot, deltaf, Tf, flag); //creates kspace linear density field
  fprintf(stdout, "Done!\n");
 
  if(flag == 2)//do temperature
    {
#ifndef NO_TEMPERATURE  
      grid->T = (double *) fftw_malloc(sizeof(double) * N); //will eventually use this
      doInverseFFT(Tf, grid->T, N);
      fftw_free(Tf);
#endif
#ifndef NO_DELTA
      grid->delta[1] = (double *) fftw_malloc(sizeof(double) * N); //will eventually use this
      doInverseFFT(deltaf, grid->delta[1], N);
      fftw_free(deltaf);
#endif
    }


  for(i=0;i<3;i++)
    {
      dphif[i]= (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nf);
      dphifdot[i]= (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nf);
    }
  
  fprintf(stdout, "Setting dphif and dphiftot\n");
  setdphi(dphif, phif);  //calculates dphi
  setdphi(dphifdot, phifdot);  //calculates dphidot
  fftw_free(phif);  fftw_free(phifdot);
  fprintf(stdout, "Done!\n");


  fprintf(stdout, "Generating dphi\n");
  for(i=0;i<3;i++)
    {
      dphi[i] = (double *) fftw_malloc(sizeof(double) * N);
      doInverseFFT(dphif[i], dphi[i], N); //gets real space dphi
      dphidot[i] = (double *) fftw_malloc(sizeof(double) * N);
      doInverseFFT(dphifdot[i], dphidot[i], N); //gets real space dphi
      fftw_free(dphif[i]);   fftw_free(dphifdot[i]);
    }
  fprintf(stdout, "Done!\n"); 

  for(i=0;i<3;i++)
    {
      grid->v[flag-1][i] = (double *) fftw_malloc(sizeof(double) * N);
      grid->dis[flag-1][i] = (double *) fftw_malloc(sizeof(double) * N);
    }

  calculateDispField(grid, dphi, dphidot, flag-1);

  //probably do not need to allocate so many arrays at the same time
  for(i=0;i<3;i++)
    {
      fftw_free(dphi[i]);
      fftw_free(dphidot[i]);
    }
}


/*******************************************************************
generates displacements in k-space
*******************************************************************/
void setdphi(fftw_complex **dphif, fftw_complex *phif)
{
  int i, j, k, l, ii;
  double K[3], dk = 2.*M_PI/BOXSIZE;;

    for(i =  0; i< SIZE; i++) 
      for(j =  0; j< SIZE; j++) 
	for(k =  0; k< SIZE/2 + 1; k++) 
	{ 
	  l = (i*SIZE + j)*(SIZE/2 +1) + k; 
	  K[0]= dk*knum(i);
	  K[1]= dk*knum(j);
	  K[2]= dk*k;

	  for(ii = 0; ii< 3;ii++)
	    {
	      dphif[ii][l][0] = K[ii]*phif[l][1]; //real part
	      dphif[ii][l][1] = -K[ii]*phif[l][0]; //im part
	    } 
	}
}


void calculateDispField(struct dispGrid *grid, double **dphi, double **dphidot, int species)
{
  int num = 0, numdm=0;
  int i, j, k, l, s, n;
  // double H = getHz(ZINIT)/MPCTOKM;
  // double f = 1.; //need to change this
  s = species;
  void read_glass(double ***dis, int species);
  double fac = MPCTOKM/hconst*AINIT;

#ifdef GLASS_DM 
  numdm=1;
  if(species == 0)
    num = 1;
#endif
#ifdef GLASS_GAS
  if(species == 1)
    num = 1;
#endif

  if(num == 1) // initial positions are glass file
    {
      fprintf(stdout, "Reading glass file for species %d\n", species);
      read_glass(grid->dis, species);
      fprintf(stdout, "Gridding\n");
      gridWithGlass(grid, dphi, dphidot, species, fac, SIZE*SIZE*SIZE);
      fprintf(stdout, "done\n");
    }else  //inial positions are a grid
    {
      for(i =  0; i< SIZE; i++) 
	for(j =  0; j< SIZE; j++) 
	  for(k =  0; k< SIZE; k++) 
	    { 
	      l = i*SIZE*SIZE + j*SIZE + k; 

	      grid->dis[s][0][l] = (BOXSIZE*((double) i)/SIZE) - dphi[0][l];
	      grid->dis[s][1][l] = (BOXSIZE*((double) j)/SIZE) - dphi[1][l];
	      grid->dis[s][2][l] = (BOXSIZE*((double) k)/SIZE) - dphi[2][l];

	      //shifts gas particles by 0.5 mean interperaticle spacings each dimension to prohibit coupling
	      //no need to do this if dark matter is from glass fiel
	      if(s==1 && numdm == 0)
		{
		  grid->dis[s][0][l] += .5*(BOXSIZE/SIZE);
		  grid->dis[s][1][l] += .5*(BOXSIZE/SIZE);
		  grid->dis[s][2][l] += .5*(BOXSIZE/SIZE);
		}

	      //  printf("%d %d %d %d %le %le %le\n",s, i, j, k, grid->dis[s][0][l], grid->dis[s][1][l],grid->dis[s][2][l]);
	      grid->v[s][0][l] =  -dphidot[0][l]*fac;
	      grid->v[s][1][l] =  -dphidot[1][l]*fac;
	      grid->v[s][2][l] =  -dphidot[2][l]*fac;

#ifdef BOOST_EVERYTHING_BYVKMSEC
	      grid->v[s][0][l] += (double) BOOST_EVERYTHING_BYVKMSEC;
#endif

	      if(s==1)
		{
		  grid->v[s][0][l] += VSTREAM*(1+ZINIT)/(1+ZSTREAM);
		}

	      for(n=0;n<3;n++)
		grid->dis[s][n][l] =  periodic_wrap(grid->dis[s][n][l]); 
	    }
    }
 
}

void gridWithGlass(struct dispGrid *grid, double **dphi, double **dphidot, int s, double fac, int Np)
{
  int n, i, j, k, ii, jj, kk, axes;
  double u, v, w, dis, dis2;
  double f1,f2,f3,f4,f5,f6,f7,f8;

  //#pragma omp parallel for private(n, i, j, k, ii, jj, kk, axes, u, v, w, dis, dis2, f1,f2,f3,f4,f5,f6,f7,f8)
  for(n = 0; n < Np; n++)
    {
      u = grid->dis[s][0][n] / BOXSIZE * SIZE;
      v = grid->dis[s][1][n] / BOXSIZE * SIZE;
      w = grid->dis[s][2][n] / BOXSIZE * SIZE;
	      
      i = (int) u;
      j = (int) v;
      k = (int) w;
	      
      if(i == SIZE)
	i = SIZE - 1;
      if(j == SIZE)
	j = SIZE - 1;
      if(k == SIZE)
	k = SIZE - 1;
	      
      u -= i;
      v -= j;
      w -= k;
	      
      ii = i + 1;
      jj = j + 1;
      kk = k + 1;
	      
      if(ii >= SIZE)
	ii -= SIZE;
      if(jj >= SIZE)
	jj -= SIZE;
      if(kk >= SIZE)
	kk -= SIZE;
	      
      f1 = (1 - u) * (1 - v) * (1 - w);
      f2 = (1 - u) * (1 - v) * (w);
      f3 = (1 - u) * (v) * (1 - w);
      f4 = (1 - u) * (v) * (w);
      f5 = (u) * (1 - v) * (1 - w);
      f6 = (u) * (1 - v) * (w); 
      f7 = (u) * (v) * (1 - w);
      f8 = (u) * (v) * (w);

#ifdef NGP
      f1 = 1; f2 = f3 = f4 = f5 = f6 = f7 =f8 = 0;
#endif
	     
      for(axes = 0; axes < 3; axes++)
	{
		 
	  dis = dphi[axes][(i * SIZE + j) * SIZE + k] * f1 +
	    dphi[axes][(i * SIZE + j) * SIZE + kk] * f2 +
	    dphi[axes][(i * SIZE + jj) * SIZE + k] * f3 +
	    dphi[axes][(i * SIZE + jj) * SIZE + kk] * f4 +
	    dphi[axes][(ii * SIZE + j) * SIZE + k] * f5 +
	    dphi[axes][(ii * SIZE + j) * SIZE + kk] * f6 +
	    dphi[axes][(ii * SIZE + jj) * SIZE + k] * f7 +
	    dphi[axes][(ii * SIZE + jj) * SIZE + kk] * f8;

	  dis2 = dphidot[axes][(i * SIZE + j) * SIZE + k] * f1 +
	    dphidot[axes][(i * SIZE + j) * SIZE + kk] * f2 +
	    dphidot[axes][(i * SIZE + jj) * SIZE + k] * f3 +
	    dphidot[axes][(i * SIZE + jj) * SIZE + kk] * f4 +
	    dphidot[axes][(ii * SIZE + j) * SIZE + k] * f5 +
	    dphidot[axes][(ii * SIZE + j) * SIZE + kk] * f6 +
	    dphidot[axes][(ii * SIZE + jj) * SIZE + k] * f7 +
	    dphidot[axes][(ii * SIZE + jj) * SIZE + kk] * f8;
		  		
	  grid->dis[s][axes][n] -= dis;
	  grid->v[s][axes][n] = -dis2*fac;

	  grid->dis[s][axes][n] = periodic_wrap(grid->dis[s][axes][n]);

	}
      if(s==1)
	{
	  grid->v[s][0][n] += VSTREAM*(1+ZINIT)/(1+ZSTREAM);
	}
    }
}



/******************************************************************************
Puts down Gaussian Field
flag == 1 DM
flag == 2 Baryons
******************************************************************************/
void addGaussianField(fftw_complex *phif, fftw_complex *phidotf, fftw_complex *delta, fftw_complex *temp, int flag)
{
  int i, j,  k, l, glassdm=0, glassgas=0;
    double K, Ksq, costh;     
    double dk = 2.*M_PI/BOXSIZE;
    double Volume = BOXSIZE*BOXSIZE*BOXSIZE;
    double sqrtVol = sqrt(Volume);
    double g1, g2; // sq2 = sqrt(2);
    double getPk(double k, int flag);
 
    double smth = 1.;
    double ampl, phase;
    double fx, fy, fz, ff, K0, K1, K2;


    gsl_rng *random_generator;
    random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);
    gsl_rng_set(random_generator, SEEDRAND);
    unsigned int *seedtable= (unsigned int*) malloc(sizeof(unsigned int)*SIZE*SIZE);
    void genSeedTable(unsigned int seedtable[], gsl_rng *random_generator, int Nmesh);
#ifdef GLASSDM
    glassdm=1.;
#endif
#ifdef GLASSGAS
    glassgas=1.;
#endif

    genSeedTable(seedtable, random_generator, SIZE);
   

#pragma omp parallel for private(K, l, ampl, phase, i, j, k,  g1, g2, Ksq, costh, K0, K1, K2,fx, fy,fz,ff, smth)
    for(i =  0; i< SIZE; i++) 
      {
#ifdef GSL_INTERP
	gsl_interp_accel **acc[20];
	for(int n=0; n< DIM_ARRAY; n++)
	  {    
	    acc[n] = (gsl_interp_accel **) malloc(sizeof(gsl_interp_accel *)*NUMK_PAR_ARRAY);
	    for(int nn=0; nn< NUMK_PAR_ARRAY;nn++)
	      acc[n][nn]= gsl_interp_accel_alloc();
	  }
#endif

	double *G = (double *) calloc(20, sizeof(double));
	gsl_rng *random_generator2;
	random_generator2 = gsl_rng_alloc(gsl_rng_ranlxd1);

      for(j =  0; j< SIZE; j++) 
	{
	  gsl_rng_set(random_generator2, seedtable[i * SIZE + j]); 
  
	  for(k =  0; k< SIZE/2 + 1; k++) 
	    { 

	      K= dk*sqrt(knum(i)*knum(i) +knum(j)*knum(j) + k*k); 
	      l = (i*SIZE + j)*(SIZE/2 +1) + k; 

	      {
	      // Always draw the uniform (keeps the RNG stream aligned so a FIXAMPL run shares phases
	      // with a standard run of the same seed -> directly comparable + clean pairing).  Angulo &
	      // Pontzen fixing: since <ampl^2>=1 for the Rayleigh draw here, the FIXED amplitude is 1.
	      double uamp;
	      do
		uamp = gsl_rng_uniform(random_generator2);
	      while(!FIXAMPL && sqrt(-log(uamp)) == 0);
	      ampl = FIXAMPL ? 1.0 : sqrt(-log(uamp));
	      phase = gsl_rng_uniform(random_generator2)*2*M_PI;
	      if(FLIPPHASE) phase += M_PI;   // paired run: delta_k -> -delta_k
	      }
	      g1= ampl*cos(phase);
	      g2 = ampl*sin(phase);
	    
	      Ksq = K*K;
	      if(i!=0 || j!=0 || k!=0)//don't do anything for zero mode
		{
		  costh = dk*fabs(knum(i))/K; //that this is positive doesn't matter because of random phase
#ifdef GSL_INTERP
		  interpPk(K, costh, G, acc, 1);
#else
		  interpPk(K, costh, G, NULL, 1);
#endif
		}
	      else
		{
		  g1 = 0.; g2 = 0.; //so things are zero
		  Ksq = 1.e30;//large number
		}


	      if( (glassdm & flag == 1) || (glassgas & flag != 1)) //correct for CIC gridding if using glass file
		{
		  /* calculate smooth factor for deconvolution of CIC interpolation */
		  K0 = dk*knum(i);
		  K1 = dk*knum(j);
		  K2 = dk*k;
		  fx = fy = fz = 1;
		  if(K0 != 0)
		    {
		      fx = (K0 * BOXSIZE / 2) / SIZE;
		      fx = sin(fx) / fx;
		    }
		  if(K1 != 0)
		    {
		      fy = (K1 * BOXSIZE / 2) / SIZE;
		      fy = sin(fy) / fy;
		    }
		  if(K2 != 0)
		    {
		      fz = (K2 * BOXSIZE / 2) / SIZE;
		      fz = sin(fz) / fz;
		    }
		  ff = 1 / (fx * fy * fz);
		  smth = ff * ff;
#ifdef NGP
		  smth = ff;
#endif
		  g1*=smth; g2*=smth;
		}
	      else
		{
		  smth = 1;
		}  

	      

	      
	      if(flag == 1)//dark matter
		{
		  phif[l][0] = G[1]*cos(G[2])*g1*sqrtVol/Ksq - G[1]*sin(G[2])*g2*sqrtVol/Ksq;
		  phif[l][1] = G[1]*sin(G[2])*g1*sqrtVol/Ksq + G[1]*cos(G[2])*g2*sqrtVol/Ksq;
	    
		  phidotf[l][0] = -(G[3]*cos(G[4])*g1*sqrtVol/Ksq - G[3]*sin(G[4])*g2*sqrtVol/Ksq);
		  phidotf[l][1] = -(G[3]*sin(G[4])*g1*sqrtVol/Ksq + G[3]*cos(G[4])*g2*sqrtVol/Ksq);

		}else //baryons + temp
		{
		  phif[l][0] = G[5]*cos(G[6])*g1*sqrtVol/Ksq - G[5]*sin(G[6])*g2*sqrtVol/Ksq;
		  phif[l][1] = G[5]*sin(G[6])*g1*sqrtVol/Ksq + G[5]*cos(G[6])*g2*sqrtVol/Ksq;

		  phidotf[l][0] = -(G[7]*cos(G[8])*g1*sqrtVol/Ksq - G[7]*sin(G[8])*g2*sqrtVol/Ksq);
		  phidotf[l][1] = -(G[7]*sin(G[8])*g1*sqrtVol/Ksq + G[7]*cos(G[8])*g2*sqrtVol/Ksq);

#ifndef NO_DELTA
		  delta[l][0] = G[5]*cos(G[6])*g1*sqrtVol - G[5]*sin(G[6])*g2*sqrtVol;
		  delta[l][1] = G[5]*sin(G[6])*g1*sqrtVol + G[5]*cos(G[6])*g2*sqrtVol;

#ifndef OUTPUT_GADGET //don't need this correction unless gridding onto particles
		  delta[l][0] /= smth; delta[l][1] /= smth; //do not CIC temperature and density if they are part of a grid
#endif
#endif

#ifndef NO_TEMPERATURE 
		  temp[l][0] = G[11]*cos(G[12])*g1*sqrtVol - G[11]*sin(G[12])*g2*sqrtVol;
		  temp[l][1] = G[11]*sin(G[12])*g1*sqrtVol + G[11]*cos(G[12])*g2*sqrtVol;
		  
		  // printf("Temp =%le\n", temp[l][0]);


#ifndef OUTPUT_GADGET
		  temp[l][0] /= smth; temp[l][1] /= smth; //do not CIC temperatur and density to particles
#endif
#endif
		}
	    }
	}
	
      free(G);
      gsl_rng_free(random_generator2);

#ifdef GSL_INTERP
	  for(int n=0; n< DIM_ARRAY; n++)    
	    for(int nn=0; nn< NUMK_PAR_ARRAY;nn++)
	      gsl_interp_accel_free(acc[n][nn]);
#endif
      }
    free(seedtable);

}


/*************************************genSeedTable***************************************
Taken from Volker Springel's n-genic code (http://www.gadgetcode.org/right.html#ICcode)
*****************************************************************************************
Initializes random numbers used to set density perturbations.
It nicely maintains the same long wavelength modes at the same seed for different mesh sizes,
which allows nice resolution tests.
****************************************************************************************/
void genSeedTable(unsigned int seedtable[], gsl_rng *random_generator, int Nmesh)
{
  int i, j;

  for(i = 0; i < Nmesh / 2; i++)
    {
      for(j = 0; j < i; j++)
	seedtable[i * Nmesh + j] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[j * Nmesh + i] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i; j++)
	seedtable[(Nmesh - 1 - i) * Nmesh + j] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[(Nmesh - 1 - j) * Nmesh + i] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i; j++)
	seedtable[i * Nmesh + (Nmesh - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[j * Nmesh + (Nmesh - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i; j++)
	seedtable[(Nmesh - 1 - i) * Nmesh + (Nmesh - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_generator);

      for(j = 0; j < i + 1; j++)
	seedtable[(Nmesh - 1 - j) * Nmesh + (Nmesh - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_generator);
    }
}



/*********************************************************************
May be able to consolidate code as this is somewhat redunant with routine below 
 ********************************************************************/
void gridDeltabAndTemperatureCIC(dispGrid *grid, int flag)
{
  const int flm = 1; //only baryons
  int n, N = SIZE*SIZE*SIZE;
  int i, j, k, ii, jj, kk;
  double f1,f2,f3,f4,f5,f6,f7,f8, u,v,w;
  double *d = (double *) calloc(N, sizeof(double));
  double *T = (double *) calloc(N, sizeof(double));

  double *Temp = grid->T;
  double *delta = grid->delta[flm];

  //make CIC grid at particle postions
  for(n = 0; n < N; n++)
    {
      u = grid->dis[flm][0][n] / BOXSIZE * SIZE;
      v = grid->dis[flm][1][n] / BOXSIZE * SIZE;
      w = grid->dis[flm][2][n] / BOXSIZE * SIZE;
	      
      i = (int) u;
      j = (int) v;
      k = (int) w;


  if(i >= SIZE || j >= SIZE || k >= SIZE)
    {
      fprintf(stderr, "Position off grid\n");
      exit(-5);
    }
  
      u -= i;
      v -= j;
      w -= k;
	      
      ii = i + 1;
      jj = j + 1;
      kk = k + 1;
	      
      if(ii >= SIZE)
	ii -= SIZE;
      if(jj >= SIZE)
	jj -= SIZE;
      if(kk >= SIZE)
	kk -= SIZE;
	      
      f1 = (1 - u) * (1 - v) * (1 - w);
      f2 = (1 - u) * (1 - v) * (w);
      f3 = (1 - u) * (v) * (1 - w);
      f4 = (1 - u) * (v) * (w);
      f5 = (u) * (1 - v) * (1 - w);
      f6 = (u) * (1 - v) * (w); 
      f7 = (u) * (v) * (1 - w);
      f8 = (u) * (v) * (w);
	     
#ifdef NGP
      f1 = 1.; f2 = f3 = f4 = f5 = f6 = f7 =f8 = 0;
#endif

      if(flag == 0) //goes from grid to particle positions 
	{
#ifndef NO_DELTA
	  d[n] = delta[(i * SIZE + j) * SIZE + k] * f1 +
	    delta[(i * SIZE + j) * SIZE + kk] * f2 +
	    delta[(i * SIZE + jj) * SIZE + k] * f3 +
	    delta[(i * SIZE + jj) * SIZE + kk] * f4 +
	    delta[(ii * SIZE + j) * SIZE + k] * f5 +
	    delta[(ii * SIZE + j) * SIZE + kk] * f6 +
	    delta[(ii * SIZE + jj) * SIZE + k] * f7 +
	    delta[(ii * SIZE + jj) * SIZE + kk] * f8;
#endif
#ifndef NO_TEMPERATURE
	  T[n] = Temp[(i * SIZE + j) * SIZE + k] * f1 +
	    Temp[(i * SIZE + j) * SIZE + kk] * f2 +
	    Temp[(i * SIZE + jj) * SIZE + k] * f3 +
	    Temp[(i * SIZE + jj) * SIZE + kk] * f4 +
	    Temp[(ii * SIZE + j) * SIZE + k] * f5 +
	    Temp[(ii * SIZE + j) * SIZE + kk] * f6 +
	    Temp[(ii * SIZE + jj) * SIZE + k] * f7 +
	    Temp[(ii * SIZE + jj) * SIZE + kk] * f8;
#endif

	}else{  //go from particle positions to grid

#ifndef NO_DELTA
	d[(i * SIZE + j) * SIZE + k] += delta[n]*f1;
	d[(i * SIZE + j) * SIZE + kk] += delta[n]*f2;
	d[(i * SIZE + jj) * SIZE + k] += delta[n]*f3;
	d[(i * SIZE + jj) * SIZE + kk] += delta[n]*f4;
	d[(ii * SIZE + j) * SIZE + k] += delta[n]*f5;
	d[(ii * SIZE + j) * SIZE + kk]  += delta[n]*f6;
	d[(ii * SIZE + jj) * SIZE + k]  += delta[n]*f7;
	d[(ii * SIZE + jj) * SIZE + kk] += delta[n]*f8;
#endif

#ifndef NO_TEMPERATURE
	T[(i * SIZE + j) * SIZE + k] += Temp[n]*f1;
	T[(i * SIZE + j) * SIZE + kk] += Temp[n]*f2;
	T[(i * SIZE + jj) * SIZE + k] += Temp[n]*f3;
	T[(i * SIZE + jj) * SIZE + kk] += Temp[n]*f4;
	T[(ii * SIZE + j) * SIZE + k] += Temp[n]*f5;
	T[(ii * SIZE + j) * SIZE + kk]  += Temp[n]*f6;
	T[(ii * SIZE + jj) * SIZE + k]  += Temp[n]*f7;
	T[(ii * SIZE + jj) * SIZE + kk] += Temp[n]*f8;
#endif 

      }
    }
  //copy back
  for(n = 0; n < SIZE*SIZE*SIZE; n++)
    {
      Temp[n] = T[n];  delta[n] = d[n];
    }

  free(T); free(d);//free
}


/**************************************************************************
puts vector val[3] at position d0, d1, d2 onto g[3][] using CIC kernel
 ************************************************************************/
void gridCIC(double d0, double d1, double d2, double val[], double **g, int Naxes)
{
  int axes, i, j, k, ii, jj, kk;
  double f1,f2,f3,f4,f5,f6,f7,f8, u,v,w;

  u = d0 / BOXSIZE * SIZE;
  v = d1 / BOXSIZE * SIZE;
  w = d2 / BOXSIZE * SIZE;
	      
  i = (int) u;
  j = (int) v;
  k = (int) w;

  if(i >= SIZE || j >= SIZE || k >= SIZE)
    {
      fprintf(stderr, "Position off grid\n");
      exit(-5);
    }
	      
  u -= i;
  v -= j;
  w -= k;
	      
  ii = i + 1;
  jj = j + 1;
  kk = k + 1;
	      
  if(ii >= SIZE)
    ii -= SIZE;
  if(jj >= SIZE)
    jj -= SIZE;
  if(kk >= SIZE)
    kk -= SIZE;
	      
  f1 = (1 - u) * (1 - v) * (1 - w);
  f2 = (1 - u) * (1 - v) * (w);
  f3 = (1 - u) * (v) * (1 - w);
  f4 = (1 - u) * (v) * (w);
  f5 = (u) * (1 - v) * (1 - w);
  f6 = (u) * (1 - v) * (w); 
  f7 = (u) * (v) * (1 - w);
  f8 = (u) * (v) * (w);

#ifdef NGP  //never really used currently because I have redundant code
      f1 = 1.; f2 = f3 = f4 = f5 = f6 = f7 =f8 = 0;
#endif
	     
  for(axes = 0; axes < Naxes; axes++)
    {
      g[axes][(i * SIZE + j) * SIZE + k] += val[axes]*f1;
      g[axes][(i * SIZE + j) * SIZE + kk] += val[axes]*f2;
      g[axes][(i * SIZE + jj) * SIZE + k] += val[axes]*f3;
      g[axes][(i * SIZE + jj) * SIZE + kk] += val[axes]*f4;
      g[axes][(ii * SIZE + j) * SIZE + k] += val[axes]*f5;
      g[axes][(ii * SIZE + j) * SIZE + kk]  += val[axes]*f6;
      g[axes][(ii * SIZE + jj) * SIZE + k]  += val[axes]*f7;
      g[axes][(ii * SIZE + jj) * SIZE + kk] += val[axes]*f8;
    }

  //  fprintf(stdout, "sum f = %le\n", f1+f2+f3+f4+f5+f6+f7+f8);
}


/**************************doFFT***********************************
Forward fft using FFTW3 library.  Uses somewhat strange Fourier convention
******************************************************************/
void doFFT(double *R, fftw_complex *Rf, long Nf)
{
    int i;
    double dvol = pow(BOXSIZE/SIZE, 3);
    fftw_plan p;

    p  = fftw_plan_dft_r2c_3d(SIZE, SIZE, SIZE, R, Rf, FFTW_ESTIMATE);
    
    fftw_execute(p);

    for(i = 0; i<Nf; i++)
    {
	Rf[i][0] *= dvol;  //d^3x differential
	Rf[i][1] *= dvol;
     }
    fftw_destroy_plan(p);
}

/**************************doFFT***********************************
Inverse fft using FFTW3 library.  Uses somewhat strange Fourier convention
******************************************************************/
void doInverseFFT(fftw_complex *Rf, double *R, long N)
{
    int i;
    fftw_plan p;
    double dvol = pow(BOXSIZE, 3.);

    p  = fftw_plan_dft_c2r_3d(SIZE, SIZE, SIZE, Rf, R, FFTW_ESTIMATE);
   
    fftw_execute(p);

   for(i = 0; i<N; i++)
    {
	R[i] /= dvol;
    }
    fftw_destroy_plan(p);
}

/************************periodic_wrap*******************************************
Enforces Periodic boundary conditions
**********************************************************************************/
double periodic_wrap(double x)
{
  while(x >= BOXSIZE)
    x -= BOXSIZE;

  while(x < 0)
    x += BOXSIZE;

  return x;
}


void setParams(int argc, char *argv[])
{
  int glass=0, glassfilesize=128;

  sprintf(OutputDir, "./");  /* default values */
  sprintf(BASEOUT, "test");

  //  GlassFile = malloc(sizeof(char)*200);
  sprintf(GlassFile, "../glass/glass_128_usethis");

  /****************Read in parameters************************/
  while ((argc>1) && (argv[1][0]=='-')) {
    switch (argv[1][1]) {
	case 'H':
	    hconst = atof(&argv[1][2]);
	    break;
	case 'O':
	    OmegaM = atof(&argv[1][2]);
	    break;
	case 'B':
	    OmegaB = atof(&argv[1][2]);
	    break;
	case 'L':
	    BOXSIZE = atof(&argv[1][2]);
	    break;
	case 'Z':
	    ZINIT = atof(&argv[1][2]);
	    break;
	case 'V':
	    VSTREAM = atof(&argv[1][2]);
	    break;
	case 'N':
	    SIZE = atoi(&argv[1][2]);
	    break;
	case 'R':
	    SEEDRAND = atoi(&argv[1][2]);
	    break;
	case 'F':
	    FIXAMPL = atoi(&argv[1][2]);   // Angulo & Pontzen fixed amplitude (1 = on)
	    break;
	case 'I':
	    FLIPPHASE = atoi(&argv[1][2]); // paired-run phase inversion (1 = on)
	    break;
	case 'P':
	    PRINT_INPUT_POWER = atoi(&argv[1][2]);
	    break;
	case 'S':
	    NUMSPECIES = atoi(&argv[1][2]);
	    if(NUMSPECIES == 0 || NUMSPECIES > 2){
	      fprintf(stderr, "Value not allowed %s\n", argv[1]);
	      exit(-1);
	    }
	    break;
         case 'o':
	   sprintf(OutputDir, &(argv[1][2]));
	   break;
         case 'b':
	   sprintf(BASEOUT, &(argv[1][2]));
	   break;
	case 'G':
	    glassfilesize = atoi(&argv[1][2]);
	    glass++;
	    break;
         case 'g':
	   sprintf(GlassFile, &(argv[1][2]));
	   glass++;
	   break;
    default:
      fprintf(stderr, "Bad option %s\n", argv[1]);
    }
    --argc;
    ++argv;
  }
  fprintf(stdout, "Cosmology set to h=%.3lf, OmegaM=%.3lf, OmegaB=%.3lf\n", hconst, OmegaM, OmegaB);

  if(BOXSIZE < 0){
    fprintf(stderr, "Boxsize is not set:  use -B option\n");
    exit(-1);
  }
  if(SIZE < 0){
    fprintf(stderr, "Gridsize is not set:  use -N option\n");
    exit(-1);
  }


  
  fprintf(stdout, "Run Specs:  output redshift =%.3lf, BoxSize =%.3le cMpc/h, GridSize=%d\n", ZINIT, BOXSIZE, SIZE);
  fprintf(stdout, "Filename conventions:  output directory = %s, Base file name = %s\n", OutputDir, BASEOUT);

  /*********************************Glass file input options*********************************************/
#if defined(GLASS_GAS) || defined(GLASS_DM)
  if(glass<=1){
    fprintf(stderr, "Either glass filename is not provided and/or the glass file size. Use -g and -G options.  Defaults to:\n");
  }
  fprintf(stderr, "Glass file = %s with dimension = %d\n", GlassFile, glassfilesize);
  
  if(SIZE%glassfilesize !=0){
    fprintf(stderr, "Glass file size (%d) is not an integer divisor of grid size (%d) \n", SIZE, glassfilesize);
    exit(-1);
  }
  GlassTileFac = (int) SIZE/glassfilesize;
#endif
  printf("\n\n");
}


/*****************************************************************************************
Outputs initial power spectrum into *BASEFILENAME*.pk
 ****************************************************************************************/
void printInputPower()
{
  double k, G[100], fac;
  char filename[200];
  sprintf(filename, "%s.pk", BASEOUT);
  FILE *outfile = fopen(filename, "w");
  fprintf(outfile, "#k Deltak_baryons Deltak_dm Deltak_velocitybaryons Deltak_velocity_dm\n");
  interpPk(0, 0, NULL, NULL, 0);//initialize power spectrum

#ifdef GSL_INTERP
	gsl_interp_accel **acc[20];
	for(int n=0; n< DIM_ARRAY; n++)
	  {    
	    acc[n] = (gsl_interp_accel **) malloc(sizeof(gsl_interp_accel *)*NUMK_PAR_ARRAY);
	    for(int nn=0; nn< NUMK_PAR_ARRAY;nn++)
	      acc[n][nn]= gsl_interp_accel_alloc();
	  }
#endif

  for(k=2.*M_PI/BOXSIZE;k< 2*M_PI/BOXSIZE*SIZE;k+=2.*M_PI/BOXSIZE)
    {
#ifdef GSL_INTERP
      interpPk(k, 1., G, acc, 1);
#else
      interpPk(k, 1., G, NULL, 1);
#endif
      fac =  k*k*k/19.73;
      //printf("%g %g %g\n", k, fac*G[1]*G[1],  fac*G[5]*G[5]);
      fprintf(outfile, "%g %g %g %g %g\n", k, fac*G[1]*G[1],  fac*G[5]*G[5],  fac*G[3]*G[3], fac*G[7]*G[7]);
    }

#ifdef GSL_INTERP
	  for(int n=0; n< DIM_ARRAY; n++)    
	    for(int nn=0; nn< NUMK_PAR_ARRAY;nn++)
	      gsl_interp_accel_free(acc[n][nn]);
#endif
       
  fclose(outfile);
}

//in km/sec/Mpc
double getHz(double zg)
{
  double OmegaR = 4.15e-5/(hconst*hconst);
  return hconst*100.*sqrt(OmegaM*pow(1.+zg, 3.) + (1-OmegaM -OmegaR) + OmegaR*pow(1.+zg, 4.));
}

int knum(int i)
{
    if(i < SIZE/2 +1)
	return i;
    else
	return -(SIZE - i);
}
