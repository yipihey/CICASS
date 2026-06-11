/*********************************************************************
Generates LCDM transfer function -- that can have a velocity
difference between the dark matter and baryons -- used by CICASS
initial conditions code.  See O'leary & McQuinn (2012) and McQuinn &
O'Leary (2012) for one application.
----------------------------------------------------------------------
usage: ./transfer.x -B0.2 -N128 -J1 -Z100 -D1 -SinitSB_transfer_out
This command generates the transfer function output for a 0.2 Mpc/h box and a
simulations with 128^3 particles starting at redshift=100.  (You can
also use this output for smaller particle number simulations.) J>0
indicates that the code will *not* do an internal transfer function
calculations to evolve from the input CAMB transfer fuctions to the
output redshift. Rather, it reads in a transfer function in the CAMB
format (http://camb.info/) at redshifts of z=0, 99, 100, 101 (-Z100 &
-D1 arguments) with names initSB_transfer_out_z%3d.dat and outputs a
file at the same redshift.  Also, you will need to add one number at
the top of the raw CAMB outputs specifying the number of rows in the
file. Four transfer functions are required to set the normalization at
z=0 (make sure SIGMA8 is set to what you want) and the time derivative
at initialization.

If the -S option is not provided, the input transfer function filename
defaults to "initSB_transfer_out" as this is the name used in the
example transfer function files that are provided.  The examples are
calculated for the default flat LCDM cosmology (with h = 0.71, OmegaM
= 0.27, OmegaB = 0.046, Sigma8 = 0.8, Tilt = 0.95, and 3 relativistic
species of neutrino)

TO CHANGE COSMOLOGICAL PARAMETERS, YOU NEED TO CHANGE THEM AT THE TOP
OF MAIN.CC AND RECOMPILE.  The transfer function files are supplied
are for the default cosmology.  Thus, you will also need to use CAMB
to generate a new set of files with the appropriate cosmology (which
is easy).

Another Example: 
./transfer.x -B0.2 -N128 -V30 -Z100 -D3 -SinitSB_transfer_out 
is the same as previous example except that the
input transfer functions are for z= 0 and z=997, 1000, 1003 (set
-I1100 to change these redshifts by z+=100).  In addtion, this example
evolves the CAMB transfer functions forward using the approximations
described in O'Leary and McQuinn 2012 (and originally motivated in
Tseliakhovich & Hirata 2010), here assuming an initial velocity of
vbc=30 km s^-1 at z=1000 (set with the-V30 option; which is approximately to
the RMS velocity difference between the baryons and dark matter at this redshift).

This code either uses RECFAST values for temperature and electron density
(RECFAST flag) or, for faster calculations, the analytic fit to the
temperature evolution in Tseliakhovich and Hirata (2010) [the default
setting].  We recommend setting the RECFAST flag (add -DRECFAST to
CFLAGS in Makefile).  This reads in the file more
recfast/xeTrecfast.out, which has format redshift -- electron fraction
-- temperature and interpolates.  This file is generated using the
default cosmology, but depends weakly on cosmological paramters
(mostly Omegab h^2 and Tcmb).  However, the user could easily generate a new such file with RECFAST.

Makefile:  Change path to include GSL library.  Set or unset the desired flags.

Compile time options:

PRINT_PK -- Have code first print the power spectra as function of
         wavenumber to stdout at the requested output redshift.  This
         is useful to check that the results make sense. costheta of the
         wavevector with vbc can be set at runtime (-C0.8 means costh = 0.8)

DONT_PRINT_ICS -- If you are only interested in 1D power spectra, this
         results in the code not generating an initial conditions
         file.

GSL -- ALWAYS have this set!  It will not compile otherwise.  The code
         initially used numerical recipes rather than gsl functions,
         and still can with some minor changes when this option is
         set.


Outputs: 
The output is put in the IC_outputs directory with format
sprintf(filename,"IC_outputs/initSimCartZI%.1lf_Vbc%.1lf_%d_%.1lf.dat", ZINIT, VSTREAM,
NUMK_VALS, boxsize). This is the format that is read into the IC code.  It outputs a 
grid of transfer function values indexed by kperp and kpar that are read in by CICASS.
The numbers should be read as the initial redshift, the streaming
velocity, the maximum grid size the supplied transfer function file
should be used for and the box size of the calculation.


---------------------------------------------------
current problems: -doesn't work with internal temperature calculation (integration is unstable)--uses RECFAST temperature
                  -does not include fluctuations in the electron fraction
                  (these are smaller than other fluctuations)
                     --a small modification to code would be required to include such fluctuations
*********************************************************************/


#define MPCTOKM 3.08568e19
#define MPCTOCM 3.08568e24
#define DELTAC 1
#define DELTAB 5
#define REAL 0
#define IMAG 1
#define VELREAL 2
#define VELIMAG 3

#include <math.h>
#include <iostream>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_integration.h>
#ifndef GSL
#include "dnumrecipes.h"
#endif

double setInterpTF(double k, double z, double *TFdir, int flag);
double interpolateRecFast(double z, double *xe, int flag);
void initialConditionsCartesian(double kmin);
double getPowerZ(double k, double z, double G[], int flag);
double TF_Exact(double k);
double alphaB_recomb(double T, int species);
void returnGrowth(double G[], double z, double k, double costh);
void setICs(double *y, double *dy, double k, double costh, double vstream);
void setCalcGrowthInt(double a, const double *y, double *deriv, double k1, 
		      double vstream1, double costh1,  int flag);
void calcGrowthInt(double a, double *y, double *dy);
double hubbleZ(double z);
void resetPowerSpectrum();
void setParams(int argc, char *argv[]);
#ifdef GSL
double *dvector(long nl, long nh);
void free_dvector(double *v, long nl, long nh);
#endif

/**********************************************************
Important physical constants
 **********************************************************/
const double TCMB(2.726);                   // CMB temperature at z=0
const double MPROTON(1.6726e-24);           // in grams
const double MELECTRON(9.11e-28);
const double MEL_EV(5.11e5);
const double LIGHTSPEED(3.0e10);
const double CRITDENSITY(1.8791e-29);   // Current critical density, 
                                        // in g/cm^3, not including h^2!
const double SIGMAT(0.665e-24);   //Thomson cross section in cm^2
const double YHE(0.25);
const double BOLTZK(1.3806e-16);            // in erg/K
const double MUB(1.22);           // Mean molecular weight, for primordial
const double aSB(7.56e-15); //u = aSB T^4 

/***************************************************
Basic parameters: all can be changed at runtime
 ***************************************************/
double BOXSIZE = 0.2;
int SIZE =128;
double VSTREAM = 0.;
double ZSTART = 1000.;     //The redshift VSTREAM is valid at
double ZINIT = 50;
int DREDSHIFT = 3; //spacing of redshift files to make derivative
char BASENAME_CAMB[100] = "initSB_transfer_out";

int JUSTCAMB = 0; 

/**************************************************************
Cosmological parameters!!!!   These must be set prior to 
compilation
 *************************************************************/
double hconst = 0.71; 
double OmegaM = 0.27;        //assumes flat universe
double OmegaB = 0.046;
double Sigma8 = 0.8;
double Tilt = 0.95;
double OmegaR =  4.15e-5/(hconst*hconst); //assumes 3 species of relativistic neutrinos, 
                                // from Dodelson, eqn 2.87
double COSTH_PRINT_PK = 1;
 
#ifndef ISOTHERMAL
int  NUMEQNS = 12;
#else
int  NUMEQNS = 8; //not doing temperature yet
#endif 

double sNorm; //sigma8


using namespace std;

int main(int argc, char *argv[])
{
  setParams(argc, argv);

  if((ZSTART - ZINIT)*(ZSTART - ZINIT) < 1e-5)  
    {
      JUSTCAMB = 1;
    }else if(JUSTCAMB >= 1)
    {
      ZSTART = ZINIT;
    }
  cout << "#JUSTCAMB = " << JUSTCAMB << " (if J>=1, then code directly outputs CAMB transfer function and its time derivative)" << endl << endl;

  setInterpTF(0, 0, NULL, 0);
  interpolateRecFast(0., NULL, 0);
  resetPowerSpectrum();

  double G[100], k=1;

#ifdef PRINT_PK
  cout << "printing power spectrum at ZINIT to stdout:" << endl << endl;
  cout << "k [Mpc^-1]   k^3 P_DM/2pi^2  k^3 P_gas/2pi^2  k^3 P_Temp/2pi^2" << endl;  
  cout.precision(5);
  for(k =.1; k< 10000; k*=1.2)
    {
      returnGrowth(G, ZINIT, k, COSTH_PRINT_PK);
      cout << k << "\t" << scientific << k*k*k/19.73*getPowerZ(k, ZINIT, G, 1) << "\t" << k*k*k/19.73*getPowerZ(k, ZINIT, G, 2)  << "\t" << k*k*k/19.73*getPowerZ(k, ZINIT, G, 2)  << endl;
    }
  cout << endl << endl;
#endif
  
#ifndef DONT_PRINT_ICS
  initialConditionsCartesian(2.*M_PI/BOXSIZE);
#endif

  return 1;
}

/********************************************************************
Reads in Parameters supplied at runtime
 ******************************************************************/
void setParams(int argc, char *argv[])
{
  while ((argc>1) && (argv[1][0]=='-')) {
    switch (argv[1][1]) {
    case 'B':
      BOXSIZE = atof(&argv[1][2]);//kpc
      break;
    case 'V':
      VSTREAM = atof(&argv[1][2]);//kpc
      break;
    case 'D':
      DREDSHIFT = atoi(&argv[1][2]);//kpc
      break;
    case 'Z':
      ZINIT = atof(&argv[1][2]);//kpc
      break;
    case 'I':
      ZSTART = atof(&argv[1][2]);//kpc
      break;
   case 'C':
      COSTH_PRINT_PK = atof(&argv[1][2]);//kpc
      break;
    case 'N':
      SIZE = atoi(&argv[1][2]);
      break;
    case 'J':
      JUSTCAMB = atoi(&argv[1][2]);
      break;
    case 'S':
      stpcpy(BASENAME_CAMB, &argv[1][2]);
      break;
    default:
      fprintf(stderr, "Bad option %s\n", argv[1]);
    }
    --argc;
    ++argv;
  }
}




/*****************************************************************************************
Generates cartesian initial conditions grid in kx & ky that is used by code when vbc > 0
 *****************************************************************************************/
void initialConditionsCartesian(double kmin)
{
  int NUMK_VALS = SIZE;
  int NUMK_PAR = (NUMK_VALS + 1);
  int NUMK_PERP = ((int) ((NUMK_VALS + 1)*sqrt(1.25))); 
  if(JUSTCAMB)
    {
      NUMK_PAR = 1;
      NUMK_PERP *=2;
    }
  double   boxsize = BOXSIZE;

  int i, j, n, DIM = NUMEQNS;
  double costh, Tk, xe, k, g[20], gold[20], growth[20];
  double *kvec_par, *kvec_perp;    
  char filename[200];
  double delta0, deltarmsvec[100];
  FILE *outfile;


  kvec_par = (double *) malloc(sizeof(double)*NUMK_PAR);
  kvec_perp = (double *) malloc(sizeof(double)*NUMK_PERP);

#ifdef GSL
  sprintf(filename, "IC_outputs/initSimCartZI%.1lf_Vbc%.1lf_%d_%.1lf.dat", ZINIT, VSTREAM, NUMK_VALS, boxsize); 
#else
  sprintf(filename, "IC_outputs/initSimCartNRZI%.1lf_Vbc%.1lf_%d_%.1lf.dat", ZINIT, VSTREAM, NUMK_VALS, boxsize); 
#endif
  if((outfile = fopen(filename, "w"))==NULL)
    {
      fprintf(stderr, "makeGFGrid Cartesian:  Could not open %s for writing.\n",
	      filename);
      exit(-9);
    }
  fprintf(stdout, "#makeGFGrid  Cartesian: opened %s\n", filename);

  for(i=0; i < NUMK_PAR; i++)
    kvec_par[i] = 2.*M_PI/boxsize*(i);//let this one go to zero 
  for(i=0; i < NUMK_PERP; i++)
    kvec_perp[i] = 2.*M_PI/boxsize*(i);
    
  fwrite(&DIM, sizeof(int), 1, outfile);
  fwrite(&(NUMK_PAR), sizeof(int), 1, outfile);
  fwrite(&(NUMK_PERP), sizeof(int), 1, outfile);
  fwrite(kvec_par, sizeof(double), NUMK_PAR, outfile);
  fwrite(kvec_perp, sizeof(double), NUMK_PERP, outfile);

  Tk = interpolateRecFast(ZINIT, &xe, 1);
  cout << "# " << Tk << " " << xe <<  endl;
  fwrite(&Tk, sizeof(double), 1, outfile);
  fwrite(&xe, sizeof(double), 1, outfile);
  
  for(i=0; i < NUMK_PAR; i++)
    {  
      for(j=0; j< NUMK_PERP;j++)
	{
	  // cout << i << " " << j << endl;//EVENTUALLY REMOVE

	  k= sqrt(kvec_par[i]*kvec_par[i]+kvec_perp[j]*kvec_perp[j]);

	  if(i == 0 && j==0) k = kmin; 

	  costh = kvec_par[i]/k;

	  delta0 = sqrt(getPowerZ(k*hconst, ZSTART, NULL, 1)); //dark matter power

	  returnGrowth(growth, ZINIT, k*hconst, costh);

	  for(n=0;n<DIM;n+=2)//interpolate using amplitude and phase
	    {
	      g[n] = sqrt(growth[n+1]*growth[n+1]+ growth[n+2]*growth[n+2]);
	      g[n+1] = acos(growth[n+1]/g[n]);


	      if(growth[n+2] < 0)
		g[n+1] = -g[n+1];

	      if(j!= 0)  //to make interpolation continous
		{
		  if(g[n+1] - gold[n+1] > .6)
		    {
		      g[n+1] -= 2.*M_PI;
		    }else if(g[n+1] - gold[n+1] < -.6)
		    {
		      g[n+1] += 2.*M_PI;
		    }
#ifdef DEBUG
		  if(fabs(g[n+1] - gold[n+1]) > .6)
		    {
		      fprintf(stderr, "Too big of a jump in phase of %le at %d %d %d.  Try more k-bins!\n", 
			      fabs(g[n+1] - gold[n+1]), n, i, j);
		      cout <<  k << " " << g[n+1] << " " << gold[n+1] << endl;
		    }
#endif
		}
	      gold[n]=g[n]; gold[n+1]=g[n+1];
	    }

	  for(n=0; n< DIM; n+=2)//only mutiply those that aren't phases
	    {	
	      deltarmsvec[n] = g[n]*delta0*pow(hconst, 1.5);
	      deltarmsvec[n+1] = g[n+1];
	      //    cout << delta0 << " " << deltarmsvec[n] << endl; exit(-5);
	    }

	  fwrite(deltarmsvec, sizeof(double), DIM, outfile);
	}
    }

  fclose(outfile);
   
  cout << "#Done outputing " << filename << " for IC generator." << endl; 

  free( kvec_par );  free( kvec_perp ); 
  return;
}




/***************************************************************
calculates growth factor as a function of z, k and costh
 **************************************************************/
void returnGrowth(double G[], double z, double k, double costh)
{
  double *y = dvector(1, NUMEQNS), *dy = dvector(1, NUMEQNS);
  int i;
  
  setICs(y, dy, k, costh, VSTREAM); //set initial conditions

  setCalcGrowthInt(0., NULL, NULL, k, VSTREAM, costh, 0);

#ifdef GSL
  int calcGrowthIntGSL(double a, const double *y, double *dy, void *params);
  gsl_odeiv2_system sys = {calcGrowthIntGSL, NULL, (size_t)NUMEQNS, NULL};

  gsl_odeiv2_driver * d =  gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45,
				    1e-3, 1e-6, 0.0);
  double t = 1./(1.+ZSTART), ti = 1./(1.+ z);
  int status = gsl_odeiv2_driver_apply (d, &t, ti, y+1);
 
  if (status != GSL_SUCCESS)
    {
      printf ("error, return value=%d\n", status);
      exit(-5);
    }
  gsl_odeiv2_driver_free(d);
#else
  int goodSteps,badSteps;
  odeint(y, NUMEQNS, 1./(1.+ZSTART), 1./(1.+ z), 1e-3, 1e-6, 0.,
	 &goodSteps,&badSteps, calcGrowthInt, bsstep); //ODEs
#endif

  for(i=1; i<= NUMEQNS; i++)
    G[i] = y[i];

  free_dvector(y, 1, NUMEQNS); free_dvector(dy, 1, NUMEQNS);

}


/************************************************************************
sets initial conditions for calculation
 **********************************************************************/
void setICs(double *y, double *dy, double k, double costh, double vstream)
{
  const double DELTA0 = 1.; //sets amplitude to unity

  //double a = 1./(1+ZSTART);
  double H = 100.*hconst*hubbleZ(ZSTART)/MPCTOKM;
  double deltab, deltacdot, deltabdot;  
  double TFc, TFdirc, TFb, TFdirb; 
  TFc = setInterpTF(k, 0., &TFdirc, 6); //dz of dark matter at z=1000
  TFb = setInterpTF(k, 0., &TFdirb, 7); //dz of dark matter at z=1000
  deltab = TFb/TFc*DELTA0;
  deltacdot = TFdirc/TFc*H*(1+ZSTART)*DELTA0;
  deltabdot = TFdirb/TFb*deltab*H*(1+ZSTART);

  /********************Initial Conditions*************************/
  y[1] = DELTA0; //real part of dark matter density at specified k
  y[2] = 0.;  //imaginary part (start with zero for simplicity)
  y[3] = -deltacdot; //1st derivative of delta_dm, real part
  y[4] = 0.; //1st derivative of delta_dm, imaginary part
  y[5] = deltab;  //real part of baryonic overdensity
  y[6] = 0.; //imaginary part of baryonic overdensity
  y[7] = -deltabdot; //real part of baryonic velocity
  y[8] = 0.;        //imaginary part

#ifndef ISOTHERMAL  //if set, does not track temperature
  y[9]  = TCMB*(1+ZSTART); //temperature -- unused by other eqns unless CALCULATE_MEAN_TEMP
  y[10] = 1.;  //electron fraction after recombination.  
               /***********Even though this is a total kludge because I don't include radition, I find that this does pretty well--electron       
                   fraction is enough of an attractor solution.***********************************************************/
  y[11] = 0.; //temperature
              // initially assumed isothermal (which should be a good assumption on simulated scales -- a better assumption would be to set
              //equal to temperature inhomogenety as the two are tightly coupled (as in Bovi and Dvorkin 2012) 
  y[12] = 0.;
#endif
}

/***********************************************************
Dummy function for differential equation solver
 **********************************************************/
void calcGrowthInt(double a, double *y, double *dy)
{
  setCalcGrowthInt(a, y, dy, 0., 0., 0., 1);
}

/***********************************************************
Dummy function for differential equation solver
 **********************************************************/
int calcGrowthIntGSL(double a, const double *y, double *dy, void *params)
{
  setCalcGrowthInt(a, y-1, dy-1, 0., 0., 0., 1);
  return GSL_SUCCESS;
}



/****************************************************************
solves for dy/dt for delta_c, delta_b, etc 
 ***************************************************************/
void setCalcGrowthInt(double a, const double *y, double *deriv, double k1, 
		      double vstream1, double costh1,  int flag)
{
  const double fHe = .25*YHE/((1. - YHE) + .25*YHE);
  static double k, costh, h, OC0, OB0, OR0, nH, ucmb0, tgamma0, vstream;
  double H, vbck, OMEGAC, OMEGAB, cs, T, xe, eta1;
  int i;

  if(flag == 0)
    {
      k = k1; costh = costh1; vstream = vstream1;
      OC0 = OmegaM-OmegaB;
      OB0 = OmegaB;
      h = hconst;
      OR0 = OmegaR;
      nH = CRITDENSITY*OB0*h*h*(1. - YHE)/(MPROTON);  //Hydrogen density
                              //hydrogen density, assuming helium is neutral
      ucmb0 = aSB*TCMB*TCMB*TCMB*TCMB;
      tgamma0 = 3.*MELECTRON*LIGHTSPEED/(8.*SIGMAT*ucmb0);
      return;
    }

  H = 100.*hconst*hubbleZ(1./a - 1.)/MPCTOKM;
  vbck = vstream/MPCTOKM/(a*(1.+ZSTART)); //VSTREAM is in km/s
  OMEGAC = OC0/(OC0 +OB0 + OR0/a);
  OMEGAB = OB0/(OC0 +OB0 + OR0/a);

  // cout << y[9]<< endl;
  T = y[9];
  xe = y[10];
#ifdef RECFAST //fit to RECFAST: recommended
  T = interpolateRecFast(1./a-1., &xe, 1);
#elif USEPROPERTEMP
  //don't do anything -- currently the integration is unstable for this option
#else  //use Tseliakhovich & Hirata '10 interpolation formula
  const double a1 = 0.008403, a2 = 0.008696; //fit to recfast in TH11
  T= 2.726/a/(1+ a/a1/(1+ a2/a*sqrt(a2/a)));
  xe = y[10];
#endif

  cs = sqrt(BOLTZK*T/MUB/MPROTON)/MPCTOCM; //sound speed for iosthermal gas

#ifndef GSL
  //if(NUMEQNS < 8){y[11] = 0.; y[12] = 0.;}
#endif

  /*************************DARK MATTER*********************************/
  deriv[DELTAC+REAL] =  -y[DELTAC+VELREAL];
  deriv[DELTAC+IMAG] =  -y[DELTAC+VELIMAG];
  deriv[DELTAC+VELREAL] =  
    - 1.5*H*H*(OMEGAC*y[DELTAC+REAL]+OMEGAB*y[DELTAB+REAL])
    - 2.*H*y[DELTAC+VELREAL];
  deriv[DELTAC+VELIMAG] =  
    - 1.5*H*H*(OMEGAC*y[DELTAC+IMAG]+OMEGAB*y[DELTAB+IMAG])
    - 2.*H*y[DELTAC+VELIMAG];

  //baryons
  deriv[DELTAB+REAL] = vbck*k*costh*y[DELTAB+IMAG]/a - y[DELTAB+VELREAL];
  deriv[DELTAB+IMAG] = -vbck*k*costh*y[DELTAB+REAL]/a - y[DELTAB+VELIMAG];
  deriv[DELTAB+VELREAL] =  vbck*k*costh*y[DELTAB+VELIMAG]/a 
    - 1.5*H*H*(OMEGAC*y[DELTAC+REAL]+OMEGAB*y[DELTAB+REAL])
    - 2.*H*y[DELTAB+VELREAL] + cs*cs*k*k/(a*a)*(y[DELTAB+REAL] + y[11]);
  deriv[DELTAB+VELIMAG] =  -vbck*k*costh*y[DELTAB+VELREAL]/a  
    - 1.5*H*H*(OMEGAC*y[DELTAC+IMAG]+OMEGAB*y[DELTAB+IMAG])
    - 2.*H*y[DELTAB+VELIMAG] + cs*cs*k*k/(a*a)*(y[DELTAB+IMAG] + y[12]); 
    

#ifndef ISOTHERMAL //evolve temperature and electrons
  eta1 = (1.+ fHe + xe);

  //EQNS 9 and, if RECFAST is on, 10 are not currently used
  deriv[9] =-2.*H*y[9] + y[10]/(eta1*tgamma0*(a*a*a*a))
    *(TCMB/a - y[9]);//mean temperature (adiabatic and compton cooling)
  deriv[10] = -alphaB_recomb(T, 0)*y[10]*y[10]*nH/(a*a*a); 

  deriv[11] =  vbck*k*costh*y[12]/a + 2./3.*(deriv[DELTAB+REAL]- vbck*k*costh*y[DELTAB+IMAG]/a) - xe/(eta1*tgamma0*(a*a*a*a))
    *(TCMB/a/T)*y[11]; //temperature fluctuations --uses either recfast temp or y[9]
  deriv[12] = -vbck*k*costh*y[11]/a + 2./3.*(deriv[DELTAB+IMAG] + vbck*k*costh*y[DELTAB+REAL]/a) - xe/(eta1*tgamma0*(a*a*a*a))
    *(TCMB/a/T)*y[12]; //temperature fluctuations (imaginary)

  /******Currently do not track electron fraction fluctuations***********/
#endif
   
  for(i=1; i<=NUMEQNS; i++)
    {
      deriv[i] /= (H*a); //to switch from time to scale factor derivative
    }
}



/***********************************************************
Reads in CAMB TFs for z= 0 and 1000
currently z does nothing -- should solve for growth factors
Tdir gives the derivative at z= 1000
 ***********************************************************/
double setInterpTF(double k, double z, double *TFdir, int flag)
{
  int i, j;
  static int init = 0, numk, numk2;
  static double *lkarr, *lkarr2, *T[12], *T2[12], *Tdir[12];
#ifdef GSL
  static gsl_spline *Tspline[12], *T2spline[12], *Tdirspline[12];
  static gsl_interp_accel *Tacc[12], *T2acc[12], *Tdiracc[12];
#else 
  const double natural(1.0e30);
#endif
  double TF;
  

  if(init == 0)
    {      
      cout << "#Initializing TFs" << endl;
      init++;
      int numk3, numk4;
      double *lkarr3;
      double t[10];

      //temporary
      char *BASETRANSFER = (char *) BASENAME_CAMB;
      int z = (int) ZSTART;

      char filename0[200], filename1[200], filename1p[200], filename1m[200];
      sprintf(filename0, "TFs/%s_z%03d.dat", BASETRANSFER, 0);
      sprintf(filename1, "TFs/%s_z%03d.dat", BASETRANSFER, z);
      sprintf(filename1m, "TFs/%s_z%03d.dat", BASETRANSFER, z-DREDSHIFT);
      sprintf(filename1p, "TFs/%s_z%03d.dat", BASETRANSFER, z+DREDSHIFT);

      // char *infz0_name = (char *)"TFs/initSB_transfer_out_z000.dat";
      FILE *infz0 = fopen(filename0, "r");
      //char *infz1000_name = (char *)"TFs/initSB_transfer_out_z1000.dat";
      FILE *infzinit = fopen(filename1, "r");
      //char *infz1003_name = (char *)"TFs/initSB_transfer_out_z1003.dat";
      FILE *infzinitp = fopen(filename1p, "r");
      // char *infz997_name = (char *)"TFs/initSB_transfer_out_z997.dat";
      FILE *infzinitm = fopen(filename1m, "r");

      if(infz0 == NULL || infzinit == NULL || infzinitp == NULL || infzinitm == NULL)
	{
	  fprintf(stderr, "Could not open one of transfer function files:  %s %s %s %s\n", 
		  filename0, filename1, filename1m, filename1p);
	  exit(-6);
	}

      fscanf(infz0, "%d\n", &numk);
      fscanf(infzinit, "%d\n", &numk2);
      fscanf(infzinitp, "%d\n", &numk3);
      fscanf(infzinitm, "%d\n", &numk4);

      if(numk !=numk2 || numk != numk3 || numk3 != numk4){
	cerr << "Input ransfer function arrays not equal!!!   " << numk << " " << numk2 << " " << numk3 << " " << numk4 << endl; 
	exit(-6);
      }

      lkarr = dvector(1, numk);
      lkarr2 = dvector(1, numk);
      lkarr3 = dvector(1, numk);      
      for(i=0;i<12;i++)
	{
	  T[i] = dvector(1, numk);
	  T2[i] = dvector(1, numk);
	  Tdir[i] = dvector(1, numk);

#ifdef GSL
	  Tspline[i] = gsl_spline_alloc(gsl_interp_cspline, numk);
	  Tacc[i]  = gsl_interp_accel_alloc ();
	  T2spline[i] = gsl_spline_alloc(gsl_interp_cspline, numk);
	  T2acc[i]  = gsl_interp_accel_alloc ();
	  Tdirspline[i] = gsl_spline_alloc(gsl_interp_cspline, numk);
	  Tdiracc[i]  = gsl_interp_accel_alloc ();

	  if(i >= 6) break;
#endif
	}

      /***********************************Read in TRANSFERFUCNTIONS**********************/
      for(i=1;i<=numk;i++)
	{
	  fscanf(infz0, "   %lg   %lg   %lg   %lg   %lg   %lg   %lg\n", lkarr+i,
	       T[0]+i, T[1]+i, T[2]+i,T[3]+i,T[4]+i,T[5]+i);
	  fscanf(infzinit, "%le %le %le %le %le %le %le\n", lkarr2+i,
	       T2[0]+i, T2[1]+i, T2[2]+i,T2[3]+i,T2[4]+i,T2[5]+i);
	  fscanf(infzinitp, "%le %le %le %le %le %le %le\n", lkarr3+i,
	       Tdir[0]+i, Tdir[1]+i, Tdir[2]+i,Tdir[3]+i,Tdir[4]+i,Tdir[5]+i);
	  fscanf(infzinitm, "%le %le %le %le %le %le %le\n", &t[9],
	       &t[0], &t[1], &t[2], &t[3], &t[4], &t[5]);
	  for(j=0;j<6;j++)
	    Tdir[j][i] = (t[j] - Tdir[j][i])/(2.*DREDSHIFT);  //Not terribly general

	  lkarr[i] = log10(lkarr[i]);
	  lkarr2[i] = log10(lkarr2[i]);
	}


      
      for(i=0;i<6;i++)
	{
#ifdef GSL
	  gsl_spline_init(Tspline[i], lkarr+1, T[i]+1, numk);
   	  gsl_spline_init(T2spline[i], lkarr2+1, T2[i]+1, numk);
	  gsl_spline_init(Tdirspline[i], lkarr2+1, Tdir[i]+1, numk);
#else
	  spline(lkarr, T[i], numk, natural, natural, T[6+i]);  //change to GSL
	  spline(lkarr2, T2[i], numk, natural, natural, T2[6+i]);
	  spline(lkarr2, Tdir[i], numk, natural, natural, Tdir[6+i]);
#endif
	}
      cout << "#done!" << endl;

      fclose(infz0); fclose(infzinit); fclose(infzinitm); fclose(infzinitp);
      free_dvector(lkarr3, 1,numk3);
      return -1.;
    }else if(flag == 2){
    free_dvector(lkarr, 1,numk2);
    free_dvector(lkarr2, 1,numk2);


    for(i=0;i<12;i++)
      {
	free_dvector(T[i], 1, numk);
	free_dvector(T2[i], 1, numk);
	free_dvector(Tdir[i], 1, numk);

#ifdef GSL
	  gsl_spline_free(Tspline[i]);
	  gsl_interp_accel_free(Tacc[i]);
	  gsl_spline_free(T2spline[i]);
	  gsl_interp_accel_free(T2acc[i]);
	  gsl_spline_free(Tdirspline[i]);
	  gsl_interp_accel_free(Tdiracc[i]);

	  if(i >= 6) break;
#endif
      }
    return 1;
  }
  
  double lgk = log10(k/hconst); //to put in h/Mpc

  if(lkarr[numk] < lgk || lkarr[1] > lgk)
    {
      fprintf(stderr, "k is out of range\n");
	 return 0.;
    }  

  if(flag >=6)//z=1000 TF and dirivative
    {
#ifdef GSL
      TF= gsl_spline_eval(T2spline[flag-6], lgk, T2acc[flag-6]);
#else
      splint(lkarr2, T2[flag-6], T2[flag], numk2, lgk, &TF);  
#endif
      if(TFdir != NULL)
	{
#ifdef GSL
	  *TFdir = gsl_spline_eval(Tdirspline[flag-6], lgk, Tdiracc[flag-6]);
#else
	  splint(lkarr2, Tdir[flag-6], Tdir[flag], numk2, lgk, TFdir);  
#endif
	}
    }
  else //otherwise
#ifdef GSL
    TF= gsl_spline_eval(Tspline[flag], lgk, Tacc[flag]);
#else
    splint(lkarr, T[flag], T[6+flag], numk, lgk, &TF);
#endif 

  return TF;
}


/***********************************************
Returns density power spectrum
flag = 0 total
flag = 1 dark matter
flag = 2 baryon
flag = 3 temperature
*************************************************/
double getPowerZ(double k, double z, double G[], int flag)
{
  double temp, g1, g2, g3;

  /***********Initializes directly from z=1000 transfer function*********/
  if((z -ZSTART)*(z -ZSTART) < 1e-5)//will use this part to init sims
    {

      if(flag == 1)  //dark matter (z=1000)
	temp = 2.0*pow(M_PI*sNorm*setInterpTF(k, 0., NULL, 6),2.0);
      else if(flag == 2) //baryons
	temp = 2.0*pow(M_PI*sNorm*setInterpTF(k, 0., NULL, 7),2.0);
      else if(flag == 0) //total
	temp = 2.0*pow(M_PI*sNorm*setInterpTF(k, 0., NULL, 11),2.0);	
      else
	{
	  cout << "#powerZ: This option does not work here" << endl;
	  exit(-5);
	}
    }else
    {	 
      g1= sqrt(G[2]*G[2]+G[1]*G[1]); //dm
      g2 =  sqrt(G[5]*G[5]+G[6]*G[6]);//baryons
      g3 =  sqrt(G[11]*G[11]+G[12]*G[12]);//temperature

      if(flag == 0) //tot
	{
	  double fb = OmegaB/OmegaM;
	  temp = 2.0*pow(M_PI*sNorm*(g1*(1.-fb)*setInterpTF(k, 0., NULL, 6)
				   +g2*fb*setInterpTF(k, 0., NULL, 6)),2.0);
	}
      else if (flag == 1)
	temp = 2.0*pow(M_PI*sNorm*g1*setInterpTF(k, 0., NULL, 6), 2.);
      else if (flag == 2)
	temp = 2.0*pow(M_PI*sNorm*g2*setInterpTF(k, 0., NULL, 6), 2.);
      else if (flag == 3)
	temp = 2.0*pow(M_PI*sNorm*g3*setInterpTF(k, 0., NULL, 6), 2.);
      else
	{
	  cout << "powerZ: This option does not work here" << endl;
	  exit(-6);
	}
    }

  temp *= pow(k,Tilt);
  return temp;
}


//returns total matter z= 0 transfer function using CAMB
double TF_Exact(double k)
{
  // double setInterpTF(double k, double z, double *Tdir, int flag);
 
  return setInterpTF(k, 0., NULL, 5); 
}


/********************************************************
Interpolates a recfast file to return temperature
(and a pointer to the electron fraction) as a function of z
*********************************************************/
double interpolateRecFast(double z, double *xe, int flag)
{
  static int numz;
  static double *zarr, *Tarr, *xearr, *Tarr2, *xearr2;
  int i;
#ifdef GSL
  static gsl_spline *Tspline, *xespline;
  static gsl_interp_accel *Tacc, *xeacc;
#else 
  const double natural(1.0e30);
#endif

  if(flag == 0)
    {
      char *fname = (char *)"recfast/xeTrecfast.out";
      FILE *infile = fopen(fname, "r");
      fscanf(infile, "%d\n", &numz);

      zarr = dvector(1, numz);
      Tarr = dvector(1, numz);
      xearr = dvector(1, numz);
      Tarr2 = dvector(1, numz); //relics fron when using numerical recipes
      xearr2 = dvector(1, numz);

      for(i=numz; i>=1; i--) //need to read backwards so z is increasing
	fscanf(infile, "%le %le %le\n", &zarr[i], &xearr[i], &Tarr[i]);


#ifdef GSL
      Tspline = gsl_spline_alloc(gsl_interp_cspline, numz);
      Tacc  = gsl_interp_accel_alloc();
      xespline = gsl_spline_alloc(gsl_interp_cspline, numz);
      xeacc  = gsl_interp_accel_alloc();
      gsl_spline_init(Tspline, zarr+1, Tarr+1, numz);
      gsl_spline_init(xespline, zarr+1, xearr+1, numz);
#else
      spline(zarr, Tarr, numz, natural, natural, Tarr2);
      spline(zarr, xearr, numz, natural, natural, xearr2);
#endif
      fclose(infile);
      return -1.;
    }
  else if(flag ==2)
    {
      free_dvector(Tarr, 1, numz);
      free_dvector(Tarr2, 1, numz);
      free_dvector(xearr, 1, numz);
      free_dvector(xearr2, 1, numz);
      free_dvector(zarr, 1, numz);

#ifdef GSL
      gsl_spline_free(Tspline);
      gsl_interp_accel_free(Tacc);
      gsl_spline_free(xespline);
      gsl_interp_accel_free(xeacc);
#endif
      
      return -1.;
    }

  double T;

#ifdef GSL
  if(xe != NULL)
    *xe= gsl_spline_eval(xespline, z, xeacc);   
  T = gsl_spline_eval(Tspline, z, Tacc);
#else
  if(xe != NULL)
    splint(zarr, xearr, xearr2, numz, z, xe);
  splint(zarr, Tarr, Tarr2, numz, z, &T);
#endif

  return T;
}


/************************************************************
Returns value of Hubble constant at zCurrent in units of H0. 
************************************************************/
double hubbleZ(double zCurrent)
{
  double temp;

  zCurrent += 1.0;
  temp = ((1.0 - OmegaM-OmegaR) 
	  + OmegaM*pow(zCurrent,3.0) + OmegaR*pow(zCurrent,4.0));
  return sqrt(temp);
}


/********************************************************************
Below are routines to correctly normalize power spectrum
 *******************************************************************/
void resetPowerSpectrum()
{
  double acc,sig8;
  double setSigmatop(double kl, int flag);

  // Normalize sigma8 at present time
  double scale = 8.0/hconst;
  acc = 1.0e-6;
  setSigmatop(0.0,1);
#ifdef GSL
  double sigmatopGSL(double kl, void *param);
  double sigmatop2GSL(double k, void *param);
  gsl_function F;
  double error, sig82;
  gsl_integration_workspace *w 
    = gsl_integration_workspace_alloc(1000);
  F.function = &sigmatop2GSL;
  gsl_integration_qags(&F, 1e-5, 0.001/scale, 0, acc, 1000,
                             w, &sig8, &error); 
  F.function = &sigmatopGSL;
  gsl_integration_qags(&F, log(0.001/scale), log(100.0/scale), 
		       0, acc, 1000, w, &sig82, &error); 
  sig8 += sig82;
#else
  double sigmatop2(double k); double sigmatop(double kl);
  sig8 = qromb(sigmatop2,1.0e-5,0.001/scale,acc);
  sig8 += qromb(sigmatop,log(0.001/scale),log(0.1/scale),acc);
  sig8 += qromb(sigmatop,log(0.1/scale),log(1.0/scale),acc);
  sig8 += qromb(sigmatop,log(1.0/scale),log(10.0/scale),acc); 
  sig8 += qromb(sigmatop,log(10.0/scale),log(100.0/scale),acc);
#endif
  sig8 = sqrt(sig8);
  sNorm = Sigma8/sig8;

  cout << "#Sigma8 set: sNorm= " << sNorm << endl; 
}


/* Calculates sigmatophat, assuming k is entered as ln(k) */
double setSigmatop(double kl, int flag)
{
  double k,x,sigtop;
  double scale = 8.0/hconst; //for sigma8

  if (flag == 1) {
    return 0.0;
  }

  k = exp(kl);
  x = scale*k;
 
   sigtop = pow(k,3.0+Tilt)*pow(TF_Exact(k),2.0)*
           pow(3.0*(x*cos(x) - sin(x))/pow(x,3.0),2.0);

  return sigtop;
}

double sigmatopGSL(double kl, void *param)
{
  return setSigmatop(kl,0);
}

/* Calculates sigmatophat, assuming k is entered as linear. */
double sigmatop2GSL(double k, void *param) 
{
  double sigmatop(double kl);
  return sigmatop(log(k))/k;
}

double sigmatop(double kl)
{
  return setSigmatop(kl,0);
}

/* Calculates sigmatophat, assuming k is entered as linear. */
double sigmatop2(double k) 
{
  return sigmatop(log(k))/k;
}


//case A and B rec coefficients
//updated H to be valid at ultra low temperatures (using recfast rate)
//currently only H is used
double alphaB_recomb(double T, int species)
{
   //Hydrogen
   if(species == 0)
     {
     //eqn 70 in RECFAST paper from Hummer (1994) --valid at low temperatures
     const double a = 4.309, b=-0.6166, c=0.6703, d=0.5300;
     return 1.e-13*a*pow(T/1.e4,b)/(1.+c*pow(T/1e4,d));
     }
   
   cerr << "Species not supported!\n" << endl;
   exit(-5);
   return 0.;
}

#ifdef GSL
#define FREE_ARG char*
#define NR_END 1 // used by vector

double *dvector(long nl, long nh)
{
  double *v;

  v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
  return v-nl+NR_END;
}
void free_dvector(double *v, long nl, long nh)
{
  free((FREE_ARG) (v+nl-NR_END));
}

#endif
