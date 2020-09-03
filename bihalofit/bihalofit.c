/*
A code for computing the non-linear matter bispectrum (BS) based on BiHalofit (arXiv:1911.07886)
written by Ryuichi Takahashi in Dec 2019
a bug fixed in Mar 2020 (Thanks Sven Heydenreich for indicating it) 

Plese type "gcc bihalofit.c -O2 -lm" to compile this

Wave numbers (k1,k2,k3) and redshift are given in the follwing lines.
line 77: z is a redshift
     84: k1,k2,k3 are wavenumbers of triangle configulation in units of [h/Mpc]

The default cosmological parameters are consistent with the Planck 2015 LambdaCDM model.
If you want to change the cosmological model, please modify the following lines.
lines 41-46: cosmological parameters, this code can be used for flat wCDM model (CDM + dark energy with a constant equation of state)
line 65: table of linear P(k) (1st & 2nd columns are k[h/Mpc] & P(k)[(Mpc/h)^3]). 
         if you give a P(k) table, then the code uses it.
         if not, the code uses the Eisenstein & Hu (1999) fitting formula as the linear P(k).
line 51: n_data_max should be larger than the number of lines in the P(k) table
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
double bispec(double k1, double k2, double k3, double z);
double bispec_tree(double k1, double k2, double k3, double z);
double F2(double k1, double k2, double k3, double z);
double F2_tree(double k1, double k2, double k3);
double baryon_ratio(double k1, double k2, double k3, double z);

double calc_r_sigma(double z);
double sigmam(double r, int j); 
double window(double x, int i);
double linear_pk(double k); 
double linear_pk_data(double k); 
double linear_pk_eh(double k);   
double lgr(double z);
double lgr_func(int j, double x, double y[2]);

// cosmological parameters are given below
double h;     // Hubble parameter 
double sigma8; // sigma 8
double omb;   // Omega baryon
double omc;   // Omega CDM
double ns;    // spectral index of linear P(k) 
double w;       // equation of state of dark energy
double om,ow,norm;   

double r_sigma,n_eff,D1;

#define n_data_max 10000
double k_data[n_data_max],pk_data[n_data_max];
int n_data=0;

double eps=1.e-4;   // fractional accuracy of BS computaion 

int main()
{
  FILE *fp;
  double z,k1,k2,k3;
  printf("h sigma8 omb omc ns w k1 k2 k3 z ");
  scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &h, &sigma8, &omb, &omc, &ns, &w, &k1, &k2, &k3, &z);
  om=omb+omc;  // Omega matter
  ow=1.-om;  // Omega dark energy 

  fp=NULL;   // linear P(k) table
  if(fp!=NULL){
    while(fscanf(fp, "%lf %lf", &k_data[n_data], &pk_data[n_data])!=EOF){
      n_data++;
      if(n_data>n_data_max) printf("n_data_max should be larger than the number of data lines \n");
    }  // k[h/Mpc]   P(k)[(Mpc/h)^3]
    fclose(fp);
  }
  
  norm=1.;
  norm*=sigma8/sigmam(8.,0);   // P(k) amplitude normalized by sigma8
  
  //z=0.4;  // redshift

  // calculating D1, r_sigma & n_eff for a given redshift
  D1=lgr(z)/lgr(0.);   // linear growth factor
  r_sigma=calc_r_sigma(z);  // =1/k_NL [Mpc/h] in Eq.(B1) 
  n_eff=-3.+2.*pow(D1*sigmam(r_sigma,2),2);   // n_eff in Eq.(B2)

  // k1=1., k2=1.5, k3=2.;   // [h/Mpc]
  //printf("k2? ");
  //scanf("%lf", &k2);
  //printf("k3? ");
  //scanf("%lf", &k3);

  // non-linear BS w/o baryons, tree-level BS [(Mpc/h)^6] & baryon ratio for given k1,k2,k3 and z 
  printf("%lf %lf %lf %lf %lf %lf \n", k1, k2, k3, bispec(k1,k2,k3,z), bispec_tree(k1,k2,k3,z), baryon_ratio(k1,k2,k3,z));  
    
  return 0;
}


double bispec(double k1, double k2, double k3, double z)   // non-linear BS w/o baryons [(Mpc/h)^6]
{
  int i,j;
  double q[4],k[4],kt,logsigma8z,r1,r2;
  double an,bn,cn,en,fn,gn,hn,mn,nn,pn,alphan,betan,mun,nun,BS1h,BS3h,PSE[4];

  if(z>10.) return bispec_tree(k1,k2,k3,z); 
  
  q[1]=k1*r_sigma, q[2]=k2*r_sigma, q[3]=k3*r_sigma;  // dimensionless wavenumbers

  k[1]=k1, k[2]=k2, k[3]=k3;
  // sorting k[i] so that k[1]>=k[2]>=k[3]
  for(i=1;i<=3;i++){
    for(j=i+1;j<=3;j++){
      if(k[i]<k[j]){
        kt=k[j];
        k[j]=k[i];
        k[i]=kt;
      }}}
  r1=k[3]/k[1], r2=(k[2]+k[3]-k[1])/k[1];   // Eq.(B8)
  if(k[1]>k[2]+k[3]) printf("Error: triangle is not formed \n");

  logsigma8z=log10(D1*sigma8);
  
  // 1-halo term parameters in Eq.(B7)
  an=pow(10.,-2.167-2.944*logsigma8z-1.106*pow(logsigma8z,2)-2.865*pow(logsigma8z,3)-0.310*pow(r1,pow(10.,0.182+0.57*n_eff)));
  bn=pow(10.,-3.428-2.681*logsigma8z+1.624*pow(logsigma8z,2)-0.095*pow(logsigma8z,3));
  cn=pow(10.,0.159-1.107*n_eff);
  alphan=pow(10.,-4.348-3.006*n_eff-0.5745*pow(n_eff,2)+pow(10.,-0.9+0.2*n_eff)*pow(r2,2));
  if(alphan>1.-(2./3.)*ns) alphan=1.-(2./3.)*ns;
  betan=pow(10.,-1.731-2.845*n_eff-1.4995*pow(n_eff,2)-0.2811*pow(n_eff,3)+0.007*r2);

  // 1-halo term bispectrum in Eq.(B4)
  BS1h=1.;
  for(i=1;i<=3;i++){
    BS1h*=1./(an*pow(q[i],alphan)+bn*pow(q[i],betan))/(1.+1./(cn*q[i]));
  }
  
  // 3-halo term parameters in Eq.(B9)
  fn=pow(10.,-10.533-16.838*n_eff-9.3048*pow(n_eff,2)-1.8263*pow(n_eff,3));
  gn=pow(10.,2.787+2.405*n_eff+0.4577*pow(n_eff,2));
  hn=pow(10.,-1.118-0.394*n_eff);
  mn=pow(10.,-2.605-2.434*logsigma8z+5.71*pow(logsigma8z,2));
  nn=pow(10.,-4.468-3.08*logsigma8z+1.035*pow(logsigma8z,2));
  mun=pow(10.,15.312+22.977*n_eff+10.9579*pow(n_eff,2)+1.6586*pow(n_eff,3));
  nun=pow(10.,1.347+1.246*n_eff+0.4525*pow(n_eff,2));
  pn=pow(10.,0.071-0.433*n_eff);
  en=pow(10.,-0.632+0.646*n_eff);

  for(i=1;i<=3;i++){
    PSE[i]=(1.+fn*pow(q[i],2))/(1.+gn*q[i]+hn*pow(q[i],2))*pow(D1,2)*linear_pk(q[i]/r_sigma)+1./(mn*pow(q[i],mun)+nn*pow(q[i],nun))/(1.+pow(pn*q[i],-3));  // enhanced P(k) in Eq.(B6) 
  }

  // 3-halo term bispectrum in Eq.(B5)
  BS3h=2.*(F2(k1,k2,k3,z)*PSE[1]*PSE[2]+F2(k2,k3,k1,z)*PSE[2]*PSE[3]+F2(k3,k1,k2,z)*PSE[3]*PSE[1]);
  for(i=1;i<=3;i++) BS3h*=1./(1.+en*q[i]);
  
  return BS1h+BS3h;
}


double bispec_tree(double k1, double k2, double k3, double z)  // tree-level BS [(Mpc/h)^6]
{
  return pow(D1,4)*2.*(F2_tree(k1,k2,k3)*linear_pk(k1)*linear_pk(k2)
		      +F2_tree(k2,k3,k1)*linear_pk(k2)*linear_pk(k3)
		      +F2_tree(k3,k1,k2)*linear_pk(k3)*linear_pk(k1));
}


double F2(double k1, double k2, double k3, double z)
{
  double a,q[4],dn,omz,logsigma8z;

  q[3]=k3*r_sigma;
  
  logsigma8z=log10(D1*sigma8);
  a=1./(1.+z);
  omz=om/(om+ow*pow(a,-3.*w));   // Omega matter at z

  dn=pow(10.,-0.483+0.892*logsigma8z-0.086*omz);

  return F2_tree(k1,k2,k3)+dn*q[3];
}


double F2_tree(double k1, double k2, double k3)  // F2 kernel in tree level 
{
  double costheta12=0.5*(k3*k3-k1*k1-k2*k2)/(k1*k2);
  return (5./7.)+0.5*costheta12*(k1/k2+k2/k1)+(2./7.)*costheta12*costheta12;
}


double baryon_ratio(double k1, double k2, double k3, double z)   // bispectrum ratio with to without baryons  // k[h/Mpc]
{
  int i;
  double a,A0,A1,mu0,mu1,sigma0,sigma1,alpha0,alpha2,beta2,ks,k[4],x[4],Rb;

  if(z>5.) return 1.;  // baryon_ratio is calbrated at z=0-5   
  
  a=1./(1.+z);
  k[1]=k1, k[2]=k2, k[3]=k3;
  for(i=1;i<=3;i++) x[i]=log10(k[i]);  
  
  if(a>0.5) A0=0.068*pow(a-0.5,0.47);
  else A0=0.;
  mu0=0.018*a+0.837*a*a;
  sigma0=0.881*mu0;
  alpha0=2.346;
  
  if(a>0.2) A1=1.052*pow(a-0.2,1.41);
  else A1=0.;
  mu1=fabs(0.172+3.048*a-0.675*a*a);
  sigma1=(0.494-0.039*a)*mu1;
  
  ks=29.90-38.73*a+24.30*a*a;
  alpha2=2.25;
  beta2=0.563/(pow(a/0.060,0.02)+1.)/alpha2;

  Rb=1.;
  for(i=1;i<=3;i++){
    Rb*=A0*exp(-pow(fabs(x[i]-mu0)/sigma0,alpha0))-A1*exp(-pow(fabs(x[i]-mu1)/sigma1,2))+pow(1.+pow(k[i]/ks,alpha2),beta2);   // Eq.(C1)
  }

  return Rb;
}

  
double calc_r_sigma(double z)  // return r_sigma[Mpc/h] (=1/k_sigma)
{
  double k,k1,k2;

  k1=k2=1.;
  for(;;){
    if(D1*sigmam(1./k1,1)<1.) break;
    k1*=0.5;
  }
  for(;;){
    if(D1*sigmam(1./k2,1)>1.) break;
    k2*=2.;
  }

  for(;;){
    k=0.5*(k1+k2);
    if(D1*sigmam(1./k,1)<1.) k1=k; 
    else if(D1*sigmam(1./k,1)>1.) k2=k;
    if(D1*sigmam(1./k,1)==1. || fabs(k2/k1-1.)<eps*0.1) break;
  }

  return 1./k;
}


double linear_pk(double k)  // linear P(k)   k[h/Mpc], P(k)[(Mpc/h)^3]
{
  if(n_data!=0) return linear_pk_data(k); 
  else return linear_pk_eh(k);
}


double linear_pk_data(double k)   // linear P(k) interpolated from the given table,  k[h/Mpc]  P(k)[(Mpc/h)^3]
{
  int j,j1,j2,jm;
  double lk,dlk,f;

  lk=log10(k);
  if(k<k_data[0]) return 0.;
  if(k>k_data[n_data-1]) return 0.;
  
  j1=0, j2=n_data-1, jm=(j1+j2)/2;
  for(;;){
    if(k>k_data[jm]) j1=jm;
    else j2=jm;
    jm=(j1+j2)/2;

    if(j2-j1==1) break;
  }
  j=j1;

  f=(log10(pk_data[j+1])-log10(pk_data[j]))/(log10(k_data[j+1])-log10(k_data[j]))*(lk-log10(k_data[j]))+log10(pk_data[j]);
  
  return norm*norm*pow(10.,f);  
}


double linear_pk_eh(double k)   // Eisenstein & Hu (1999) fitting formula without wiggle,      k[h/Mpc], P(k)[(Mpc/h)^3]
{
  double pk,delk,alnu,geff,qeff,L,C;
  k*=h;  // unit conversion from [h/Mpc] to [1/Mpc]
  
  double fc=omc/om;
  double fb=omb/om;
  double theta=2.728/2.7;
  double pc=0.25*(5.0-sqrt(1.0+24.0*fc));
  double omh2=om*h*h;
  double ombh2=omb*h*h;
  double zeq=2.5e+4*omh2/pow(theta,4);
  double b1=0.313*pow(omh2,-0.419)*(1.0+0.607*pow(omh2,0.674));
  double b2=0.238*pow(omh2,0.223);
  double zd=1291.0*pow(omh2,0.251)/(1.0+0.659*pow(omh2,0.828))*(1.0+b1*pow(ombh2,b2));
  double yd=(1.0+zeq)/(1.0+zd);
  double sh=44.5*log(9.83/(omh2))/sqrt(1.0+10.0*pow(ombh2,0.75));

  alnu=fc*(5.0-2.0*pc)/5.0*(1.0-0.553*fb+0.126*fb*fb*fb)*pow(1.0+yd,-pc)
    *(1.0+0.5*pc*(1.0+1.0/(7.0*(3.0-4.0*pc)))/(1.0+yd));

  geff=omh2*(sqrt(alnu)+(1.0-sqrt(alnu))/(1.0+pow(0.43*k*sh,4)));
  qeff=k/geff*theta*theta;

  L=log(2.718281828+1.84*sqrt(alnu)*qeff/(1.0-0.949*fb));
  C=14.4+325.0/(1.0+60.5*pow(qeff,1.11));

  delk=pow(norm,2)*pow(k*2997.9/h,3.+ns)*pow(L/(L+C*qeff*qeff),2);
  pk=2.0*M_PI*M_PI/(k*k*k)*delk;

  return pow(h,3)*pk;
}


double sigmam(double r, int j)   // r[Mpc/h]
{
  int n,i,l;
  double k1,k2,xx,xxp,xxpp,k,a,b,hh,x;

  k1=2.*M_PI/r;
  k2=2.*M_PI/r;

  xxpp=-1.0;
  for(;;){
    k1=k1/10.0;
    k2=k2*2.0;

    a=log(k1),b=log(k2);
    
    xxp=-1.0;
    n=2;
    for(;;){
      n=n*2;
      hh=(b-a)/(double)n;

      xx=0.;
      for(i=1;i<n;i++){
	k=exp(a+hh*i);
	xx+=k*k*k*linear_pk(k)*pow(window(k*r,j),2);
      }
      xx+=0.5*(k1*k1*k1*linear_pk(k1)*pow(window(k1*r,j),2)+k2*k2*k2*linear_pk(k2)*pow(window(k2*r,j),2));
      xx*=hh;

      if(fabs((xx-xxp)/xx)<eps) break;
      xxp=xx; 
    }

    if(fabs((xx-xxpp)/xx)<eps) break;
    xxpp=xx;
  }

  return sqrt(xx/(2.0*M_PI*M_PI));
}


double window(double x, int i)
{
  if(i==0) return 3.0/pow(x,3)*(sin(x)-x*cos(x));  // top hat
  if(i==1) return exp(-0.5*x*x);   // gaussian
  if(i==2) return x*exp(-0.5*x*x);  // 1st derivative gaussian
}


double lgr(double z)  // linear growth factor at z (not normalized at z=0)
{
  int i,j,n;
  double a,a0,x,h,yp;
  double k1[2],k2[2],k3[2],k4[2],y[2],y2[2],y3[2],y4[2];

  a=1./(1.+z);
  a0=1./1100.;

  yp=-1.;
  n=10;

  for(;;){
    n*=2;
    h=(log(a)-log(a0))/n;
  
    x=log(a0);
    y[0]=1., y[1]=0.;
    for(i=0;i<n;i++){
      for(j=0;j<2;j++) k1[j]=h*lgr_func(j,x,y);

      for(j=0;j<2;j++) y2[j]=y[j]+0.5*k1[j];
      for(j=0;j<2;j++) k2[j]=h*lgr_func(j,x+0.5*h,y2);

      for(j=0;j<2;j++) y3[j]=y[j]+0.5*k2[j];
      for(j=0;j<2;j++) k3[j]=h*lgr_func(j,x+0.5*h,y3);

      for(j=0;j<2;j++) y4[j]=y[j]+k3[j];
      for(j=0;j<2;j++) k4[j]=h*lgr_func(j,x+h,y4);
      
      for(j=0;j<2;j++) y[j]+=(k1[j]+k4[j])/6.+(k2[j]+k3[j])/3.;
      x+=h;
    }

    if(fabs(y[0]/yp-1.)<0.1*eps) break;
    yp=y[0];
  }

  return a*y[0];
}


double lgr_func(int j, double la, double y[2])
{
  if(j==0) return y[1];
  
  double g,a;
  a=exp(la);
  g=-0.5*(5.*om+(5.-3*w)*ow*pow(a,-3.*w))*y[1]-1.5*(1.-w)*ow*pow(a,-3.*w)*y[0];
  g=g/(om+ow*pow(a,-3.*w));
  if(j==1) return g;
}
