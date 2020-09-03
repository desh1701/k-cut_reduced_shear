! A code for computing the non-linear matter bispectrum (BS) based on BiHalofit (arXiv:1911.07886) 
! written by Ryuichi Takahashi in Dec 2019
! a bug fixed in Mar 2020 (Thanks Sven Heydenreich for indicating it) 
!
! Plese type "gfortran bihalofit.f90 -O2" to compile this
!
! Wave numbers (k1,k2,k3) and redshift are given in the follwing lines.
! line 55: z is a redshift
!      62: k1,k2,k3 are wavenumbers of triangle configulation in units of [h/Mpc]
!
! The default cosmological parameters are consistent with the Planck 2015 LambdaCDM model.
! If you want to change the cosmological model, please modify the following lines.
! lines 21-28: cosmological parameters, this code can be used for flat wCDM model (CDM + dark energy with a constant equation of state)
! line 43: table of linear P(k) (1st & 2nd columns are k[h/Mpc] & P(k)[(Mpc/h)^3]). 
!          if you give a P(k) table, then the code uses it.
!          if not, the code uses the Eisenstein & Hu (1999) fitting formula as the linear P(k).
! line 31: n_data_max should be larger than the number of lines in the P(k) table


module input_params
  !! cosmological parameters are given below
  real(8), parameter :: h=0.6727     ! Hubble parameter 
  real(8), parameter :: sigma8=0.831 ! sigma 8
  real(8), parameter :: omb=0.0492   ! Omega baryon
  real(8), parameter :: omc=0.2664   ! Omega CDM
  real(8), parameter :: ns=0.9645    ! spectral index of linear P(k) 
  real(8), parameter :: w=-1.d0      ! equation of state of dark energy
  real(8), parameter :: om=omb+omc   ! Omega matter
  real(8), parameter :: ow=1.d0-om   ! Omega dark energy
  real(8) :: norm,D1,n_eff,r_sigma

  integer, parameter :: n_data_max=1000
  real(8) :: k_data(n_data_max),pk_data(n_data_max)
  integer :: n_data=1
  
  real(8), parameter :: pi=4.d0*atan(1.d0)   
  real(8), parameter :: eps=1.e-4  ! fractional accuracy of BS computaion 
end module input_params

program bihalofit
  use input_params
  real(8) :: z,k1,k2,k3

  open(16, file='linear_pk_planck2015.txt', status='old', err=5)
  do
    read (16,*,iostat=ios) k_data(n_data),pk_data(n_data)
    if(ios/=0) exit
    n_data=n_data+1
  end do
  close(16)
5 n_data=n_data-1
  
  norm=1.
  norm=norm*sigma8/sigmam(8.d0,0)   ! P(k) amplitude normalized by sigma8
  
  z=0.4   ! redshift

  ! calculating D1, r_sigma & n_eff for a given redshift
  D1=lgr(z)/lgr(0.d0)
  r_sigma=calc_r_sigma(z)  ! =1/k_NL [Mpc/h] in Eq.(B1)
  n_eff=-3.+2.*(D1*sigmam(r_sigma,2))**2   ! n_eff in Eq.(B2)

  k1=1.5; k2=2.; k3=1.    ! [h/Mpc]

  ! non-linear BS w/o baryons, tree-level BS [(Mpc/h)^6] & baryon ratio for given k1,k2,k3 and z 
  write(*,*) k1,k2,k3,bispec(k1,k2,k3,z),bispec_tree(k1,k2,k3,z),baryon_ratio(k1,k2,k3,z)
    
contains


real(8) function bispec(k1,k2,k3,z)   ! non-linear BS [(Mpc/h)^6] w/o baryons
  use input_params
  implicit none
  integer i,j
  real(8) k1,k2,k3,z,q(3),k(3),kt,logsigma8z,r1,r2
  real(8) an,bn,cn,en,fn,gn,hn,mn,nn,pn,alphan,betan,mun,nun,BS1h,BS3h,PSE(3)

  if(z>10.) then
    bispec=bispec_tree(k1,k2,k3,z)
  else   
    q(1)=k1*r_sigma; q(2)=k2*r_sigma; q(3)=k3*r_sigma   ! dimensionless wavenumbers

    k(1)=k1; k(2)=k2; k(3)=k3 
    ! sorting k(i) so that k(1)>=k(2)>=k(3)
    do i=1,3
      do j=i+1,3
        if(k(i)<k(j)) then
	  kt=k(j)
          k(j)=k(i)
	  k(i)=kt
        end if
      end do
    end do 
    r1=k(3)/k(1); r2=(k(2)+k(3)-k(1))/k(1)   ! Eq.(B8)
    if(k(1)>k(2)+k(3)) write(*,*) "Error: triangle is not formed"

    logsigma8z=log10(D1*sigma8)
  
    ! 1-halo term parameters in Eq.(B7)
    an=10.d0**(-2.167-2.944*logsigma8z-1.106*logsigma8z**2-2.865*logsigma8z**3-0.310*r1**(10.**(0.182+0.57*n_eff)))
    bn=10.d0**(-3.428-2.681*logsigma8z+1.624*logsigma8z**2-0.095*logsigma8z**3)
    cn=10.d0**(0.159-1.107*n_eff)
    alphan=10.d0**(-4.348-3.006*n_eff-0.5745*n_eff**2+10.**(-0.9+0.2*n_eff)*r2*r2)
    if(alphan>1.-(2./3.)*ns) alphan=1.-(2./3.)*ns
    betan=10.d0**(-1.731-2.845*n_eff-1.4995*n_eff**2-0.2811*n_eff**3+0.007*r2)

    ! 1-halo term bispectrum in Eq.(B4)
    BS1h=1.
    do i=1,3
      BS1h=BS1h/(an*q(i)**alphan+bn*q(i)**betan)/(1.+1./(cn*q(i)))
    end do
  
    ! 3-halo term parameters in Eq.(B9)
    fn=10.d0**(-10.533-16.838*n_eff-9.3048*n_eff**2-1.8263*n_eff**3)
    gn=10.d0**(2.787+2.405*n_eff+0.4577*n_eff**2)
    hn=10.d0**(-1.118-0.394*n_eff)
    mn=10.d0**(-2.605-2.434*logsigma8z+5.71*logsigma8z**2)
    nn=10.d0**(-4.468-3.08*logsigma8z+1.035*logsigma8z**2)
    mun=10.d0**(15.312+22.977*n_eff+10.9579*n_eff**2+1.6586*n_eff**3)
    nun=10.d0**(1.347+1.246*n_eff+0.4525*n_eff**2)
    pn=10.d0**(0.071-0.433*n_eff)
    en=10.d0**(-0.632+0.646*n_eff)

    do i=1,3
      PSE(i)=(1.+fn*q(i)**2)/(1.+gn*q(i)+hn*q(i)**2)*D1*D1*linear_pk(q(i)/r_sigma) &
             +1./(mn*q(i)**mun+nn*q(i)**nun)/(1.+(pn*q(i))**(-3))  ! enhanced P(k) in Eq.(B6) 
    end do

    ! 3-halo term bispectrum in Eq.(B5)
    BS3h=2.*(F2(k1,k2,k3,z)*PSE(1)*PSE(2)+F2(k2,k3,k1,z)*PSE(2)*PSE(3)+F2(k3,k1,k2,z)*PSE(3)*PSE(1))
    do i=1,3
      BS3h=BS3h/(1.+en*q(i))
    end do   
 
    bispec=BS1h+BS3h
  end if
end function 

  
real(8) function bispec_tree(k1,k2,k3,z)  ! tree-level BS [(Mpc/h)^6]
  use input_params  
  real(8) k1,k2,k3,z
  bispec_tree=D1**4*2.*(F2_tree(k1,k2,k3)*linear_pk(k1)*linear_pk(k2) &
	 	       +F2_tree(k2,k3,k1)*linear_pk(k2)*linear_pk(k3) &
                       +F2_tree(k3,k1,k2)*linear_pk(k3)*linear_pk(k1))
end function
  

real(8) function F2(k1,k2,k3,z)
  use input_params
  implicit none
  real(8) k1,k2,k3,z,a,q3,dn,omz,logsigma8z

  q3=k3*r_sigma
  
  logsigma8z=log10(D1*sigma8)
  a=1./(1.+z)
  omz=om/(om+ow*a**(-3.*w))   ! Omega matter at z

  dn=10.d0**(-0.483+0.892*logsigma8z-0.086*omz)

  F2=F2_tree(k1,k2,k3)+dn*q3
end function
  

real(8) function F2_tree(k1,k2,k3)  ! F2 kernel in tree level 
  real(8) k1,k2,k3,costheta12
  costheta12=0.5*(k3*k3-k1*k1-k2*k2)/(k1*k2)
  F2_tree=(5./7.)+0.5*costheta12*(k1/k2+k2/k1)+(2./7.)*costheta12*costheta12
end function  

  
real(8) function baryon_ratio(k1,k2,k3,z)   ! bispectrum ratio with to without baryons  // k[h/Mpc]
  implicit none
  integer i
  real(8) a,A0,A1,mu0,mu1,sigma0,sigma1,alpha0,alpha2,beta2,ks,k(3),x(3)
  real(8) Rb,k1,k2,k3,z

  if(z>5.) then  ! baryon_ratio is calbrated at z=0-5   
    Rb=1.0
  else   
    a=1./(1.+z)
    k(1)=k1; k(2)=k2; k(3)=k3
    do i=1,3
      x(i)=log10(k(i))
    end do

    A0=0.
    if(a>0.5) A0=0.068*(a-0.5)**0.47
    mu0=0.018*a+0.837*a*a
    sigma0=0.881*mu0
    alpha0=2.346

    A1=0.
    if(a>0.2) A1=1.052*(a-0.2)**1.41
    mu1=abs(0.172+3.048*a-0.675*a*a)
    sigma1=(0.494-0.039*a)*mu1
  
    ks=29.90-38.73*a+24.30*a*a
    alpha2=2.25
    beta2=0.563/((a/0.060)**0.02+1.)/alpha2;
    
    Rb=1.;
    do i=1,3
      Rb=Rb*(A0*exp(-(abs(x(i)-mu0)/sigma0)**alpha0)-A1*exp(-(abs(x(i)-mu1)/sigma1)**2)+(1.+(k(i)/ks)**alpha2)**beta2)   ! Eq.(C1)
    end do
  end if
 
  baryon_ratio=Rb
end function


real(8) function calc_r_sigma(z)  ! return r_sigma[Mpc/h] (=1/k_sigma)
  use input_params
  implicit none
  real(8) k,k1,k2,z

  k1=1.d0; k2=1.d0
  do
    if(D1*sigmam(1./k1,1)<1.d0) exit
    k1=k1*0.5
  end do
  do
    if(D1*sigmam(1./k2,1)>1.d0) exit
    k2=k2*2.
  end do

  do
    k=0.5d0*(k1+k2)
    if(D1*sigmam(1./k,1)<1.d0) then
      k1=k 
    else if(D1*sigmam(1./k,1)>1.d0) then
      k2=k
    end if
    if(D1*sigmam(1./k,1)==1.d0 .or. abs(k2/k1-1.d0)<eps*0.1) exit
  end do

  calc_r_sigma=1./k;
end function  
  
  
real(8) function linear_pk(k)  ! linear P(k)   k[h/Mpc], P(k)[(Mpc/h)^3]
  real(8) k
  if(n_data/=0) then     
     linear_pk=linear_pk_data(k)
  else
     linear_pk=linear_pk_eh(k)
  end if
end function
  
     
real(8) function linear_pk_data(k)   ! linear P(k) interpolated from the given table,  k[h/Mpc]  P(k)[(Mpc/h)^3]
  implicit none
  integer j,j1,j2,jm
  real(8) lk,dlk,f,k

  lk=log10(k);
  if(k<k_data(1)) then
    f=0.
  else if(k>k_data(n_data)) then
    f=0.
  else
    j1=1; j2=n_data; jm=(j1+j2)/2
    do
      if(k>k_data(jm)) then
        j1=jm
      else 
        j2=jm
      end if
      jm=(j1+j2)/2;

      if(j2-j1==1) exit
    end do
    j=j1;
    f=(log10(pk_data(j+1))-log10(pk_data(j)))/(log10(k_data(j+1))-log10(k_data(j)))*(lk-log10(k_data(j)))+log10(pk_data(j));
  end if
    
  linear_pk_data=norm*norm*10.**f;  
end function


real(8) function linear_pk_eh(k)   ! Eisenstein & Hu (1999) fitting formula without wiggle,      k[h/Mpc], P(k)[(Mpc/h)^3]
  use input_params
  implicit none
  real(8) pk,delk,alnu,geff,qeff,L,C,k
  real(8) fc,fb,theta,pc,omh2,ombh2,zeq,b1,b2,zd,yd,sh
  k=k*h   ! unit conversion from [h/Mpc] to [1/Mpc]
  
  fc=omc/om
  fb=omb/om
  theta=2.728/2.7
  pc=0.25*(5.0-sqrt(1.0+24.0*fc))
  omh2=om*h*h
  ombh2=omb*h*h
  zeq=2.5e+4*omh2/theta**4
  b1=0.313*omh2**(-0.419)*(1.0+0.607*omh2**(0.674))
  b2=0.238*omh2**(0.223)
  zd=1291.0*omh2**(0.251)/(1.0+0.659*omh2**(0.828))*(1.0+b1*ombh2**b2)
  yd=(1.0+zeq)/(1.0+zd)
  sh=44.5*log(9.83/(omh2))/sqrt(1.0+10.0*ombh2**(0.75))

  alnu=fc*(5.0-2.0*pc)/5.0*(1.0-0.553*fb+0.126*fb*fb*fb)*(1.0+yd)**(-pc)*(1.0+0.5*pc*(1.0+1.0/(7.0*(3.0-4.0*pc)))/(1.0+yd))

  geff=omh2*(sqrt(alnu)+(1.0-sqrt(alnu))/(1.0+(0.43*k*sh)**4))
  qeff=k/geff*theta*theta

  L=log(2.718281828+1.84*sqrt(alnu)*qeff/(1.0-0.949*fb))
  C=14.4+325.0/(1.0+60.5*qeff**(1.11))

  delk=norm**2*(k*2997.9/h)**(3.+ns)*(L/(L+C*qeff*qeff))**2
  pk=2.0*pi*pi/(k*k*k)*delk

  k=k/h   ! unit conversion again from [1/Mpc] to [h/Mpc]
  linear_pk_eh=pk*h**3
end function


real(8) function sigmam(r,j)   ! r[Mpc/h]
  use input_params
  implicit none
  integer :: j,n,i
  real(8) :: r,k1,k2,xx,xxp,xxpp,k,a,b,hh

  k1=2.*pi/r;  k2=2.*pi/r

  xxpp=-1.0
  do 
    k1=k1/10.0;  k2=k2*2.0

    a=log(k1); b=log(k2)
    
    xxp=-1.d0
    n=2
    do 
      n=n*2
      hh=(b-a)/dble(n)

      xx=0.d0
      do i=1,n-1
	k=exp(a+hh*dble(i))
	xx=xx+k*k*k*linear_pk(k)*window(k*r,j)**2
      end do
      xx=xx+0.5*(k1*k1*k1*linear_pk(k1)*window(k1*r,j)**2+k2*k2*k2*linear_pk(k2)*window(k2*r,j)**2)
      xx=xx*hh

      if(abs((xx-xxp)/xx)<eps) exit
      xxp=xx
    end do

    if(abs((xx-xxpp)/xx)<eps) exit
    xxpp=xx
  end do

  sigmam=sqrt(xx/(2.d0*pi*pi));
end function


real(8) function window(x,i)
  implicit none
  integer :: i
  real(8) :: x

  if(i==0) window=3.d0/(x**3)*(sin(x)-x*cos(x))  ! top hat
  if(i==1) window=exp(-0.5d0*x*x)    ! gaussian
  if(i==2) window=x*exp(-0.5d0*x*x)   ! 1st derivative gaussian
end function
  
  
real(8) function lgr(z)
  use input_params
  implicit none
  integer :: i,j,n
  real(8) :: z,a,a0,x,hh,yp
  real(8) :: k1(2),k2(2),k3(2),k4(2),y(2),y2(2),y3(2),y4(2)

  a=1./(1.+z); a0=1./1100.
  yp=-1.; n=10

  do
    n=2*n
    hh=(log(a)-log(a0))/dble(n)

    x=log(a0)
    y(1)=1.d0; y(2)=0.d0
    do i=1,n
      k1(1)=hh*lgr_func(1,x,y); k1(2)=hh*lgr_func(2,x,y)

      y2(1)=y(1)+0.5d0*k1(1); y2(2)=y(2)+0.5d0*k1(2)
      k2(1)=hh*lgr_func(1,x+0.5d0*hh,y2); k2(2)=hh*lgr_func(2,x+0.5d0*hh,y2)

      y3(1)=y(1)+0.5d0*k2(1); y3(2)=y(2)+0.5d0*k2(2)
      k3(1)=hh*lgr_func(1,x+0.5d0*hh,y3); k3(2)=hh*lgr_func(2,x+0.5d0*hh,y3)

      y4(1)=y(1)+k3(1); y4(2)=y(2)+k3(2)
      k4(1)=hh*lgr_func(1,x+hh,y4); k4(2)=hh*lgr_func(2,x+hh,y4)

      y(1)=y(1)+(k1(1)+k4(1))/6.d0+(k2(1)+k3(1))/3.d0; y(2)=y(2)+(k1(2)+k4(2))/6.d0+(k2(2)+k3(2))/3.d0
      x=x+hh;
    end do

    if(abs(y(1)/yp-1.d0)<0.1*eps) exit
    yp=y(1)
  end do

  lgr=a*y(1)
end function 


real(8) function lgr_func(j,la,y)
  use input_params
  implicit none
  integer :: j
  real(8) :: x,la,y(2) 

  if(j==1) then
    lgr_func=y(2)
  else
    x=ow*exp(la)**(-3.d0*w)
    lgr_func=(-0.5d0*(5.d0*om+(5.d0-3.d0*w)*x)*y(2)-1.5d0*(1.d0-w)*x*y(1))/(om+x)
  end if
 
end function

end program 
