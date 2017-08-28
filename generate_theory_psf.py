import numpy as np
import scipy.special as sp
import scipy.integrate as integrate
import math

###
# For reference, here are the rough LSST parameters that I used

# r0_500 = .2m
# L0s = [10,15,25,50,75,100] m
# lams = xrange(300, 1200, 100) nm
# diam = 8.0 m
###
PATH = dirname(realpath(__file__))+'/theory_PSFs/'

arcsec_per_rad = 648000./math.pi
diam0 = 8
lam0 = 500.*10**-9

args_def = dict(
r0_500_def      = 0.2   ,  
lam_def         = 700.0 ,   
diam_def        = 8.0   ,    
L0_def          = 10.0  ,          
alphas_def       = -1.0) 


def make_PSF(args):
    """ Outputs the value of the theoretical radial PSF (the theoretical psf is azimuthally symmetric) at the 
        provided radial alphas (in arcseconds). The normalization in this program is arbitrary 
        since I only used the FWHM of the radial PSF.
    """

    def phase_fluc(kappa,lam,L0, r0_500):
        r0 = r0_500*(lam/lam0)**(6./5.)
        return .033/.423*r0**(-5.0/3.0)*(kappa**2.+(2*math.pi/L0)**2.)**(-11.0/6.0)

    def struct_fn_integrand(kappa,p,lam,L0,r0_500):
        return phase_fluc(kappa,lam,L0,r0_500)*(1.-sp.jv(0,kappa*p))*kappa

    def struct_fn(p,lam,L0, r0_500):
        return 8*math.pi**2*integrate.quad(struct_fn_integrand,0.,np.inf,args=(p,lam,L0,r0_500))[0]

    def optical_transf_fn(p,diam):
        return 2./math.pi*(math.acos(p/diam)-p/diam*math.sqrt(1.-(p/diam)**2.))

    def intensity_integrand(p,alpha,lam,L0,diam, r0_500):
        return p*sp.jv(0,math.pi*2./lam*alpha*p)*optical_transf_fn(p,diam)*math.exp(-.5*struct_fn(p,lam,L0, r0_500))

    def expectedIntensity(alpha,lam,L0,diam,r0_500): #alpha here is in radians
        #The leading constant didn't really matter for me, so I tried to get the results to be O(1)
        return (1000*lam0**2/diam0**2)*math.pi*(diam**2.)/4.0/(lam**2.)*integrate.quad(intensity_integrand,0,diam, args=(alpha,lam,L0,diam,r0_500))[0]

    lam = args.lam
    L0 = args.L0
    r0_500 = args.r0_500
    diam = args.diam


    saved_theoretical_file = PATH+'theory_lam{}_L0{}_diam{}_r0_500{}.txt'.format(lam,L0,diam,r0_500)

    if(args.alphas < 0):
        alpha_max = .5*arcsec_per_rad*(.98/(r0_500*(300./500.)**(6./5.))*(300.*10**(-9)))   #set the maximum alpha to be 1/2 of the kolmogorov FWHM at 300nm
        alphas = np.arange(0.0,alpha_max+.005,.005)
    else:
        alphas = args.alphas

    with open(saved_theoretical_file, 'w+') as f:
        f.seek(0)
        for alpha in alphas:
            PSF_val= expectedIntensity(alpha/arcsec_per_rad,lam*10**(-9),L0,diam,r0_500)
            print alpha, PSF_val
            f.write("{}\n".format(PSF_val))
        f.truncate()



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
     
    parser.add_argument("--r0_500", type=float, default = args_def['r0_500_def'],
                        help="Fried parameter at wavelength 500 nm in meters.  Default: 0.2")
   
    parser.add_argument("--lam", type=float, default  = args_def['lam_def'], 
                        help="Wavelength in nanometers.  Default: 700.0")
     
    parser.add_argument("--diam", type=float, default = args_def['diam_def'],
                        help="Size of circular telescope pupil in meters.  Default: 8.0")

    parser.add_argument("--L0", type=float, default = args_def['L0_def'],
                        help="Outer length scale for generated atmosphere screens. "
                             "Typical range around 10-100  Default: 10")
             
    parser.add_argument("--alphas", type=float, nargs='+',  default  = args_def['alphas_def'], 
                        help="The angles in arcseconds off axis to generate a point of the radial PSF"
                             "Eg: np.linspace(0.0, .3, num=31)"
                             "Default np.arange(0.0,alpha_max+.005,.005), where alpha_max is set by Kolmogorov")

    args = parser.parse_args()
    make_PSF(args)