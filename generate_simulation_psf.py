# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
###################
#
# Modified by Kyle Gulshen
#
###################

import time
import numpy as np
import galsim

args_def = dict(
seed        = 1     ,        
r0_500      = 0.2   ,      
nlayers     = 6     ,     
lam         = 700.0 ,        
time_step   = .03   ,     
exptime     = 15.0  ,       
pause_exp   = False ,    
x           = 0.0   ,           
y           = 0.0   ,           
psf_nx      = 512   ,      
psf_scale   = 0.005 ,  
diam        = 8.0   ,        
obscuration = 0.0   , 
nstruts     = 0     ,     
strut_thick = 0.05  ,
strut_angle = 0.0   ,
screen_size = 102.4 ,
screen_scale = 0.1  ,
pad_factor  = 1.0   ,  
oversampling = 1.0  ,
L0          = 10.0  ,          
max_speed   = 20.0  ,
take_snapshots = -1.)

def make_PSF(args):
    """Save image of PSF given command line arguments stored in `args`.
    """

    # Initiate some GalSim random number generators.
    rng = galsim.BaseDeviate(args.seed)
    u = galsim.UniformDeviate(rng)

    # The GalSim atmospheric simulation code describes turbulence in the 3D atmosphere as a series
    # of 2D turbulent screens.  The galsim.Atmosphere() helper function is useful for constructing
    # this screen list.

    # First, we estimate a weight for each screen, so that the turbulence is dominated by the lower
    # layers consistent with direct measurements.  The specific values we use are from SCIDAR
    # measurements on Cerro Pachon as part of the 1998 Gemini site selection process
    # (Ellerbroek 2002, JOSA Vol 19 No 9).

    Ellerbroek_alts = [0.0, 2.58, 5.16, 7.73, 12.89, 15.46]  # km
    Ellerbroek_weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
    Ellerbroek_interp = galsim.LookupTable(Ellerbroek_alts, Ellerbroek_weights,
                                           interpolant='linear')

    # Use given number of uniformly spaced altitudes
    alts = np.max(Ellerbroek_alts)*np.arange(args.nlayers)/(args.nlayers-1)
    weights = Ellerbroek_interp(alts)  # interpolate the weights
    weights /= sum(weights)  # and renormalize

    # Each layer can have its own turbulence strength (roughly inversely proportional to the Fried
    # parameter r0), wind speed, wind direction, altitude, and even size and scale (though note that
    # the size of each screen is actually made infinite by "wrapping" the edges of the screen.)  The
    # galsim.Atmosphere helper function is useful for constructing this list, and requires lists of
    # parameters for the different layers.

    max_speed = args.max_speed # an arbitrary maximum wind speed in m/s.
    spd = []  # Wind speed in m/s
    dirn = [] # Wind direction in radians
    r0_500 = [] # Fried parameter in m at a wavelength of 500 nm.
    for i in range(args.nlayers):
        spd.append(u()*max_speed)  # Use a random speed between 0 and max_speed
        dirn.append(u()*360*galsim.degrees)  # And an isotropically distributed wind direction.

        # The turbulence strength of each layer is specified by through its Fried parameter r0_500,
        # which can be thought of as the diameter of a telescope for which atmospheric turbulence
        # and unaberrated diffraction contribute equally to image resolution (at a wavelength of
        # 500nm).  The weights above are for the refractive index structure function (similar to a
        # variance or covariance), however, so we need to use an appropriate scaling relation to
        # distribute the input "net" Fried parameter into a Fried parameter for each layer.  For
        # Kolmogorov turbulence, this is r0_500 ~ (structure function)**(-3/5):
        r0_500.append(args.r0_500*weights[i]**(-3./5))
        print ("Adding layer at altitude {:5.2f} km with velocity ({:5.2f}, {:5.2f}) m/s, "
               "and r0_500 {:5.3f} m."
               .format(alts[i], spd[i]*dirn[i].cos(), spd[i]*dirn[i].sin(), r0_500[i]))

    atmt0 = time.time()

    # Additionally, we set the screen temporal evolution `time_step`, and the screen size and scale.
    atm = galsim.Atmosphere(r0_500=r0_500, speed=spd, direction=dirn, altitude=alts, rng=rng,
                            time_step=args.time_step, screen_size=args.screen_size,
                            screen_scale=args.screen_scale, L0 = args.L0)
    # `atm` is now an instance of a galsim.PhaseScreenList object.

    print("atmosphere generation time: {}".format(time.time()-atmt0))


    # Field angle (angle on the sky wrt the telescope boresight) at which to compute the PSF.
    theta = (args.x*galsim.arcmin, args.y*galsim.arcmin)

    # Construct an Aperture object for computing the PSF.  The Aperture object describes the
    # illumination pattern of the telescope pupil, and chooses good sampling size and resolution
    # for representing this pattern as an array.
    aper = galsim.Aperture(diam=args.diam, lam=args.lam, obscuration=args.obscuration,
                           nstruts=args.nstruts, strut_thick=args.strut_thick,
                           strut_angle=args.strut_angle*galsim.degrees,
                           screen_list=atm, pad_factor=args.pad_factor,
                           oversampling=args.oversampling)

    # generate unique filename with lam, seed, L0 and any other non-default parameters listed, seperated by _
    file_name = "lam"+str(args.lam) + "_seed"+str(args.seed) + "_L0"+str(args.L0)   #always save these three args
    for attr, value in args_def.iteritems():
        if(args.__dict__[attr] != value and attr!="lam" and attr!="seed" and attr!="L0"): #if there is a non-defualt arg, include in filename
            file_name += "_" + attr + str(args.__dict__[attr])

    # make psf and report time taken
    psft0 = time.time()

    if args.take_snapshots<=.00: #by default do not save multiple consecutive snapshots of exposure, just save one exposure with specified pause time
        if args.pause_exp:
            psf = atm.makePSF(lam=args.lam, theta=theta, aper=aper, exptime=args.exptime/2.0)
            atm.makePSF(lam=args.lam, theta=theta, aper=aper, exptime=3.0)  #run the simulation for three seconds without contributing to psf
            psf += atm.makePSF(lam=args.lam, theta=theta, aper=aper, exptime=args.exptime/2.0)
            psf = psf/2 #renormalize
        else:
            psf = atm.makePSF(lam=args.lam, theta=theta, aper=aper, exptime=args.exptime)
    else: #save consecutive snapshots of exposure
        numsnapshots = int(args.exptime/args.take_snapshots)
        excess = args.exptime%args.take_snapshots
        num=1

        psf = atm.makePSF(lam=args.lam, theta=theta, aper=aper, exptime=args.take_snapshots)
        cumulative_psf = psf
        psf_img = psf.drawImage(nx=args.psf_nx, ny=args.psf_nx, scale=args.psf_scale)
        psf_img.write(file_name+"_snap{}.fits".format(num))

        for _ in range(numsnapshots-1):
            num = num+1
            psf = atm.makePSF(lam=args.lam, theta=theta, aper=aper, exptime=args.take_snapshots)
            cumulative_psf+=psf
            psf_img = psf.drawImage(nx=args.psf_nx, ny=args.psf_nx, scale=args.psf_scale)
            psf_img.write(file_name+"_snap{}.fits".format(num)) #save each snapshot

        if(excess>.001): #take one more exposure if the total exposure time is not an integer multiple of snapshot time
            psf = atm.makePSF(lam=args.lam, theta=theta, aper=aper, exptime=excess)
            cumulative_psf+=psf
            psf_img = psf.drawImage(nx=args.psf_nx, ny=args.psf_nx, scale=args.psf_scale)
            psf_img.write(file_name+"_snap{}.fits".format(num+1)) 

        psf = cumulative_psf
        file_name = file_name + "_snap_total"

    print("PSF time: {}".format(time.time()-psft0))

    #save image
    file_name = file_name +".fits"
    psf_img = psf.drawImage(nx=args.psf_nx, ny=args.psf_nx, scale=args.psf_scale)
    psf_img.write(file_name)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
     
    parser.add_argument("--seed", type=int, default = args_def['seed'],
                        help="Random number seed for generating turbulence.  Default: 1")
     
    parser.add_argument("--r0_500", type=float, default = args_def['r0_500'],
                        help="Fried parameter at wavelength 500 nm in meters.  Default: 0.2")
         
    parser.add_argument("--nlayers", type=int, default = args_def['nlayers'],
                        help="Number of atmospheric layers.  Default: 6")
     
    parser.add_argument("--lam", type=float, default  = args_def['lam'], 
                        help="Wavelength in nanometers.  Default: 700.0")
     
    parser.add_argument("--time_step", type=float, default  = args_def['time_step'], 
                        help="Incremental time step for advancing phase screens and accumulating "
                             "instantaneous PSFs in seconds.  Default: 0.03")
     
    parser.add_argument("--exptime", type=float, default  = args_def['exptime'], 
                        help="Total amount of time to integrate in seconds.  Default: 15.0")
     
    parser.add_argument("--pause_exp", action='store_true',
                        help="Add a 3 second delay in the middle of the exposure.  Default: false")
     
    parser.add_argument("-x", "--x", type=float, default  = args_def['x'], 
                        help="x-coordinate of PSF in arcmin.  Default: 0.0")
     
    parser.add_argument("-y", "--y", type=float, default  = args_def['y'], 
                        help="y-coordinate of PSF in arcmin.  Default: 0.0")
         
    parser.add_argument("--psf_nx", type=int, default = args_def['psf_nx'],
                        help="Output PSF image dimensions in pixels.  Default: 512")
             
    parser.add_argument("--psf_scale", type=float, default = args_def['psf_scale'],
                        help="Scale of PSF output pixels in arcseconds.  Default: 0.005")
         
    parser.add_argument("--diam", type=float, default = args_def['diam'],
                        help="Size of circular telescope pupil in meters.  Default: 8.0")
         
    parser.add_argument("--obscuration", type=float, default = args_def['obscuration'],
                        help="Linear fractional obscuration of telescope pupil.  Default: 0.0")
         
    parser.add_argument("--nstruts", type=int, default = args_def['nstruts'],
                        help="Number of struts supporting secondary obscuration.  Default: 0")
             
    parser.add_argument("--strut_thick", type=float, default = args_def['strut_thick'], 
                        help="Thickness of struts as fraction of aperture diameter.  Default: 0.05")
             
    parser.add_argument("--strut_angle", type=float, default = args_def['strut_angle'], 
                        help="Starting angle of first strut in degrees.  Default: 0.0")
         
    parser.add_argument("--screen_size", type=float, default = args_def['screen_size'],  
                        help="Size of atmospheric screen in meters.  Note that the screen wraps "
                             "with periodic boundary conditions.  Default: 102.4")
             
    parser.add_argument("--screen_scale", type=float, default = args_def['screen_scale'],
                        help="Resolution of atmospheric screen in meters.  Default: 0.1")
             
    parser.add_argument("--pad_factor", type=float, default = args_def['pad_factor'],
                        help="Factor by which to pad PSF InterpolatedImage to avoid aliasing. "
                             "Default: 1.0")
         
    parser.add_argument("--oversampling", type=float, default  = args_def['oversampling'], 
                        help="Factor by which to oversample the PSF InterpolatedImage. "
                             "Default: 1.0")
         
    parser.add_argument("--L0", type=float, default = args_def['L0'],
                        help="Outer length scale for generated atmosphere screens. "
                             "Typical range around 10-100  Default: 10")
             
    parser.add_argument("--max_speed", type=float, default  = args_def['max_speed'], 
                        help="Maximum windspeed for generated atmosphere screens. "
                             "Somewhat arbitrary  Default: 20")

    parser.add_argument("--take_snapshots", type=float, default  = args_def['take_snapshots'], 
                        help="Save consecutive x-second snapshots of the exposure. "
                             "Default: -1.0, no snapshots taken")

    args = parser.parse_args()
    make_PSF(args)
