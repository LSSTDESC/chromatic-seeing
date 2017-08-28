import sys
from os import listdir
from os.path import isfile, join, dirname, realpath
import math
import galsim
import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rand
import matplotlib.patches as mpatches
import itertools
import re

PATH = dirname(realpath(__file__))+'/'

sim_args_def = dict(
seed        = 1     ,	# seed of random number generator for both wind and phase screen generation   
r0_500      = 0.2   ,	# fried parameter in meters at lam=500nm
nlayers     = 6     ,   # number of phase screens to generate
lam         = 700.0 ,	# wavelength in nm 
time_step   = .03   ,	# time resolution for which to advance phase screen and generate psf     
exptime     = 15.0  ,	# total exposure time in seconds       
pause_exp   = False ,	# if true, pause for 3.0s in middle of exposure
x           = 0.0   ,	# x coordinate in focal plane for which to generate the psf           
y           = 0.0   ,	# y coordinate in focal plane for which to generate the psf           
psf_nx      = 512   ,	# number of pixels in one dimension    
psf_scale   = 0.005 ,   # arcseconds per pixel
diam        = 8.0   ,	# diameter of telescope aperature 
obscuration = 0.0   ,	# linear fractional obscuration of telescope pupil.	 
nstruts     = 0     ,	# number of struts supporting secondary obscuration. 
strut_thick = 0.05  ,	# thickness of struts as fraction of aperture diameter.
strut_angle = 0.0   ,	# starting angle of first strut in degrees.
screen_size = 102.4 ,	# size of atmospheric screen in meters. Note that the screen wraps around with periodic boundary conditions
screen_scale = 0.1  ,	# resolution of atmospheric screen in meters. 
pad_factor  = 1.0   ,	# factor by which to pad PSF InterpolatedImage to avoid aliasing.
oversampling = 1.0  ,	# factor by which to oversample the PSF InterpolatedImage.
L0          = 10.0  ,	# outer length scale for generated atmosphere screens.        
max_speed   = 20.0  ,	# maximum windspeed for generated atmosphere screens. 
take_snapshots = -1.) 	# save consecutive x second snapshots of the exposure. 

arcsec_per_rad = 648000./math.pi
sigma_def       = 300 # in arcseconds. You may want to tune this--I did not test it

#some color dicts I used. 
L0_colors = {4:'r', 6:'orange', 8:'gold', 10:'g', 15:'c' , 25:'b', 50:'purple', 75:'m', 100:'black', 1000: 'red', 1000000: 'green', 1000000000: 'blue', np.inf:'grey'}
other_L0_colors = {20.:'r', 30.:'orange', 40.:'gold', 50.:'g', 60.:'c', 70.:'b', 80.: 'purple', 90.: 'm', 100.:'black', np.inf:'grey'}
lam_colors = {1100.0:'r', 1000.0:'orange', 900.0 :'gold', 800.0: 'greenyellow', 700.0 :'g', 600.0 :'c', 500.0 :'b', 400.0 :'purple', 300.0 :'m'}
seed_colors = { 1:'blue', 2:'orange', 3:'green', 4:'greenyellow', 5:'gold', 6:'c', 7:'red', 8:'navy', 9:'m', 10:'grey', 11:'black', 12:'#7f0000', 13: '#7f7f00', 14: '#007f00', 15: '#00007f'} 
colors = {(25.0,30.0):'red',(100.0,30.0):'blue',(25.0,300.0):'#7f0000',(100.0,300.0):'#00007f'}
band_colors = {'a':'purple', 'g':'blue', 'r':'green', 'i':'gold', 'z':'magenta', 'y':'red'}

def simple_moments(img):  ####NOTE: this assumes a square image
    """Compute unweighted 0th, 1st, and 2nd moments of image (units should be arcseconds).  Return result as a dictionary.
    """
    array = img.array
    scale = img.scale

    # create x,y coords with 0,0 at center of middle pixel.
    x = y = (np.arange(array.shape[0])-array.shape[0]/2.+.5) * scale # if you want pixels instead, remove *scale
    y, x = np.meshgrid(y, x)

    I0 = np.sum(array)
    Ix = np.sum(x*array)/I0
    Iy = np.sum(y*array)/I0
    Ixx = np.sum((x-Ix)**2*array)/I0
    Iyy = np.sum((y-Iy)**2*array)/I0
    Ixy = np.sum((x-Ix)*(y-Iy)*array)/I0

    return dict(I0=I0, Ix=Ix, Iy=Iy, Ixx=Ixx, Iyy=Iyy, Ixy=Ixy)

def weighted_moments(img, sigma): ####NOTE: this assumes a square image
    """Compute weighted 0th, 1st, and 2nd moments of image.  Return result as a dictionary.
    """
    array = arrayg.array
    scale = img.scale
    # create x,y coords with 0,0 at center of middle pixel.
    x = y = (np.arange(array.shape[0])-array.shape[0]/2.+.5) * scale #if you want pixels instead, remove *scale
    y, x = np.meshgrid(y, x)

    # Get initial centroid estimate from unweighted first moments.
    I0 = np.sum(array)
    Ix = np.sum(array*x)/I0
    Iy = np.sum(array*y)/I0

    # Define weight fn wrt. unweighted first moments
    w = np.exp(-0.5*((x-Ix)**2 + (y-Iy)**2) / sigma**2)

    # Now recompute moments using weight and then recompute weight
    I0 = np.sum(array * w)
    Ix = np.sum(array * w * x)/I0
    Iy = np.sum(array * w * y)/I0
    w = np.exp(-0.5*((x-Ix)**2 + (y-Iy)**2) / sigma**2)

    # And finally, compute weighted second moments
    Ixx = np.sum(array * w * (x-Ix) * (x-Ix))/I0
    Ixy = np.sum(array * w * (x-Ix) * (y-Iy))/I0
    Iyy = np.sum(array * w * (y-Iy) * (y-Iy))/I0
    return dict(I0=I0, Ix=Ix, Iy=Iy, Ixx=Ixx, Iyy=Iyy, Ixy=Ixy)

def ellip(mom):
    """Convert moments dictionary into dictionary with ellipticity (e1, e2) and size (rsqr).
    """
    rsqr = mom['Ixx'] + mom['Iyy']
    return dict(rsqr=rsqr, e1=(mom['Ixx']-mom['Iyy'])/rsqr, e2=2*mom['Ixy']/rsqr)

def get_saved_radial(saved_radial_file, data_entry_name, Ix, Iy, alphas, image_array):
	"""read in simulated saved radial PSF. The original PSFs were integrated via montecarlo method
		along radial points in the range of np.linspace(0.0,.3,31) arcseconds.
	"""
	with open(saved_radial_file, 'r') as f:
	    lines = f.read()
	    data = re.findall(r'{}\[(.*?)\]'.format(data_entry_name),lines,re.DOTALL)

	    if len(data) == 0:
			print data_entry_name+" radial not found, generating radial now."
			radial = generate_monte_radial_function(Ix, Iy, alphas, image_array)
			with open(saved_radial_file, 'a') as f:
				# append to the save file the complete data entry name and array of radial values at the alpha values you provided
				f.write(data_entry_name+"{}\n".format(radial)) 
	    else:
	        radial = map(float, data[0].split())
	return radial

def generate_monte_radial_function(x_center, y_center, sample_alphas, im_array):
	""" Average the values of the given image file around a circle at the various sample_alphas which extend radially from given x,y center.
		This uses a montecarlo method, i.e. randomly generates points in an area whose length is 2pi and height is the value of PSF at the centroid,
		computes the fraction of points less than the function (im_array) in this area and multiplies the result by the maximum value of the function.
		This provides a 'radial PSF' that is essentially an average of cross sections intersecting the centroid at different angles.
		Hence this radial PSF can be compared with saved theoretical results.
	"""
	
	maximum = np.amax(im_array)
	num_iterations = 100000

	result = [] 

	for alpha in sample_alphas:
		integral = 0
		for theta in rand.random(num_iterations)*2*np.pi:
			x = int(round((alpha * math.cos(theta) + x_center)/sim_args_def['psf_scale']+im_array.shape[0]/2.-.5))
			y = int(round((alpha * math.sin(theta) + y_center)/sim_args_def['psf_scale']+im_array.shape[0]/2.-.5))
			
			if(im_array[x,y]>rand.random()*maximum):
				integral += 1.

		result = np.append(result,integral/num_iterations*maximum)

	return result

def get_FWHM(radial,alphas,avg_init=True): ####NOTE: By default this averages the first three points to use as maximum
    """return the FWHM of the provided radial PSF at the sampling points, alphas
    """
    if(avg_init):
    	initial = (radial[0]+ radial[1]+radial[2])/3
    else:
    	initial = radial[0]

    fwhm = 2*np.interp(.5*initial,list(reversed(radial)),list(reversed(alphas)))
    if(np.amin(radial)>.5*initial):
    	print "Error: FWHM is poorly approximated, as it exceeds simulated range."
    return fwhm

def get_saved_theoretical(lam,L0,diam,r0_500):
    """read in saved simulated theoretical radial PSFs for given lam and L0
    """

    saved_theoretical_file = PATH+'theory_PSFs/theory_lam{}_L0{}_diam{}_r0_500{}.txt'.format(lam,L0,diam,r0_500)

    data = ""
    
    with open(saved_theoretical_file, 'r') as f:
    	for lines in f.readlines():
    		data+=lines
   	
   	theoretical = map(float, data.split())

    return theoretical

def plot_theory_loglogslope_vs_L0(alphas,L0s,diam,r0_500,band):
	band_ranges = {'u':(300.,400.), 'g':(400.,500.), 'r':(500.,700.), 'i':(700.,800.), 'z':(800.,900.), 'y':(900.,1000.)}
	lam1 = band_ranges[band][0]
	lam2 = band_ranges[band][1]

	fwhm_slope = []
	for L0 in L0s:
		if(L0 == np.inf):
			plt.plot(L0s,[-.2]*len(L0s), lw=4, ls = '-', marker = 'o', color = 'grey')
			continue
		theoretical1 = get_saved_theoretical(lam1,L0,diam,r0_500)
		theoretical2 = get_saved_theoretical(lam2,L0,diam,r0_500)
		loglogslope = (np.log(get_FWHM(theoretical2,alphas))-np.log(get_FWHM(theoretical1,alphas)))/(np.log(lam2)-np.log(lam1))
		fwhm_slope.append((L0,loglogslope))
	plt.plot([i[0] for i in fwhm_slope],[i[1] for i in fwhm_slope],lw=4, ls = '-', marker = 'o', color = band_colors[band])

def plot_theory_FWHM_vs_Wavelength(alphas,L0s,lams,diam,r0_500):
	""" plot dashed theoretical FWHM(arcsecond) vs. wavelength(nm) curves. Include np.inf in theory_L0s if you want to plot kolmogorov.
		Refer to Tokovinin(2002) "From Differential Image Motion to Seeing" for discussion of kolmogorov prediction, and also numerical 
		fit formula to the vonKarman spectrum results that this function plots. Note Tokovinin's formula uses r0_500=.15m
	"""
	r0 = r0_500*(lams/500.)**(6./5.)
	Kolmogorov_fwhm = arcsec_per_rad*(.98/r0*(lams*10**(-9)))
	
	for L0 in L0s:
		if(L0 == np.inf):
			plt.plot(lams, Kolmogorov_fwhm, lw=3, color = L0_colors[L0], ls = '--')
			continue

		thoeretical_fwhm = []
		for lam in lams:
			theoretical = get_saved_theoretical(lam,L0,diam,r0_500)	
			if(len(theoretical) == 0):
				print "lam{} L0{} thoeretical not found".format(lam, L0)
				continue
			thoeretical_fwhm.append((lam, get_FWHM(theoretical,alphas,False)))
		plt.plot([i[0] for i in thoeretical_fwhm],[i[1] for i in thoeretical_fwhm],lw=3, ls = ':', color = L0_colors[L0])

		# # plot numerical fit from eq. 19 of Tokovinin 2002 "From Differential Image Motion to Seeing" for r0_500 = .15m in limit diam>>r0
		# if(L0>6. and r0_500 == .15):
		# 	FWHM_numerical_fit = [Kolmogorov_fwhm[i]*math.sqrt(1.-2.183*(r0[i]/L0)**.356) for i in range(len(lams))]
		# 	plt.plot(lams, FWHM_numerical_fit, lw=1, ls = '-', color = L0_colors[L0])


def generate_file_name(args):
	"""this will create a file name identical to the name of the fits file generated when simulating a psf with given args
	"""
	file = "lam"+str(args["lam"]) + "_seed"+str(args["seed"]) + "_L0"+str(args["L0"])
	for attr, value in sim_args_def.iteritems():
		if(args[attr] != value and attr!="lam" and attr!="seed" and attr!="L0"):
			file += "_" + attr + str(args[attr])

	return file+".fits"

def main(argv):
	""" Specify the range of paramters for which you want to retrieve simulated .fits image files. 
		Obtain shape meausres or FWHM of radial PSF as desired for plotting. Compare against theoretical
		predictions, saved as radial PSFs.
	"""

	fig, ax = plt.subplots()
	plt.subplots_adjust(left=0.1, bottom=0.15)
	fig.set_size_inches(15,10,forward=True)

	alphas = np.linspace(0.0,.3,31) # radial sampling points of PSF in arcseconds (This is the range that I often used for simulated psf generation)
	fwhms = []

	# get all filenames in specified path
	path = PATH+'sample_images/'
	files = [join(path,f) for f in listdir(path) if isfile(join(path, f))]
	
	# select filenames by specifying args of the simulation
	args = sim_args_def.copy()	# start with the default args, change individual args that 

	# put all of the arguments you wish to range over in this dictionary. Be sure to use the exact corresponding sim_args_def key
	L0s = [100.,75.,50.,25.,15.,10.] 
	lams = np.linspace(300.,1100.,9)
	ranged_args = {"L0":L0s, "lam":lams}

	# update individual args that differ from default which you do not range over.
	args["exptime"] = 30.
	args["pause_exp"] = True

	if(args["r0_500"] == .15):
		alphas = np.linspace(0.0,.4,41) # (This is the range that I used for r0500=.15 simulated psf generation)

	for values in itertools.product(*ranged_args.values()):	# iterate over every combination of ranged args
		key_value_tuples = zip(ranged_args.keys(),values)

		for key_value in key_value_tuples:
			args[key_value[0]]=key_value[1]	# save the ranged args values under appropriate key

		file = generate_file_name(args)	# make an appropriate filename to retrieve a .fits file of the simulation if it exists
		file = path+file

		if file not in files:
			print "file not found: {}".format(file)
			continue

		image = galsim.fits.read(file) #read in image

		# obtain moments and other shape measures
		simple_mom = simple_moments(image)
		e1 = ellip(simple_mom)['e1']*30
		e2 = ellip(simple_mom)['e2']*30
		emag = math.sqrt(e1**2+e2**2)

		# print e1, e2, emag, simple_mom['Ix'], simple_mom['Iy'], simple_mom['Ixx'], simple_mom['Iyy']

		# get the saved montecarlo averaged radial PSF. If not saved, generate one and save it to file
		saved_radial_file = PATH+'saved_radial_profiles.txt'
		data_entry_name = file[file.find("lam"):]		

		radial = get_saved_radial(saved_radial_file, data_entry_name, simple_mom['Ix'], simple_mom['Iy'], alphas, image.array)	


		this_fwhm = get_FWHM(radial,alphas) 	#get the FWHM from radial 
		# #fwhms.append(key_value_tuples+[('fwhm',this_fwhm)]) #if desired, store information in a list of tuples. Here, we simply scatter plot.

		# select plotting styles and plot fwhm vs. wavelength
		color = L0_colors[args['L0']]
		marker = 'o'
		s = 30

		plt.scatter(args["lam"], this_fwhm, s=s, color=color, marker = marker)
	#print fwhms

	# parameters for theoretical prediction plotting
	theory_L0s = [np.inf] + L0s 
	theory_lams = np.linspace(300.,1100.,9)
	alpha_max = .5*arcsec_per_rad*(.98/(args['r0_500']*(300./500.)**(6./5.))*(300.*10**(-9)))	# set the maximum alpha to be 1/2 of the kolmogorov FWHM at 300nm
	theory_alphas = np.arange(0.0,alpha_max+.005,.005)
	
	# plot theoretical fwhm vs. wavelength
	plot_theory_FWHM_vs_Wavelength(theory_alphas,theory_L0s,theory_lams,args['diam'],args['r0_500'])	
	# add legend
	patches = []
	for L0 in theory_L0s:					
		label = "{}m".format(L0)
		if(label == "infm"):
			label = "kolm (inf)"
		color = L0_colors[L0]
		patch = mpatches.Patch(color=color, label=label)
		patches.append(patch)
	plt.legend(handles=patches,title='L0',loc = 'lower left')

	# # plot the wavelength dependence exponent of fwhm vs. L0
	# patches = []
	# patches.append(mpatches.Patch(color='grey', label="Kolmogorov (L0=inf)"))
	# bands = ['r','i']
	# for band in band_colors:
	# 	color = band_colors[band]
	# 	plot_theory_loglogslope_vs_L0(theory_alphas,theory_L0s,args['diam'],args['r0_500'],band)
	# 	patches.append(mpatches.Patch(color=color,label='von Karman, {}-band'.format(band)))
	# plt.legend(handles=patches,loc='lower right')#,title='- theoretical\no simulations')

	title = "FWHM vs. Wavelength"
	plt.title(title, fontsize = 26)
	plt.ylabel("FWHM(\")", fontsize = 20)
	plt.xlabel("Wavelength(nm)", fontsize = 20)
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlim(275,1125)
	ymax = .5
	ymin = .3
	ax.set_ylim(ymin,ymax)
	ax.xaxis.set_ticks(np.arange(300,1200,100))
	ax.yaxis.set_ticks(np.arange(ymin,ymax+.05,.05))
	ax.xaxis.set_ticklabels(np.arange(300,1200,100))
	ax.yaxis.set_ticklabels(np.arange(ymin,ymax+.05,.05))

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14) 

	plt.savefig(PATH+title+".png")
	plt.show()

if __name__ == "__main__":
    main(sys.argv)

