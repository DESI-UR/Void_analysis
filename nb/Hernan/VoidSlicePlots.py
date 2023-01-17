import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator, DictFormatter

from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM, z_at_value

from scipy.spatial import cKDTree, ConvexHull

from sympy import solve_poly_system, im
from sympy.abc import x,y

from vast.voidfinder.distance import z_to_comoving_dist

matplotlib.rcParams.update({'font.size': 38})




#Coordinate Transforms
D2R = np.pi/180.

def toSky(cs):
    c1  = cs.T[0]
    c2  = cs.T[1]
    c3  = cs.T[2]
    r   = np.sqrt(c1**2.+c2**2.+c3**2.)
    dec = np.arcsin(c3/r)/D2R
    ra  = (np.arccos(c1/np.sqrt(c1**2.+c2**2.))*np.sign(c2)/D2R)%360
    return r,ra,dec

def toCoord(r,ra,dec):
    c1 = r*np.cos(ra*D2R)*np.cos(dec*D2R)
    c2 = r*np.sin(ra*D2R)*np.cos(dec*D2R)
    c3 = r*np.sin(dec*D2R)
    return c1,c2,c3




class VoidMap():
    def __init__(self,gdata,vfdata,vfdata2):
        self.gdata=gdata
        self.vfdata=vfdata
        self.vfdata2=vfdata2
        #making the data from galaxies more accessible/easier to manipulate?
        self.gr   = z_to_comoving_dist(np.array(gdata['redshift'],dtype=np.float32),0.315,1)
        self.gra  = gdata['ra']
        self.gdec = gdata['dec']

        #Converting galaxy data to cartesian
        #cKDTree finds nearest neighbors to data point
        self.gx,self.gy,self.gz = toCoord(self.gr,self.gra,self.gdec)
        self.kdt = cKDTree(np.array([self.gx,self.gy,self.gz]).T)

        #Simplifying VoidFinder maximal sphere coordinates and converting them into RA,DEC,DIS
        self.vfx = vfdata['x'] 
        self.vfy = vfdata['y']
        self.vfz = vfdata['z']
        self.vfr,self.vfra,self.vfdec = toSky(np.array([self.vfx,self.vfy,self.vfz]).T)
        self.vfrad = vfdata['radius']

        #Not sure what's going on here
        self.vfc  = matplotlib.cm.nipy_spectral(np.linspace(0,1,len(self.vfr)))
        self.vfcc = np.random.choice(range(len(self.vfc)),len(self.vfc),replace=False)

        #Same coordinate conversion, but for holes
        self.vflag = vfdata2['flag']
        self.vfx2 = vfdata2['x']
        self.vfy2 = vfdata2['y']
        self.vfz2 = vfdata2['z']
        self.vfr1,self.vfra1,self.vfdec1 = toSky(np.array([self.vfx2,self.vfy2,self.vfz2]).T)
        self.vfrad1 = vfdata2['radius']

        #matches holes to voids, and then matches galaxies within and outside of voids.
        self.vfx4   = [self.vfx2[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfy4   = [self.vfy2[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfz4   = [self.vfz2[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfr2   = [self.vfr1[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfra2  = [self.vfra1[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfdec2 = [self.vfdec1[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfrad2 = [self.vfrad1[self.vflag==vfl] for vfl in np.unique(self.vflag)]

        #Unsure, creating empty arrays with the same length as the number of voids 
        #(Specifically using the x coordinate)
        #unsure of purpose, currently
        self.gflag_vf = np.zeros(len(self.gx),dtype=bool)
        self.gflag_v2 = np.zeros(len(self.gx),dtype=bool)

        #VoidFinder - Finding all the points within a given radius of a specific point
        for vfl in np.unique(self.vflag):
            self.vfx3 = self.vfx2[self.vflag==vfl]
            self.vfy3 = self.vfy2[self.vflag==vfl]
            self.vfz3 = self.vfz2[self.vflag==vfl]
            self.vfrad3 = self.vfrad1[self.vflag==vfl]
            for i in range(len(self.vfx3)):
                galinds = self.kdt.query_ball_point([self.vfx3[i],self.vfy3[i],self.vfz3[i]],self.vfrad3[i])
                self.gflag_vf[galinds] = True

        #Marking wall galaxies as true
        self.wflag_vf = (1-self.gflag_vf).astype(bool)
        self.wflag_v2 = (1-self.gflag_v2).astype(bool)

    #calculate radii of maximal sphere-slice intersections
    def cint(self, dec):
        cr = []
        for i in range(len(self.vfr)):
            dtd = np.abs(self.vfr[i]*np.sin((self.vfdec[i]-dec)*D2R))
            if dtd>self.vfrad[i]:
                cr.append(0.)
            else:
                cr.append(np.sqrt(self.vfrad[i]**2.-dtd**2.))
        return cr
    
    #calculate radii of hole sphere-slice intersections
    def cint1(self, dec):
        cr = []
        for i in range(len(self.vfr1)):
            dtd = np.abs(self.vfr1[i]*np.sin((self.vfdec1[i]-dec)*D2R))
            if dtd>self.vfrad1[i]:
                cr.append(0.)
            else:
                cr.append(np.sqrt(self.vfrad1[i]**2.-dtd**2.))
        return cr

    #convert a circle's coordinates to ordered boundary
    def gcp(self,cc1,cc2,crad,npt):
        ccx = cc1*np.cos(cc2*D2R)
        ccy = cc1*np.sin(cc2*D2R)
        Cx = np.linspace(0.,2*np.pi,npt)
        Cy = np.linspace(0.,2*np.pi,npt)
        Cx = np.cos(Cx)*crad+ccx
        Cy = np.sin(Cy)*crad+ccy
        C1 = np.sqrt(Cx**2.+Cy**2.)
        C2 = (np.sign(Cy)*np.arccos(Cx/C1)+np.pi*(1.-np.sign(Cy)))/D2R
        return C1,C2
    
    ## Actual plotting
    def setup_axes3(self,fig, rect, ra0, ra1, zlim0, zlim1):
        """
        Sometimes, things like axis_direction need to be adjusted.
        """

        # rotate a bit for better orientation
        tr_rotate = Affine2D().translate(-225, 0)

        # scale degree to radians
        tr_scale = Affine2D().scale(np.pi/180., 1.)

        tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

        grid_locator1 = angle_helper.LocatorDMS(4)
        tick_formatter1 = angle_helper.FormatterDMS()

        grid_locator2 = MaxNLocator(3)

        cz0, cz1 = zlim0, z_to_comoving_dist(np.array([zlim1],dtype=np.float32),.315,1)[0]
        grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                            extremes=(ra0, ra1, cz0, cz1),
                                            grid_locator1=grid_locator1,
                                            grid_locator2=grid_locator2,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=None,
                                            )

        ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
        fig.add_subplot(ax1)

        # adjust axis
        ax1.axis["left"].set_axis_direction("bottom")
        ax1.axis["right"].set_axis_direction("top")

        ax1.axis["bottom"].set_visible(False)
        ax1.axis["top"].set_axis_direction("bottom")
        ax1.axis["top"].toggle(ticklabels=True, label=True)
        ax1.axis["top"].major_ticklabels.set_axis_direction("top")
        ax1.axis["top"].label.set_axis_direction("top")

        ax1.axis["left"].label.set_text(r"r [Mpc h$^{-1}$]")
        ax1.axis["top"].label.set_text(r"$\alpha$")


        # create a parasite axes whose transData in RA, cz
        aux_ax = ax1.get_aux_axes(tr)

        aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
        ax1.patch.zorder=0.8 # but this has a side effect that the patch is
                            # drawn twice, and possibly over some other
                            # artists. So, we decrease the zorder a bit to
                            # prevent this.
        aux_ax.set_facecolor("white")
        return ax1, aux_ax
    
    #plot VoidFinder maximal spheres
    def pvf(self,dec,wdth,npc, ra0, ra1, zlim0, zlim1):
        #fig = plt.figure(1, figsize=(12,6))
        fig = plt.figure(1, figsize=(1600/96,800/96))
        fig.suptitle('Dec: ' + str(dec))
        ax3, aux_ax3 = self.setup_axes3(fig, 111, ra0, ra1, zlim0, zlim1)
        aux_ax3.set_aspect(1)
        Cr = self.cint(dec)
        for i in range(len(self.vfr)):
            if Cr[i]>0:
                Cr2,Cra2 = self.gcp(self.vfr[i],self.vfra[i],Cr[i],npc)
                aux_ax3.plot(Cra2,Cr2,color='blue')
                aux_ax3.fill(Cra2,Cr2,alpha=0.2,color='blue')


        gdcut = (self.gr[self.wflag_vf]*np.sin((self.gdec[self.wflag_vf]-dec)*D2R))**2.<wdth**2.
        aux_ax3.scatter(self.gra[self.wflag_vf][gdcut],self.gr[self.wflag_vf][gdcut],color='k',s=1)
        gdcut = (self.gr[self.gflag_vf]*np.sin((self.gdec[self.gflag_vf]-dec)*D2R))**2.<wdth**2.
        aux_ax3.scatter(self.gra[self.gflag_vf][gdcut],self.gr[self.gflag_vf][gdcut],color='red',s=1)
        return fig

    #plot VoidFinder hole spheres and maximal spheres
    def pvf1(self,dec,wdth,npc, ra0, ra1, zlim0, zlim1):
        #fig = plt.figure(1, figsize=(12,6))
        fig = plt.figure(1, figsize=(1600/96,800/96))
        fig.suptitle('Dec: ' + str(dec))
        ax3, aux_ax3 = self.setup_axes3(fig, 111, ra0, ra1, zlim0, zlim1)
        aux_ax3.set_aspect(1)
        Cr = self.cint1(dec)
        for i in range(len(self.vfr1)):
            if Cr[i]>0:
                Cr2,Cra2 = self.gcp(self.vfr1[i],self.vfra1[i],Cr[i],npc)
                aux_ax3.plot(Cra2,Cr2,alpha=0.1,color='red')
                aux_ax3.fill(Cra2,Cr2,alpha=0.02,color='red')
        
        Cr = self.cint(dec)
        for i in range(len(self.vfr)):
            if Cr[i]>0:
                Cr2,Cra2 = self.gcp(self.vfr[i],self.vfra[i],Cr[i],npc)
                aux_ax3.plot(Cra2,Cr2,color='blue')
                aux_ax3.fill(Cra2,Cr2,alpha=0.2,color='blue')
                
        gdcut = (self.gr[self.wflag_vf]*np.sin((self.gdec[self.wflag_vf]-dec)*D2R))**2.<wdth**2.
        aux_ax3.scatter(self.gra[self.wflag_vf][gdcut],self.gr[self.wflag_vf][gdcut],color='k',s=1)
        gdcut = (self.gr[self.gflag_vf]*np.sin((self.gdec[self.gflag_vf]-dec)*D2R))**2.<wdth**2.
        aux_ax3.scatter(self.gra[self.gflag_vf][gdcut],self.gr[self.gflag_vf][gdcut],color='red',s=1)

        return fig