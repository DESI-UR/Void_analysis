'''
This is a compilation of the functions written by Irem Nesli Erez in her 
deltafields_plot notebook.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes

import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator

from vast.voidfinder.voidfinder_functions import xyz_to_radecz

from scipy.spatial import cKDTree
################################################################################




################################################################################
# Constants
#-------------------------------------------------------------------------------
D2R = np.pi/180
################################################################################




################################################################################
# Figure formatting
#-------------------------------------------------------------------------------
plt.rc('text', usetex=False)
plt.rc('font', family='serif', size=10)
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def setup_axes3(fig, rect, ra_range, cz0, cz1):
    '''
    Sometimes, things like axis_direction need to be adjusted


    PARAMETERS
    ==========

    fig : 

    rect : 

    ra_range : length-2 list, array, or tuple
        Minimum [0] and maximum [1] RA to include in plot, in units of degrees

    cz0, cz1 : float
        Minimum and maximum radial distances to use in the plot
    '''

    # Rotate a bit for better orientation
    #tr_rotate = Affine2D().translate(-95, 0)

    # Scale degree to radians
    tr_scale = Affine2D().scale(np.pi/180, 1)

    tr = tr_scale + PolarAxes.PolarTransform()

    grid_locator1 = angle_helper.LocatorDMS(4)
    tick_formatter1 = angle_helper.FormatterDMS()

    grid_locator2 = MaxNLocator(3)

    ra0, ra1 = ra_range + 90

    grid_helper = floating_axes.GridHelperCurveLinear(tr, 
                                                      extremes=(ra0, ra1, cz0, cz1), 
                                                      grid_locator1=grid_locator1, 
                                                      grid_locator2=grid_locator2, 
                                                      tick_formatter1=tick_formatter1, 
                                                      tick_formatter2=None)

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)

    fig.add_subplot(ax1)

    # Adjust axis
    ax1.axis['left'].set_axis_direction('bottom')
    ax1.axis['right'].set_axis_direction('top')
    ax1.axis['top'].set_axis_direction('bottom')

    ax1.axis['bottom'].set_visible(False)
    ax1.axis['top'].toggle(ticklabels=True, label=True)
    ax1.axis['top'].major_ticklabels.set_axis_direction('top')
    ax1.axis['top'].label.set_axis_direction('top')

    ax1.axis['left'].label.set_text(r'r [Mpc/$h$]')
    ax1.axis['bottom'].label.set_text(r'$\alpha$')

    # Create a parasite axis whose transData in RA, cz
    aux_ax = ax1.get_aux_axes(tr)

    aux_ax.patch = ax1.patch # For aux_ax to have a clip path as in ax
    ax1.patch.zorder = 0.8 # But this has a side effect that the path is drawn 
                           # twice, and possibly over some other artists.  So, 
                           # we decrease the zorder a bit to prevent this.
    aux_ax.set_facecolor('white')

    return ax1, aux_ax
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def cint2(dec, vr, vdec, vrad):
    '''
    Calculate radii of hole-slice intersections


    PARAMETERS
    ==========

    dec : float
        Declination (in degrees) of the center of the slice

    vr : list of lists
        For each void, a list of the comoving distances to the centers of all 
        the holes in that void.

    vdec : list of lists
        For each void, a list of the declinations of the centers of all the 
        holes in that void.
        
    vrad : list of lists
        For each void, a list of the radii for all the holes in that void.


    RETURNS
    =======

    cr : list
        List of radii of the holes' intersections with the midplane of the slice
    '''

    # Initialize output list
    cr = []

    for i in range(len(vr)):

        cr.append([])

        for j in range(len(vr[i])):

            ####################################################################
            # Calculate the distance between the center of the hole and the 
            # slice
            #-------------------------------------------------------------------
            dtd = np.abs(vr[i][j]*np.sin((vdec[i][j] - dec)*D2R))
            ####################################################################


            ####################################################################
            # If the hole intersects with the slice, append the radius of the 
            # circle of intersection.  If not, then append 0 (no intersection).
            #-------------------------------------------------------------------
            if dtd > vrad[i][j]:
                
                cr[i].append(0.)

            else:
                
                cr[i].append(np.sqrt(vrad[i][j]**2 - dtd**2))
            ####################################################################

    return cr
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def gcp2(s, ra, Cr, npt, chkdpth):
    '''
    Convert circles' coordinates to ordered boundary


    PARAMETERS
    ==========

    s : list of floats
        Comoving distance to center of circles

    ra : list of floats
        ra of center of circles

    Cr : list of floats
        Radii of circles in void that intersect with declination slice

    npt : 

    chkdpth : 
    '''

    ccx = s*np.cos(ra*D2R)
    ccy = s*np.sin(ra*D2R)

    Cx = [np.linspace(0, 2*np.pi, int(npt*Cr[k]/10)) for k in range(len(ccx))]
    Cy = [np.linspace(0, 2*np.pi, int(npt*Cr[k]/10)) for k in range(len(ccx))]

    Cx = [np.cos(Cx[k])*Cr[k]+ccx[k] for k in range(len(ccx))]
    Cy = [np.sin(Cy[k])*Cr[k]+ccy[k] for k in range(len(ccx))]

    for i in range(len(ccx)):
        for j in range(len(ccx)):

            if i == j:
                continue

            cut = (Cx[j] - ccx[i])**2 + (Cy[j] - ccy[i])**2 > Cr[i]**2

            Cx[j] = Cx[j][cut]
            Cy[j] = Cy[j][cut]

    Cp = []

    for i in range(len(ccx)):

        Cp.extend(np.array([Cx[i], Cy[i]]).T.tolist())

    Cp = np.array(Cp)

    kdt = cKDTree(Cp)

    Cpi = [0]

    while len(Cpi) < len(Cp):

        if len(Cpi) == 1:

            nid = kdt.query(Cp[Cpi[-1]], 2)[1][1]

        else:

            nids = kdt.query(Cp[Cpi[-1]], chkdpth+1)[1][1:]

            for k in range(chkdpth):

                if nids[k] not in Cpi[(-1*(chkdpth + 1)):-1]:

                    nid = nids[k]

                    break

            nids = kdt.query(Cp[Cpi[-1]], 7)[1][1:]

        Cpi.append(nid)

    C1 = np.sqrt(Cp[Cpi].T[0]**2 + Cp[Cpi].T[1]**2)

    C2 = (np.sign(Cp[Cpi].T[1])*np.arccos(Cp[Cpi].T[0]/C1) + np.pi*(1 - np.sign(Cp[Cpi].T[1])))/D2R

    return C1, C2
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def pvfmine(delta, 
            quasars, 
            ra_range, 
            dec0, 
            holes, 
            wdth, 
            npt, 
            chkdpth, 
            figure_file=None):
    '''
    Plot VoidFinder voids with delta fields


    PARAMETERS
    ==========

    delta : Astropy table
        Data of all delta values.  Required columns:
          - ra
          - dec
          - comoving
          - delta

    quasars : Astropy table
        Data of all the quasars.  Required columns:
          - ra
          - dec
          - comoving

    ra_range : length-2 list, array, or tuple
        Minimum [0] and maximum [1] RA to include in plot, in units of degrees

    dec0 : float
        Declination value (in degrees) of center of slice

    holes : Astropy table
        Contains all the details of the holes for the voids

    wdth : float
        Thickness of sky slice to plot

    npt : 

    chkdpth : 

    figure_file : string
        File name (including path) for saving the figure.  Default is None - 
        image will be shown to the screen and not saved.
    '''

    ############################################################################
    # Sort the voids so that we have a list of lists for all the necessary 
    # hole properties
    #---------------------------------------------------------------------------
    # First, convert the center coordinates from Cartesian to sky
    holes = xyz_to_radecz(holes)

    vcz_sorted = []
    vra_sorted = []
    vdec_sorted = []
    vr_sorted = []

    for vflag in np.unique(holes['flag']):

        this_void = holes['flag'] == vflag

        vcz_sorted.append(holes['r'][this_void])
        vra_sorted.append(holes['ra'][this_void])
        vdec_sorted.append(holes['dec'][this_void])
        vr_sorted.append(holes['radius'][this_void])
    ############################################################################


    ############################################################################
    # Set up axes for plot
    #---------------------------------------------------------------------------
    fig = plt.figure(figsize=(1600/96, 800/96))

    ax3, aux_ax3 = setup_axes3(fig, 
                               111, 
                               ra_range, 
                               np.min(delta['comoving']), 
                               np.max(quasars['comoving']))
    ############################################################################


    ############################################################################
    # Plot voids
    #---------------------------------------------------------------------------
    Cr = cint2(dec0, vcz_sorted, vdec_sorted, vr_sorted)

    for i in range(len(vcz_sorted)):

        if np.sum(Cr[i]) > 0:

            Cr2, Cra2 = gcp2(vcz_sorted[i], vra_sorted[i], Cr[i], npt, chkdpth)

            aux_ax3.plot(Cra2, Cr2, color='mediumpurple')
            aux_ax3.fill(Cra2, Cr2, alpha=0.2, color='mediumpurple')
    ############################################################################


    ############################################################################
    # Plot delta field values
    #---------------------------------------------------------------------------
    gdcut = (delta['comoving']*np.sin((delta['dec'] - dec0)*D2R))**2 < wdth**2

    aux_ax3.scatter(delta['ra'][gdcut], 
                    delta['comoving'][gdcut], 
                    color='dodgerblue', 
                    alpha=0.05, 
                    s=1, 
                    label='Delta fields')
    ############################################################################


    ############################################################################
    # Plot quasars
    #---------------------------------------------------------------------------
    qcut = (quasars['comoving']*np.sin((quasars['dec'] - dec0)*D2R))**2 < wdth**2

    aux_ax3.scatter(quasars['ra'][qcut], 
                    quasars['comoving'][qcut], 
                    color='orange', 
                    s=1, 
                    label='Quasars')
    ############################################################################


    ra_min = ra_range[0]
    ra_max = ra_range[1]

    plt.rc('font', size=8)

    aux_ax3.legend(bbox_to_anchor=(1.1, 1.05))

    plt.title('Voids with RA $\in$[' + str(round(ra_range[0] - 90, 2)) + ',' + str(round(ra_range[1] - 90, 2)) + '] and centered at DEC =' + str(dec0) + '$^\circ$ with ' + str(2*wdth) + 'Mpc/h thickness', x=0.5, y=1.15)

    ############################################################################
    # Determine whether to save or show the figure
    #---------------------------------------------------------------------------
    if figure_file is not None:
        plt.savefig(figure_file + '.eps', format='eps', dpi=300)
    else:
        plt.show()
    ############################################################################
################################################################################