import numpy as np

import lettuce as lt
import torch
import sys
import imageio
import os
import GPUtil
import shutil

from domain.lettuce2d import express
from util import maptorange
import shapely.validation
import rasterio.features
import shapely.affinity

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class NaNReporter:
    """Reports any NaN and aborts the simulation"""
    def __call__(self,i,t,f):
        if torch.isnan(f).any()==True:
            print ("NaN detected in time step ", i)
            print ("Abort")
            f3 = open("done", "a")
            f3.close()
            sys.exit()


def get(population, domain):
    # Express shapes
    phenotypes = express.do(population, domain)
    fitness = []
    features = []

    for i in range(len(phenotypes)):
        area = phenotypes[i].area
        perimeter = phenotypes[i].length
        enstrophy, umax = get_flowfeatures(phenotypes[i], domain['bitmap_resolution'], 'data', domain['max_t'], domain['report_interval'], domain['mean_interval'])
        features.append([area, perimeter, enstrophy, umax])

    features = np.asarray(features)
    for i in range(len(domain['feat_ranges'][0])):
        features[:,i] = maptorange.do(features[:,i], domain['feat_ranges'][0][i], domain['feat_ranges'][1][i])    
    fitness = (1/(1+features[:,3]))*2-1 ;
    fitness = np.transpose([fitness])
    return fitness, features


def get_flowfeatures(polygon, bitmap_resolution, datadir='data', max_t=30.0, report_interval=100, mean_interval=100):
    # Diameter of the Building
    D = bitmap_resolution
    
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    # "cuda:0" for GPU "cpu" for CPU
    if GPUtil.getFirstAvailable(attempts=10, interval=1):
        device=torch.device("cuda:0")
        print("Using GPU/CUDA")
    else:
        device=torch.device("cpu")
        print("Using CPU")

    rasterize_to_disk(polygon, bitmap_resolution)
    
    stencil = lt.D2Q9
    lattice = lt.Lattice(stencil,device=device,dtype=torch.float32)

    building = imageio.imread(datadir + '/building.bmp')
    building = building.astype(bool)
    building = np.rot90(building, 3)

    
    flow=lt.Obstacle2D(600,300,reynolds_number=3900,mach_number=0.075,lattice=lattice,char_length_lu=D)

    #Create a mask to determine the bounce back boundary of the cylinder
    x = flow.grid
    mask_np = np.zeros([flow.resolution_x,flow.resolution_y],dtype=bool)
    relative_position_x = int(mask_np.shape[0]/3-building.shape[0]/2)
    relative_position_y = int(mask_np.shape[1]/2-building.shape[1]/2)

    mask_np[relative_position_x:relative_position_x+building.shape[0],relative_position_y:relative_position_y+building.shape[1]]=building

    flow.mask=mask_np
    collision=lt.KBCCollision2D(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming=lt.StandardStreaming(lattice)
    lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)

    simulation=lt.Simulation(flow,lattice,collision,streaming)

    # Create and append the NaN reporter to detect instabilities
    NaN=NaNReporter()
    simulation.reporters.append(NaN)

    # If desired append a VTK reporter, reporting every n simulation steps
    vtk_rep = lt.VTKReporter(lattice,flow,report_interval,datadir + '/cylinder')
    simulation.reporters.append(vtk_rep)

    from lettuce.observables import Mass, Enstrophy, MaximumVelocity
    mass_observable = Mass(lattice,flow)
    enstrophy_observable = Enstrophy(lattice,flow)
    maxU_observable = MaximumVelocity(lattice,flow)

    mass_reporter = lt.ObservableReporter(mass_observable,report_interval)
    simulation.reporters.append(mass_reporter)

    f1 = open(datadir + '/enstrophy.csv', "a")
    enstrophy_reporter = lt.ObservableReporter(enstrophy_observable,interval=report_interval,out=f1)
    simulation.reporters.append(enstrophy_reporter)

    f2 = open(datadir + '/umax.csv', "a")
    maxU_reporter = lt.ObservableReporter(maxU_observable,interval=report_interval,out=f2)
    simulation.reporters.append(maxU_reporter)

    drag_reporter = lt.ObservableReporter

    # print("Simulating steps:", int(flow.units.convert_time_to_lu(max_t)))
    simulation.step(int(flow.units.convert_time_to_lu(max_t)))

    f1.close()
    f2.close()

    f3 = open(datadir + '/done', "a")
    f3.close()

    data_enstrophy = np.loadtxt(datadir + '/enstrophy.csv', dtype=float)
    data_umax = np.loadtxt(datadir + '/umax.csv', dtype=float)
    
    if data_enstrophy.shape[0] < mean_interval:
        mean_interval = data_enstrophy.shape[0]

    for filename in os.listdir(datadir):
        file_path = os.path.join(datadir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    return np.mean(data_enstrophy[-mean_interval:,2]), np.mean(data_umax[-mean_interval:,2])


def rasterize_to_disk(polygon, bitmap_resolution):
    polygon = shapely.affinity.scale(polygon, xfact=bitmap_resolution, yfact=bitmap_resolution, zfact=1.0, origin='center')
    centroid_x,centroid_y = polygon.centroid.xy
    polygon = shapely.affinity.translate(polygon, xoff=-centroid_x[0]+bitmap_resolution/2, yoff=-centroid_y[0]+bitmap_resolution/2, zoff=0.0)
    img = rasterio.features.rasterize([polygon], out_shape=(bitmap_resolution, bitmap_resolution))
    img *= 255
    with rasterio.Env():
        with rasterio.open("data/building.bmp", 'w',
            driver='BMP',
            height=img.shape[0],
            width=img.shape[1],
            count=1,
            dtype=rasterio.uint8,
            nodata=1) as dst:
                dst.write(img, 1)
                dst.write_colormap(
                    1, {
                    0: (0, 0, 0),
                    255: (255, 255, 255) })
