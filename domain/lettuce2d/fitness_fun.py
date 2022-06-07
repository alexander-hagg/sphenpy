import numpy as np

import lettuce as lt
import torch
import sys
import imageio
import os
import GPUtil

from domain.lettuce2d import express
from util import maptorange
import shapely.validation
import rasterio.features
import shapely.affinity
# import matplotlib.pyplot as plt


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
        flowfeatures = get_flowfeatures(phenotypes[i], 64)
        quit()
        features.append([area, perimeter])
        fitness.append(flowfeatures)

    features = np.asarray(features)
    for i in range(domain['nfeatures']):
        features[:,i] = maptorange.do(features[:,i], domain['feat_ranges'][0][i], domain['feat_ranges'][1][i])    
    fitness = maptorange.do(fitness, domain['fit_range'][0], domain['fit_range'][1])
    fitness = np.transpose([fitness])
    return fitness, features


def get_flowfeatures(polygon, resolution):
    if not os.path.exists("data"):
        os.makedirs("data")

    # "cuda:0" for GPU "cpu" for CPU
    if GPUtil.getAvailable():
        device=torch.device("cuda:0")
        print("Using GPU/CUDA")
    else:
        device=torch.device("cpu")
        print("Using CPU")

    rasterize_to_disk(polygon, resolution)
    
    stencil = lt.D2Q9
    lattice = lt.Lattice(stencil,device=device,dtype=torch.float32)

    building = imageio.imread('data/building.bmp')
    building = building.astype(bool)

    #Time period of the simulation
    max_t=5 # 30.0

    #Diameter of the Building
    D=resolution

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

    #Create and append the NaN reporter to detect instabilities
    NaN=NaNReporter()
    simulation.reporters.append(NaN)

    #If desired append a VTK reporter, reporting every n simulation steps
    vtk_rep = lt.VTKReporter(lattice,flow,100,'./data/cylinder')
    simulation.reporters.append(vtk_rep)

    from lettuce.observables import Mass, Enstrophy, MaximumVelocity
    mass_observable = Mass(lattice,flow)
    enstrophy_observable = Enstrophy(lattice,flow)
    maxU_observable = MaximumVelocity(lattice,flow)

    # ORIGINAL INTERVAL: 10
    mass_reporter = lt.ObservableReporter(mass_observable,50)
    simulation.reporters.append(mass_reporter)

    f1 = open("data/enstrophy.csv", "a")
    enstrophy_reporter = lt.ObservableReporter(enstrophy_observable,interval=50,out=f1)
    simulation.reporters.append(enstrophy_reporter)

    f2 = open("data/maxU.csv", "a")
    maxU_reporter = lt.ObservableReporter(maxU_observable,interval=50,out=f2)
    simulation.reporters.append(maxU_reporter)

    drag_reporter = lt.ObservableReporter

    print ("Simulating steps:", int(flow.units.convert_time_to_lu(max_t)))
    simulation.step(int(flow.units.convert_time_to_lu(max_t)))

    # Set a checkpoint for further investigations
    simulation.save_checkpoint('data/checkpoint')
    #simulation.load_checkpoint('checkpoint')

    f1.close()
    f2.close()

    print ("Setting DONE flag")

    f3 = open("data/done", "a")
    f3.close()
    return 0

def rasterize_to_disk(polygon, resolution):
    polygon = shapely.affinity.scale(polygon, xfact=resolution, yfact=resolution, zfact=1.0, origin='center')
    centroid_x,centroid_y = polygon.centroid.xy
    polygon = shapely.affinity.translate(polygon, xoff=-centroid_x[0]+resolution/2, yoff=-centroid_y[0]+resolution/2, zoff=0.0)
    img = rasterio.features.rasterize([polygon], out_shape=(resolution, resolution))
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