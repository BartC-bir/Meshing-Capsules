#!/usr/bin/env python-mpacts
'''
    Meshing around number of cells, requires xyz files from clusters_aggl
    This will run a large number of simulations in series, not parallelized yet
    Important variables for stability: 
        *) min_radius_hull - Lower this will make the mesh tighter but less stable
        *) r_cell - Lower this will make fit more tight but less stable
        *) dt - time step
'''
# ==========
# IMPORT MODULES
# ==========
from mpacts.commands.onarrays.setvalue import SetValueCommand
from mpacts.contact.detectors.multigrid import MultiGridContactDetector
from mpacts.contact.matrix.conjugategradient import DefaultConjugateGradientSolver
from mpacts.contact.models.collision.hertz.hertz_matrix import HertzMatrix
from mpacts.core.units import unit_registry as u
from mpacts.core.valueproperties import Variable, VariableFunction
from mpacts.geometrygenerators.polyhedron import Connectivity
from mpacts.geometrygenerators.trianglegeometries import UnitIcoSphere
from mpacts.particles.specialcases import DeformableCell
from mpacts.remeshing import TriangularMeshRemeshingCommand
from mpacts.tools.load_parameters import load_parameters
import DEMutilities.postprocessing.h5storage as h5s

import glob
import mpacts.commands.monitors.progress as pp
import mpacts.core.arrays as ar
import mpacts.core.command as cmds
import mpacts.core.simulation as sim
import mpacts.particles as prt
import mpacts.remeshing.particle_decorators as pd
import mpacts.remeshing.particle_managers as pm
import mpacts.remeshing.particles as pt
import mpacts.remeshing.stages as st
import mpacts.tools.random_seed as rs
import numpy as np
import os

# ==========
# LOAD DATA
# ==========
fs = glob.glob('capsules_*.xyz')
fs.sort()

# For paraview simplicities, I make a directory for writeout
if ~os.path.exists('./meshing/'):
    os.mkdir('./meshing/')

# ===========
# ITERATE OVER ALL CLUSTERS
# ==========
for c, pcd in enumerate(fs):
    print(fs)
    x_cells = np.loadtxt(pcd)
    # if len(x_cells) < 10:
    #     continue
    # ==========
    # INITIALIZE SIMULATION
    # ==========
    mysim = sim.Simulation(f"simulation_{c}", timestep=1.0)
    p = mysim('params')

    # ==========
    # PARAMETERS
    # ==========
    t_cortex = Variable("t_cortex",
                        p,
                        value=1.0 * u('um'),
                        description="Effective thickness of the hull's cortex")
    kv_hull = Variable("kv_hull",
                       p,
                       value=0. * u('kPa'),
                       description="Net bulk modulus of the complete hull")
    ka_cortex = Variable("ka_cortex",
                         p,
                         value=0.0 * u('nN/um'),
                         description="# Local area conservation")
    ka_comp = Variable("ka_cortex_compress",
                       p,
                       value=0.0 * u('nN/um'),
                       description="Area conservation for compression")
    kd_cortex = Variable("kd_cortex",
                         p,
                         value=0.0 * u('nN/um'),
                         description='Global area conservation')
    st_cortex = Variable("surface_tension",
                         p,
                         value=1.0 * u('nN/um'),
                         description="Surface tension")
    lw = Variable("layer_width", p, value=20.0 * u('um'))
    visc_liq = Variable("viscosity",
                        p,
                        value=1.0 * u('Pa*s'),
                        description="Liquid viscosity")
    visc_c = Variable("visc_cortex",
                      p,
                      value=2.0 * u("kPa*s"),
                      description="Viscosity of the cortex")

    # Mechanics contact
    Ehull = Variable('Ehull', p, value=8 * u("kPa"))  # constant stifness
    nuhull = Variable('nuhull', p, value=1. / 3.)  # Poisson ratio of membrane
    Ecell = Variable('Ecell', p, value=8 * u("kPa"))
    nucell = Variable("nucell", p, value=1. / 3.)
    min_radius_hull = Variable("min_radius_hull", p, value=50 * u.um)
    pressure = VariableFunction(
        "pressure", p, function='-2*$surface_tension$/$min_radius_hull$')

    gamma_n_contact = Variable("gamma_n_contact", p, value=60 * u('kPa*s/um'))
    gamma_t_contact = Variable("gamma_t_contact", p, value=60 * u('kPa*s/um'))

    # Geometry hull
    radius = Variable("radius_hull", p,
                      value=100.0 * u('um'))  # Average hull radius
    subd = Variable("subdivision", p, value=7)  # Degree of 'mesh refinement'
    Nverts = Variable("N_vertices",
                      p,
                      value=UnitIcoSphere(subd.get_value()).nVertices())
    Ntriangls = Variable("N_triangles",
                         p,
                         value=UnitIcoSphere(subd.get_value()).nTriangles())

    # Geometry cell
    r_cell = Variable("r_cell", p, value=2 * u('um'))

    # Simulation
    dt = Variable("timestep", p, value=5 * u('ms'))  # Timestep
    cd_ue = Variable("cd_update_every", p, value=5)  # cd update rate
    cg_tol = Variable("cg_tol", p, value=1e-4)
    rhv = Variable("rhv", p, value=1e8)
    out_int = Variable("output_interval", p, value=1.0 * u('s'))
    remesh_int = Variable("remesh_int", p, value=0.5 * u('s'))
    simtime = Variable("simulated_time", p, value=5.0 * u('s'))

    avg_area = VariableFunction(
        "avg_area",
        p,
        function='4*math.pi*$radius_hull$**2/ float( $N_triangles$ )')
    node_area = VariableFunction(
        "node_area",
        p,
        function='4*math.pi*$radius_hull$**2 / float( $N_vertices$ )')
    max_r = VariableFunction("max_inv_curv", p, function='10*$radius_hull$')
    min_r = VariableFunction("min_inv_curv", p, function='0.1*$radius_hull$')
    g_liquid = VariableFunction("gamma_liquid",
                                p,
                                function='1.5*$viscosity$/$radius_hull$')
    # Verlet list keep distance
    cd_kd = VariableFunction("cd_keep_distance", p, function='2*$r_cell$')
    gamma_rhv = VariableFunction("gamma_rhv",
                                 p,
                                 function='($rhv$,0,$rhv$,0,0,$rhv$)')
    liq_ratio = VariableFunction(
        "gamma_liquid_ratio",
        p,
        function=
        '$viscosity$/$visc_cortex$*1.5*3**0.5*$node_area$/( $radius_hull$*$t_cortex$ )'
    )

    mysim.set(timestep=dt)

    if os.path.exists('params.pickle'):
        load_parameters(mysim)

    rs.set_random_seed(456123)

    results = h5s.H5Storage(mysim.name() + '.h5', 'wa')
    results.add_simulation_info(mysim)

    # ----------------------------------------------------------------------------------------------------------------
    hull = prt.ParticleContainer("hull", DeformableCell, mysim)
    cells = prt.ParticleContainer('cells', prt.Sphere0, mysim)

    ar.create_array("Scalar", "layer_width", hull("triangles"))
    ar.create_array("Scalar", "edge_length", hull("triangles"))
    ar.create_array("Scalar", "surface_tension", hull('triangles'))
    ar.create_array("Scalar", "cluster", hull('triangles'))

    ar.create_array("Vector", "Fprim", cells)
    ar.create_array("Scalar", "contact_area", cells)
    ar.create_array("Index", "parentIndex", cells)
    ar.create_array("Index", "cluster", cells)

    # ----------------------------------------------------------------------------------------------------------------
    # Cmds on hull
    hull.DeformableCellGeometryCmds(minimal_radius=min_r, maximal_radius=max_r)
    hull("nodes").FrictionOnAreaCmd(area=hull('nodes')['area'],
                                    gamma_normal=g_liquid,
                                    gamma_tangential=g_liquid)
    hull.DeformableCellInternalMechanicsCmds(
        thickness=t_cortex,
        ka=ka_cortex,
        kd=kd_cortex,
        kv=kv_hull,
        internal_pressure=pressure,
        surface_tension=st_cortex,
        viscosity=visc_c,
        indirect_surface_tension_model=False,
        implicitness=1.0)
    hull("triangles").VTKWriterCmd(executeInterval=out_int,
                                   select_all=True,
                                   directory='./meshing/')

    # ----------------------------------------------------------------------------------------------------------------
    # We need to set geometrical commands on the same frequency as remeshing
    mysim(
        "loop_cmds/pre_body_force_cmds/ExecuteOnce_ComputeTriangleIndexList_hull_triangles"
    ).set(gate=cmds.ExecuteTimeInterval(interval=remesh_int))
    mysim(
        "loop_cmds/pre_body_force_cmds/ExecuteOnce_SortTriangleIndexList_hull_triangles"
    ).set(gate=cmds.ExecuteTimeInterval(interval=remesh_int))

    # compute triangle area before remeshing command
    hull("triangles").TriangleNormalsAndAreaCmd(
        "RemeshNormalsAndAreaCmd",
        parent=mysim("loop_cmds/remove_particles_cmds"),
        x=hull('nodes')['x'])

    # create node and triangle manager + decorators
    node_manager = pm.NodeManager(hull('nodes'),
                                  particle_type=pd.TriangleIndices(pt.Node))
    triangle_manager = pm.TriangleManager(hull('triangles'),
                                          particle_type=pt.Triangle)

    # Adding remeshing command
    TriangularMeshRemeshingCommand(
        "Remeshing",
        mysim,
        stages=[
            st.RemeshingStageSplitLargeTriangles, st.RemeshingStageFlipEdges,
            st.RemeshingStageDeleteSmallTriangles,
            st.RemeshingStageUpdateNodeFixedListContactList
        ],
        node_manager=node_manager,
        triangle_manager=triangle_manager,
        triangle_contact_detectors=[
            mysim("loop_cmds/contact_cmds/CD_Springs_hull")
        ],
        min_area=avg_area.get_value() / 4.,
        max_area=avg_area.get_value() * 4.,
        verbosity=0  # , max_cos = 0.5
        ,
        gate=cmds.ExecuteTimeInterval(interval=remesh_int))

    # ----------------------------------------------------------------------------------------------------------------
    # Cmds on cells
    SetValueCommand("ZeroCellContactArea",
                    mysim,
                    value=0.,
                    array=cells['contact_area'])
    SetValueCommand("ZeroCellFprim", mysim, value=(0, 0, 0), array=cells['F'])

    # Fix cell position
    cells.SetContactMatrixDiagonalCmd("SetContactMatrixCellsCmd",
                                      visc=gamma_rhv)
    # cells.VTKWriterCmd(executeOnce=True,
    #                    select_all=True,
    #                    directory='./meshing/')  # Uncomment to evaluate fit

    # ----------------------------------------------------------------------------------------------------------------
    CM_hull_cell = HertzMatrix("CM_hull_cell",
                               pc1=hull('triangles'),
                               pc2=cells,
                               flip_normals=True,
                               reject_large_overlap=False,
                               E1=Ehull,
                               E2=Ecell,
                               gamma_normal=gamma_n_contact,
                               gamma_tangential=gamma_t_contact,
                               nu1=nuhull,
                               nu2=nucell)
    CD_hull_cell = MultiGridContactDetector("CD_hull_cell",
                                            mysim,
                                            cmodel=CM_hull_cell,
                                            update_every=cd_ue,
                                            keep_distance=cd_kd)

    # ----------------------------------------------------------------------------------------------------------------
    ConjugateGradient = DefaultConjugateGradientSolver(mysim,
                                                       tolerance=cg_tol,
                                                       reset_x=False)
    # DataSaveCommand("SaveDataCommand", mysim, directory='./meshing/',
    #                 gate=cmds.ExecuteTimeInterval(interval=out_int))

    # Add a command that prints some information to the screen at a certain time interval:
    printerlist = [
        pp.DefaultProgressPrinter(mysim),
        pp.PropertyPrinter(ConjugateGradient('steps'), "CG ")
    ]
    printer = pp.PrinterChain(printerlist)
    pp.ProgressIndicator("PrintProgress", mysim, printer, print_interval=4)

    # ------------------------------------------------------------------------------------------------------------------
    # Adding the initial configuration of the cells
    m_cells = np.mean(x_cells, axis=0)
    r_cells = len(x_cells) * [r_cell.get_value()]
    for i, xi in enumerate(x_cells):
        thecells = cells.add_particle()
        thecells.x = tuple(xi)
        thecells.r = r_cell.get_value()
        thecells.cluster = c
        thecells.parentIndex = i

    # ----------------------------------------------------------------------------------------------------------------
    # Actually adding the initial configuration of the hull, with nodes, triangles and connectivity ('springs')
    print(" - subdividing icosphere")
    p = UnitIcoSphere(subd.get_value())
    p.vertices = list(map(tuple, np.array(p.vertices) * radius.get_value()))
    print(" - optimizing bandwidth")
    p.optimize_bandwidth()
    p.cleanLists()
    print(" - calculating connectivity")
    con = Connectivity(p)

    xs = m_cells
    print((" - adding hull"))
    Thehull = hull.add_particle()
    np_vertices = np.array(p.vertices)
    np_vertices += np.array(xs)
    Thehull.nodes.add_and_set_particles(x=list(map(tuple, np_vertices)))
    Thehull.triangles.add_and_set_particles(vertexIndices=p.triangles,
                                            cluster=c,
                                            layer_width=lw.get_value())
    Thehull.add_connectivity(mysim("loop_cmds/contact_cmds/CD_Springs_hull"),
                             con.edgeCorners)

    # ----------------------------------------------------------------------------------------------------------------
    print(" - starting simulation...")
    mysim.run_until(simtime.get_value())

    # ------------------------------------------------------------------------------------------------------------------
    results.simulation_completed()
