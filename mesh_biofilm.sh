# Bash script to create a surface mesh around bacteria
# Requires a csv file with centers of bacteria as columns ['X_c', 'Y_c', 'Z_c'] in meters
# !Will write out potentially a large number of .xyz files (~50 files in my experience)
# Requires meshlab software from https://www.meshlab.net/

# 1) Identify bacteria that are far away from clusters - Will make mesh extremely unstable
mpacts-run-multiple identify_outliers.py cahn* -vf -j6

# 2) Cluster the rest of the bacteria in manageable groups
mpacts-run-multiple clusters_aggl.py cahn* -vf -j6

# 3) Simulations - takes a long time (runs N simulations with N the number of written xyz files)
# ! Uncomment the cells vtk writer to evaluate fit
mpacts-run-multiple hull_mesh.py cahn* -vf -j6

# In the meshing/ directory, you'll find vtp files with the meshes

# 4) Open paraview, open all the vtp files of the mesh --> group all of these files and save as .ply files

# 5) Open .ply files in Meshlab (https://www.meshlab.net/) 
#	--> RMB on one of the files --> Flatten visible layers --> Default apply

# 6) Create poisson mesh around this mesh
#	--> Filters > Remeshing, Simplification and Reconstruction > Surface Reconstruction: Screened Poisson
#	--> Reconstruction depth to 6

# 7) Simplify the mesh
#	--> Filters > Remeshing, Simplification and Reconstruction > Simplification: Quadratic Edge Collapse Decimation
#	--> Preserve normal 

# 8) Export mesh to .ply
#	--> File > Export mesh as > .ply
#	--> Be sure to check normals to export them as well