NSG CPPN domain TODOs
=====================


- Python CPPN
	- 2 Inputs (cell position in x,y)
	- 1 Output (height 0<h<max_height, max_height = 3, discretized -> 0,1,2,3)
	- Fill in voxels beneath function's surface. 
	- Voxel is filled when (height - z) > cell_height/2
	
	

Features: 
- num_rooms
	- Calculate total number of voxels num_rooms, num_rooms should be 55
	- allow num_rooms < 55 for evolution but only should num_rooms = 55+
		- How to hold constraint? a posterior constraint (see Features)
	--> a posterior constraint
- Total roof surface area
	- approx. connectivity between rooms (or total outer wall and roof surface area)
	- approx. temperature index: Total roof surface area

Fitness:
- v_out/v_in