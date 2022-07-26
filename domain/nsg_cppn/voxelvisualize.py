import webbrowser
import os
import json
import numpy as np
from jinja2 import Template

'''Tools for io, dataset handling, browser based 3D visualization, and numpy array to 3D mesh conversion'''

def get_path(dirname, fname=''):
    '''returns absolute path for file in a directory'''
    #only works when called from same dir
    abspath = os.path.dirname(__file__)
    dirpath = os.path.join(abspath, dirname)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    return os.path.join(dirpath, fname)

def generate_coords(size):
    '''input coords for nn, vector of [x, y, radius]'''
    x = np.arange(0, size)
    y = np.arange(0, size)
    z = np.arange(0, size)
    x_points, y_points, z_points = np.meshgrid(x, y, z)
    center = size / 2
    tmp = []
    for i in range(x_points.shape[0]):
        for j in range(y_points.shape[1]):
            for k in range(z_points.shape[2]):
                dx = x_points[i, j, k]
                dy = y_points[i, j, k] 
                dz = z_points[i, j, k] 
                tmp.append([dx , dy , dz , np.sqrt((dx - center) ** 2 + (dy - center) ** 2 + (dz - center) ** 2)])

    data = np.array(tmp, dtype=np.float32)
    data_min, data_max = np.amin(data), np.amax(data)
    normal = (data - data_min)  / (data_max - data_min) 
    return normal

def gen_coord_datasets():
    '''Generate and save coordinate datasets for disk access.
       Coordinate datasets are in powers of 2 sizes 8:256'''

    if not os.path.exists('coord_datasets'):
        for i in range(3, 9, 1):
            size = 2 ** i
            save_path = get_path('coord_datasets', 'data{}'.format(size))
            data = generate_coords(size)
            np.save(save_path, data)
        print('datasets generated')
    else:
        print('datasets already generated')

def load_coord_dataset(size):
    data_path = get_path('coord_datasets', 'data{}.npy'.format(size))
    return np.load(data_path)


def np2vox(bin_array):
    '''Convert binary numpy ndarray to indexed 3D mesh data'''

    def scale_verts(x, y, z, face_type):
        x *= 2
        y *= 2
        z *= 2
        verts = []

        if face_type == 0: 
            verts = [(0.0 + x, 2.0 + y, 2.0 + z),
                     (0.0 + x, 0.0 + y, 2.0 + z),
                     (2.0 + x, 2.0 + y, 2.0 + z),
                     (2.0 + x, 0.0 + y, 2.0 + z)]

        elif face_type == 1: 
            verts = [(0.0 + x, 0.0 + y, 0.0 + z),
                     (2.0 + x, 0.0 + y, 0.0 + z),
                     (0.0 + x, 0.0 + y, 2.0 + z),               
                     (2.0 + x, 0.0 + y, 2.0 + z)]

        elif face_type == 2: 
            verts = [(0.0 + x, 2.0 + y, 0.0 + z),
                     (0.0 + x, 0.0 + y, 0.0 + z),
                     (2.0 + x, 2.0 + y, 0.0 + z),
                     (2.0 + x, 0.0 + y, 0.0 + z)]

        elif face_type == 3: 
            verts = [(0.0 + x, 2.0 + y, 0.0 + z),
                     (2.0 + x, 2.0 + y, 0.0 + z),
                     (0.0 + x, 2.0 + y, 2.0 + z),
                     (2.0 + x, 2.0 + y, 2.0 + z)]

        elif face_type == 4: 
            verts = [(2.0 + x, 2.0 + y, 0.0 + z),
                     (2.0 + x, 0.0 + y, 0.0 + z),
                     (2.0 + x, 2.0 + y, 2.0 + z),
                     (2.0 + x, 0.0 + y, 2.0 + z)]

        elif face_type == 5: 
            verts = [(0.0 + x, 2.0 + y, 0.0 + z),
                     (0.0 + x, 0.0 + y, 0.0 + z),
                     (0.0 + x, 2.0 + y, 2.0 + z),
                     (0.0 + x, 0.0 + y, 2.0 + z)]
        
        return verts

    def scale_faces(scale, face_type):

        faces = np.array([[[0, 1, 3, 2]],
                          [[0, 1, 3, 2]],
                          [[1, 0, 2, 3]],
                          [[1, 0, 2, 3]],
                          [[1, 0, 2, 3]],
                          [[0, 1, 3, 2]]])
  
        scaled_faces = faces[face_type] + 4 * scale
        return scaled_faces.tolist()
        
    print('--> BUILDING MESH')
    print('--> VOXEL VOLUME:', np.count_nonzero(bin_array))
    bin_array = np.fliplr(np.flipud(bin_array))
    verts = []
    faces = []
    x, y, z = bin_array.shape[0], bin_array.shape[1], bin_array.shape[2]
    x_max, y_max, z_max = x - 1, y - 1, z - 1
    count = 0
    bin_array[0, :, :] = 0
    bin_array[:, 0, :] = 0
    bin_array[:, :, 0] = 0
    bin_array[x_max, :, :] = 0
    bin_array[:, y_max, :] = 0
    bin_array[:, :, z_max] = 0
        
    for i in range(x): 
        for j in range(y):
            for k in range(z):
                current = bin_array[i,j,k]
                if i == x_max:
                    i = x_max - 1
                elif i == 0:
                    i = 1
                if j == y_max:
                    j = y_max - 1
                elif j == 0:
                    j = 1
                if k == z_max:
                    k = z_max - 1 
                elif k == 0:
                    k = 1

                surrounding = np.array([bin_array[i,j+1,k],    
                                        bin_array[i-1,j,k], 
                                        bin_array[i,j-1,k], 
                                        bin_array[i+1,j,k], 
                                        bin_array[i,j,k+1], 
                                        bin_array[i,j,k-1]], dtype=int)
                if current == 1:
                    for num in range(len(surrounding)):
                        if surrounding[num] == 0:
                            verts += scale_verts(k, i, j, num)
                            faces += scale_faces(count, num)
                            count += 1
    return verts, faces

def render_voxels(voxels):
    '''Render np array in the browser as a mesh using np2vox func and three.js lib'''
    verts, faces = np2vox(voxels)
    mesh_data = {'verts': verts, 'faces': faces}
    json_mesh = json.dumps(mesh_data)

    html = Template('''<html>
        <head>
            <meta charset="utf-8">
            <title>Result</title>
            <style>
                body { margin: 0; }
            </style>
        </head>
        <body>
            <script type="module">
                import * as THREE from 'https://threejsfundamentals.org/threejs/resources/threejs/r115/build/three.module.js';
                import {OrbitControls} from 'https://threejsfundamentals.org/threejs/resources/threejs/r115/examples/jsm/controls/OrbitControls.js';

                let scene, renderer, camera
                let cube

                function init() {

                    scene = new THREE.Scene()
                    renderer = new THREE.WebGLRenderer({antialias:true})
                    renderer.setSize(window.innerWidth, window.innerHeight)
                    document.body.appendChild(renderer.domElement)

                    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000)
                    camera.position.set(-2*{{gridlength}}, 2*{{gridlength}}, -{{gridlength}});
                    camera.up = new THREE.Vector3(0, 1, 0);
                    camera.lookAt(new THREE.Vector3(0, 0, 0))
                    scene.add(camera);

                    let controls = new OrbitControls(camera, renderer.domElement)
                    controls.target.set( 0, 0, 0);
                    // camera.position.set( 0, 10*{{gridlength}}, 100 );
                    // camera.position.set( 5000, 5000, 5000 );
                    controls.update();
                    // controls.minDistance = 150;
                    controls.maxDistance = 10000;
                    controls.zoomSpeed = 2;
                    controls.enablePan = true;
                    controls.rotateSpeed = 2;

                    //lights
                    var light1 = new THREE.AmbientLight(0x888888, 1.0);
                    scene.add(light1);
                    var light2 = new THREE.PointLight(0xff3333, 0.75);
                    light2.position.set(100, 200, 500);
                    scene.add(light2)
                    var light4 = new THREE.PointLight(0x3333ff, 0.75);
                    light2.position.set(-100, 200, -500);
                    scene.add(light4)
                    var light3 = new THREE.SpotLight(0xddddff, 1);
                    light3.position.set(-300, -300, 300);
                    scene.add(light3);

                    window.addEventListener('resize',windowResize, false);

                    function windowResize(){
                        camera.aspect = window.innerWidth / window.innerHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize( window.innerWidth, window.innerHeight);
                    }

                    const textureMaterial = new THREE.MeshBasicMaterial();
                    const loader = new THREE.TextureLoader();
                    loader.load( '/home/alex/repositories/sphenpy/domain/nsg_cppn/data/maps.png', 
                    function ( texture ) {    
                        // The texture has loaded, so assign it to your material object. In the 
                        // next render cycle, this material update will be shown on the plane 
                        // geometry
                        textureMaterial.map = texture;
                        textureMaterial.side = THREE.DoubleSide;
                        textureMaterial.needsUpdate = true;
                    });

                    var geo = new THREE.PlaneGeometry(3*{{gridlength}}, 3*{{gridlength}});
                    const material2 = new THREE.MeshBasicMaterial( { color: 0x555555 } );
                    const plane = new THREE.Mesh(geo, material2);
                    plane.rotateX(Math.PI / 2);
                    plane.rotateY(Math.PI);
                    scene.add( plane );

                    var mesh_data = JSON.parse(JSON.stringify({{data}}));
                    var verts = mesh_data.verts;
                    var faces = mesh_data.faces;

                    var geometry = new THREE.Geometry();

                    for (let i=0; i < verts.length; i++){
                        geometry.vertices.push(new THREE.Vector3(verts[i][1], verts[i][2], verts[i][0]));
                    }
                    for (let i=0; i < faces.length; i++){
                        geometry.faces.push(new THREE.Face3(faces[i][0], faces[i][1], faces[i][2]));
                        geometry.faces.push(new THREE.Face3(faces[i][2], faces[i][3], faces[i][0]));
                    }
                    geometry.computeBoundingSphere();
                    geometry.computeFaceNormals();
                    geometry.computeVertexNormals();

                    var material = new THREE.MeshLambertMaterial({

                                color: 0x616c72,
                            
                                
                    });
                    var material1 = new THREE.MeshLambertMaterial({

                                color: 0x616c72,
                                wireframe: false,
                                transparent: true,
                                opacity: 0.7,                                
                    });
                    
                    
                    var voxmesh = new THREE.Mesh(geometry, material);
                    
                    var x = {{x}};
                    var y = {{y}};
                    var z = {{z}};
                    voxmesh.translateX(-{{gridlength}})
                    voxmesh.translateY(-2)
                    voxmesh.translateZ(-{{gridlength}})
                    scene.add(voxmesh);

                }

                function render() {
                    renderer.render(scene, camera)
                    // controls.update(0.01)
                    requestAnimationFrame(render)
                }

                init()
                render()
            </script>
        </body>
        </html>''')

    gridlength = voxels.shape
    new_html = html.render(data=json_mesh, x=voxels.shape[0], y=voxels.shape[1], z=voxels.shape[2], gridlength=gridlength)
    rnd_intname = np.random.randint(9999999)
    path = get_path('templates', 'template' + str(rnd_intname) + '.html')
    with open(path, 'w') as f:
        f.write(new_html)
    webbrowser.open(path, new=2)

