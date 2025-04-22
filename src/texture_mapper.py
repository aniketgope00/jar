import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from OpenGL.raw.GL.VERSION.GL_1_1 import GL_UNSIGNED_INT, GLuint
import pywavefront
from PIL import Image
import numpy as np
from OpenGL.arrays import ArrayDatatype

def load_texture(image_path):
    """
    Load a texture from an image file and bind it to OpenGL.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        int: OpenGL texture ID.
    """
    # Ensure OpenGL context is active
    if not pygame.display.get_init():
        raise RuntimeError("OpenGL context is not initialized. Ensure this function is called after pygame.display.set_mode().")

    # Generate and bind texture
    texture_surface = pygame.image.load(image_path)
    texture_data = pygame.image.tostring(texture_surface, 'RGB', 1)
    width = texture_surface.get_width()
    height = texture_surface.get_height()

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, texture_data)

    # Generate mipmaps for better texture quality at different distances
    glGenerateMipmap(GL_TEXTURE_2D)

    # Set texture parameters for high-quality rendering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    return texture_id


def generate_tangent_bitangent(v1, v2, v3, uv1, uv2, uv3):
    # Calculate tangent and bitangent vectors for normal mapping
    edge1 = v2 - v1
    edge2 = v3 - v1
    deltaUV1 = uv2 - uv1
    deltaUV2 = uv3 - uv1
    
    f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])
    tangent = f * (deltaUV2[1] * edge1 - deltaUV1[1] * edge2)
    bitangent = f * (-deltaUV2[0] * edge1 + deltaUV1[0] * edge2)
    
    return tangent, bitangent

def normalize_mesh(vertices):
    verts = np.array(vertices).reshape(-1, 3)
    center = (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2
    scale = np.max(np.abs(verts - center))
    return center, scale

def calculate_uv_mapping(vertex, normal):
    # Improved UV mapping using spherical projection
    x, y, z = vertex
    nx, ny, nz = normal
    
    # Calculate spherical coordinates
    u = 0.5 + np.arctan2(x, z) / (2 * np.pi)
    v = 0.5 - np.arcsin(y) / np.pi
    
    # Adjust UVs based on normal direction for better mapping
    u = u + 0.5 * nx
    v = v + 0.5 * ny
    
    return u, v

def render_textured_mesh(mesh_path, texture_path):
    # Count total triangles first
    scene = pywavefront.Wavefront(mesh_path, collect_faces=True, create_materials=True, strict=False)
    total_triangles = sum(len(mesh.faces) for mesh in scene.mesh_list)
    print(f"Total number of triangles in the mesh: {total_triangles}")
    
    pygame.init()
    display = (1024, 768)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    # Enable all necessary OpenGL features
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)  # Add second light for better illumination
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_NORMALIZE)  # Automatically normalize normals
    glShadeModel(GL_SMOOTH)  # Enable smooth shading
    
    # Main light setup (warm light from front-right)
    glLight(GL_LIGHT0, GL_POSITION, (5.0, 3.0, 5.0, 1.0))
    glLight(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))
    glLight(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.9, 1.0))
    glLight(GL_LIGHT0, GL_SPECULAR, (1.0, 0.95, 0.9, 1.0))
    
    # Fill light setup (cool light from back-left)
    glLight(GL_LIGHT1, GL_POSITION, (-3.0, 2.0, -3.0, 1.0))
    glLight(GL_LIGHT1, GL_AMBIENT, (0.1, 0.1, 0.15, 1.0))
    glLight(GL_LIGHT1, GL_DIFFUSE, (0.2, 0.2, 0.3, 1.0))
    glLight(GL_LIGHT1, GL_SPECULAR, (0.2, 0.2, 0.3, 1.0))
    
    # Material properties for realistic surface appearance
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.4, 0.4, 0.4, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 32.0)
    
    # Load the mesh
    scene = pywavefront.Wavefront(mesh_path, collect_faces=True, create_materials=True, strict=False)
    
    # Load and set up texture
    texture_id = load_texture(texture_path)
    
    # Set up perspective
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    
    # Calculate mesh center and scale
    all_vertices = []
    all_normals = []
    for mesh in scene.mesh_list:
        for face in mesh.faces:
            vertices = [scene.vertices[i] for i in face]
            # Calculate face normal
            v1, v2, v3 = [np.array(v) for v in vertices[:3]]
            normal = np.cross(v2 - v1, v3 - v1)
            normal = normal / np.linalg.norm(normal)
            all_normals.extend([normal] * len(face))
            all_vertices.extend(vertices)
    
    center, scale = normalize_mesh(all_vertices)
    
    # Initial position and zoom
    zoom = -3.0
    min_zoom = -10.0
    max_zoom = -1.0
    zoom_speed = 0.1
    
    # Initial rotation
    rotation_x = 0
    rotation_y = 0
    
    def Model():
        glPushMatrix()
        
        # Apply transformations
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)
        glScalef(1.0/scale, 1.0/scale, 1.0/scale)
        glTranslatef(-center[0], -center[1], -center[2])
        
        # Bind texture and set material properties
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        
        for mesh in scene.mesh_list:
            glBegin(GL_TRIANGLES)
            for face_idx, face in enumerate(mesh.faces):
                # Get vertices and calculate face properties
                vertices = [np.array(scene.vertices[i]) for i in face]
                v1, v2, v3 = vertices[:3]
                
                # Calculate face normal
                normal = np.cross(v2 - v1, v3 - v1)
                normal = normal / np.linalg.norm(normal)
                
                for i, vertex_i in enumerate(face):
                    # Set normal
                    glNormal3f(*normal)
                    
                    # Get or generate texture coordinates
                    if hasattr(mesh, 'tex_coords') and mesh.tex_coords and len(mesh.tex_coords) > vertex_i:
                        u, v = mesh.tex_coords[vertex_i]
                    else:
                        # Generate UVs using spherical mapping
                        u, v = calculate_uv_mapping(scene.vertices[vertex_i], normal)
                    
                    # Apply texture coordinates with proper wrapping
                    glTexCoord2f(u, 1.0 - v)
                    
                    # Set vertex
                    glVertex3f(*scene.vertices[vertex_i])
            glEnd()
        
        glPopMatrix()
    
    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    rotation_y -= 5
                elif event.key == K_RIGHT:
                    rotation_y += 5
                elif event.key == K_UP:
                    rotation_x -= 5
                elif event.key == K_DOWN:
                    rotation_x += 5
                elif event.key == K_z:  # Zoom in
                    zoom = min(zoom + zoom_speed, max_zoom)
                elif event.key == K_x:  # Zoom out
                    zoom = max(zoom - zoom_speed, min_zoom)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel up
                    zoom = min(zoom + zoom_speed, max_zoom)
                elif event.button == 5:  # Mouse wheel down
                    zoom = max(zoom - zoom_speed, min_zoom)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, zoom)
        Model()
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    mesh_path = "test_images/stanfordbunny.obj"  # Use forward slashes
    texture_path = "test_images/Onyx006_texture.jpg"  # Use forward slashes
    render_textured_mesh(mesh_path, texture_path)