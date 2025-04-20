import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import pywavefront
from PIL import Image
import numpy as np

def load_texture(image_path):
    texture_id = glGenTextures(1)
    img = Image.open(image_path).convert('RGBA')
    img_data = np.array(img)

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return texture_id

def render_textured_mesh(mesh_path, texture_path):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)

    # Load the mesh and ignore missing .mtl files
    scene = pywavefront.Wavefront(mesh_path, collect_faces=True, parse=True, create_materials=True)

    # Load texture
    texture_id = load_texture(texture_path)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0.0, 0.0, -5)
    glScalef(2.0, 2.0, 2.0)  # Increase the scale of the object

    def Model():
        glBindTexture(GL_TEXTURE_2D, texture_id)
        for name, mesh in scene.meshes.items():
            for material in mesh.materials:
                vertices = material.vertices
                stride = 8  # 3 position, 3 normal, 2 texture coordinates
                has_tex_coords = len(vertices) % stride == 0  # Ensure texture coordinates exist
                glBegin(GL_TRIANGLES)
                for i in range(0, len(vertices), stride):
                    if has_tex_coords:
                        u, v = vertices[i+6], vertices[i+7]
                        glTexCoord2f(u, 1 - v)  # Flip V coordinate for OpenGL
                    glVertex3f(vertices[i], vertices[i+1], vertices[i+2])
                glEnd()

    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                quit()

            if event.type == KEYDOWN:
                if event.key == K_LEFT:
                    glTranslatef(-0.5, 0, 0)
                if event.key == K_RIGHT:
                    glTranslatef(0.5, 0, 0)
                if event.key == K_UP:
                    glTranslatef(0, 1, 0)
                if event.key == K_DOWN:
                    glTranslatef(0, -1, 0)

        glRotatef(1, 5, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        Model()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()


if __name__ == "__main__":
    mesh_path = "test_images/stanfordbunny.obj"  # Replace with your .obj file path
    texture_path = "test_images/Onyx006_texture.jpg"  # Replace with your texture image path
    render_textured_mesh(mesh_path, texture_path)
