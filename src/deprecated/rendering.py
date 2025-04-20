import os
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_LEFT, K_RIGHT, K_UP, K_DOWN
from OpenGL.GL import glBegin, glEnd, glVertex3f, glPushMatrix, glPopMatrix, glScalef, glTranslatef, glRotatef, glClear, glPolygonMode, glClearColor, glEnable, glDisable, GL_TRIANGLES, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_FRONT_AND_BACK, GL_LINE, GL_FILL, GL_DEPTH_TEST, glBindTexture, glTexImage2D, glTexParameteri, GL_TEXTURE_2D, GL_LINEAR, GL_REPEAT, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_RGB, GL_UNSIGNED_BYTE, glGenTextures, glTexCoord2f
from OpenGL.GLU import gluPerspective
import pywavefront
from PIL import Image

def view_original_mesh(obj_filepath, texture_filepath=None):
    # Load the .obj file and ignore missing .mtl files
    scene = pywavefront.Wavefront(obj_filepath, collect_faces=True, parse=True, create_materials=True)

    scene_box = (scene.vertices[0], scene.vertices[0])
    for vertex in scene.vertices:
        min_v = [min(scene_box[0][i], vertex[i]) for i in range(3)]
        max_v = [max(scene_box[1][i], vertex[i]) for i in range(3)]
        scene_box = (min_v, max_v)

    scene_size = [scene_box[1][i] - scene_box[0][i] for i in range(3)]
    max_scene_size = max(scene_size)
    scaled_size = 5
    scene_scale = [scaled_size / max_scene_size for i in range(3)]
    scene_trans = [-(scene_box[1][i] + scene_box[0][i]) / 2 for i in range(3)]

    def main():
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (display[0] / display[1]), 1, 500.0)
        glTranslatef(0.0, 0.0, -10)

        texture_id = None
        if texture_filepath:
            # Load the texture image after OpenGL context is initialized
            texture_image = Image.open(texture_filepath)
            texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
            texture_data = texture_image.convert("RGB").tobytes()

            # Generate and bind the texture
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_image.width, texture_image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        def Model():
            glPushMatrix()
            glScalef(*scene_scale)
            glTranslatef(*scene_trans)

            if texture_id:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, texture_id)

            for mesh in scene.mesh_list:
                has_tex_coords = hasattr(mesh, 'tex_coords') and mesh.tex_coords
                glBegin(GL_TRIANGLES)
                for face in mesh.faces:
                    for vertex_i in face:
                        if texture_id and has_tex_coords and len(mesh.tex_coords) > vertex_i:
                            # Normalize texture coordinates to ensure proper wrapping
                            u, v = mesh.tex_coords[vertex_i]
                            glTexCoord2f(u, 1 - v)  # Flip the V coordinate for OpenGL
                        glVertex3f(*scene.vertices[vertex_i])
                glEnd()

            if texture_id:
                glDisable(GL_TEXTURE_2D)

            glPopMatrix()

        while True:
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

    main()

if __name__ == "__main__":
    obj_filepath = "test_images/stanfordbunny.obj"
    texture_filepath = "test_images/Onyx006_texture.jpg"  # Path to the texture image
    if os.path.exists(obj_filepath):
        view_original_mesh(obj_filepath)
    else:
        print(f"Error: File not found at {obj_filepath}")