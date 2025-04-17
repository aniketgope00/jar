import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from dearpygui import dearpygui as dpg
import os


dpg.create_context()

dpg.create_viewport(title="rendering + Texture Mapping Demp", width=500, height=200)





if __name__ == "__main__":
    dpg.setup_dearpygui()
    screen_width = dpg.get_viewport_client_width()
    screen_height = dpg.get_viewport_client_height()
    dpg.set_viewport_width(screen_width)
    dpg.set_viewport_height(screen_height)
    dpg.show_viewport()
    dpg.start_dearpygui()
    