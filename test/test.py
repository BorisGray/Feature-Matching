import sys
import os
import subprocess

program = '../feature_matching'
image_object = ' ../data/btn.jpg'
image_scene = ' ../data/screen_shot.jpg'
#program = os.path.join('.', 'bin' , 'Release' , program)

#def features_matching(image_object, image_scene):
#  command = " ".join(
#    [program, ' ', image_object, ' ', image_scene, ' '] )

cmd=program+image_object+image_scene
print cmd
os.system(cmd)

