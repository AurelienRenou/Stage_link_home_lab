import bioviz
import numpy as np

model_name = "models/double_pendulum_with_contact_z.bioMod"
# Load the model - for bioviz
biorbd_viz = bioviz.Viz(
    model_name,
    show_floor=False,
    show_gravity_vector=False,
    show_meshes=True,
    show_global_center_of_mass=False,
    show_segments_center_of_mass=False,
    show_global_ref_frame=False,
    show_local_ref_frame=True,
    show_markers=False,
    show_muscles=False,
    show_wrappings=False,
    mesh_opacity=0.99,
)
biorbd_viz.exec()
