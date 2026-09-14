import os
import trimesh
import numpy as np
from PIL import Image

def analyze_and_prepare_deliverables():
    obj_path = "output/scene_textured.obj"
    tex_path = "output/scene_textured_material_00_map_Kd.jpg"
    
    print(f"Checking {obj_path}...")
    if not os.path.exists(obj_path):
        print("OBJ not found!")
        return
        
    mesh = trimesh.load(obj_path, process=False)
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Check bounds
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    print(f"Center: {center}, Extents: {extents}")
    print(f"Has visual: {mesh.visual is not None}")
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        print(f"UV coordinates shape: {mesh.visual.uv.shape}")
        
    # Check texture image
    if os.path.exists(tex_path):
        im = Image.open(tex_path)
        print(f"Texture image: {im.size}, mode: {im.mode}")
        # Save as PNG as well for compatibility
        im.save("output/scene_textured.png")
        im.save("output/truck_diffuse.png")
        print("Saved truck_diffuse.png from OpenMVS texture map.")
        
    # Export as glb / glTF for ultra fast Three.js loading if needed
    try:
        glb_bytes = mesh.export(file_type='glb')
        with open("output/scene_textured.glb", "wb") as f:
            f.write(glb_bytes)
        print(f"Exported scene_textured.glb ({len(glb_bytes)} bytes)")
    except Exception as e:
        print("GLB export notice:", e)

if __name__ == "__main__":
    analyze_and_prepare_deliverables()
