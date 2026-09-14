"""
UV Parameterization & 4K Texture Atlas Baker (Agent Group 3)
Bakes original 4K DSLR pixels directly onto 3D polygons using
multi-view cosine ray blending and projective texture sampling.
"""

import os
import numpy as np
from PIL import Image
from luther_core.types import SurfaceMesh, CameraView
from luther_core.safe_memory import MemoryGuardian
from luther_texture.projective_baker import ProjectiveTextureBaker
from luther_geometry.manifold_cleaner import export_mesh_to_ply

def export_scene_forensic_obj(
    obj_path: str,
    mtl_path: str,
    diffuse_img_path: str,
    mesh: SurfaceMesh
):
    """
    Exports complete production Wavefront OBJ + MTL + 4K Diffuse Texture Map.
    """
    os.makedirs(os.path.dirname(os.path.abspath(obj_path)), exist_ok=True)
    mtl_filename = os.path.basename(mtl_path)
    diffuse_filename = os.path.basename(diffuse_img_path)
    
    # 1. Write MTL
    with open(mtl_path, "w") as f:
        f.write("# lutherICPU Forensic Material\n")
        f.write("newmtl Material_Forensic_4K\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.100 0.100 0.100\n")
        f.write("Ns 10.0\n")
        f.write("d 1.0\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {diffuse_filename}\n")
        
    # 2. Write OBJ
    with open(obj_path, "w") as f:
        f.write("# lutherICPU Forensic 3D Scene Asset\n")
        f.write(f"mtllib {mtl_filename}\n")
        f.write("usemtl Material_Forensic_4K\n")
        
        # Vertices (with optional vertex colors)
        vc = mesh.vertex_colors if mesh.vertex_colors is not None else None
        for i, v in enumerate(mesh.vertices):
            if vc is not None:
                c = vc[i]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        # Normals
        normals = mesh.normals if mesh.normals is not None else np.tile([0.0, 0.0, 1.0], (len(mesh.vertices), 1))
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
        # Faces
        for face in mesh.faces:
            i1, i2, i3 = face[0] + 1, face[1] + 1, face[2] + 1
            f.write(f"f {i1}//{i1} {i2}//{i2} {i3}//{i3}\n")

def run_texture_baker(
    mesh: SurfaceMesh,
    views: dict,
    out_diffuse_path: str,
    out_obj_path: str,
    out_ply_path: str,
    atlas_res: int = 4096,
    guardian: MemoryGuardian = None
) -> dict:
    """
    Executes Agent Group 3:
    1. Bakes keyframe DSLR photographs onto the 4096x4096 texture map.
    2. Saves high-res diffuse map and exports production OBJ/MTL/PLY.
    """
    if guardian:
        guardian.checkpoint("Texture Baker: Multi-view Projection")
        
    view_list = list(views.values()) if isinstance(views, dict) else views
    baker = ProjectiveTextureBaker(atlas_resolution=atlas_res, min_cos_angle=0.15)
    
    tex_map, vert_colors = baker.bake_photorealistic_texture(
        mesh=mesh,
        views=view_list,
        max_cameras_to_blend=24
    )
    mesh.texture_map = tex_map
    mesh.vertex_colors = vert_colors
    
    # 2. Save Diffuse Texture Map
    os.makedirs(os.path.dirname(os.path.abspath(out_diffuse_path)), exist_ok=True)
    img = Image.fromarray(tex_map)
    img.save(out_diffuse_path, format="PNG")
    
    # 3. Export OBJ & MTL
    mtl_path = out_obj_path.rsplit(".", 1)[0] + ".mtl"
    export_scene_forensic_obj(
        obj_path=out_obj_path,
        mtl_path=mtl_path,
        diffuse_img_path=out_diffuse_path,
        mesh=mesh
    )
    
    # 4. Export PLY
    export_mesh_to_ply(mesh, out_ply_path)
    
    if guardian:
        guardian.checkpoint("Texture Baker: Complete")
        
    return {
        "texture_map": tex_map,
        "vertex_colors": vert_colors,
        "size": atlas_res,
        "image": img
    }
