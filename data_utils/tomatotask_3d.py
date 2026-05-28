import os
import glob
import torch
import numpy as np
from tqdm import tqdm

import trimesh
import open3d as o3d 
from plyfile import PlyData
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

#from utils.mesh_helpers import mesh_downsample, compute_downsample_matrix, preprocess_spiral, to_sparse

classes = {
    0: 'control',
    1: 'bacterial_spot',
    2: 'septoria_leaf_spot',
    3: 'early_blight'
}
   
def mesh_to_graph(verts, faces, features):
    """
    Convert mesh vertices/faces to PyG graph (Data object).
    """
    faces = torch.as_tensor(faces, dtype=torch.long)
    
    edges = torch.cat([
        faces[:, [0,1]],
        faces[:, [1,0]],
        faces[:, [1,2]],
        faces[:, [2,1]],
        faces[:, [2,0]],
        faces[:, [0,2]],
    ], dim=0)

    edge_index = edges.t().contiguous()
    edge_index = coalesce(edge_index)
    
    x = torch.as_tensor(features, dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index)

class TomatoTask3D(torch.utils.data.Dataset):
    """
    Dataset for loading PLY meshes and preparing SpiralNet structures.
    """

    def __init__(
        self,
        root,
        representation="mesh", # "mesh", "spiral", "graph", "pcd"
        seq_lengths=[21,21,21],
        ds_factors=[4, 4, 4],
        target_faces=None,
    ):
        self.representation = representation
        self.seq_lengths = seq_lengths
        self.ds_factors = ds_factors
        
        if isinstance(target_faces, str):
            if target_faces.lower() == "none":
                target_faces = None
            else:
                target_faces = int(target_faces)
        self.target_faces = target_faces
            
        all_files = sorted(glob.glob(os.path.join(root, "**", "*.ply"), recursive=True))

        candidate_files = [
            f for f in all_files
            if 'trian' not in os.path.basename(f)
            and not any(d in f for d in ["DAI22", "DAI25", "DAI28"])
        ]#[:10]

        # Keep only valid PLYs
        self.mesh_files = []
        for f in tqdm(candidate_files, desc="Identifying Candidate Files", total=len(candidate_files), unit="file"):
            verts, faces, feats = self.load_ply(f)
            if verts is not None:
                self.mesh_files.append(f)

    @staticmethod
    def load_ply(path):
        """
        Loads a PLY mesh, applies a z > 5 mask, remaps faces, 
        and returns masked vertices, faces, and per-vertex features.
        """
        try:
            ply = PlyData.read(path)

            vertex_data = ply['vertex'].data
            x = vertex_data['x']
            y = vertex_data['y']
            z = vertex_data['z']

            mask = z > 6
            orig_idx = np.arange(len(z))
            new_idx = np.cumsum(mask) - 1

            # Filter vertices
            vertices = np.stack([x[mask], y[mask], z[mask]], axis=-1).astype(np.float32)

            face_data = ply['face'].data
            if 'vertex_indices' in face_data.dtype.names:
                faces_raw = np.vstack(face_data['vertex_indices'])
            elif 'vertex_index' in face_data.dtype.names:
                faces_raw = np.vstack(face_data['vertex_index'])
            else:
                raise ValueError("No face index field in PLY.")

            # Keep faces where *all* referenced vertices survived masking
            face_mask = mask[faces_raw].all(axis=1)
            faces_filtered = faces_raw[face_mask]

            # Remap indices to new masked vertex order
            faces = new_idx[faces_filtered]

            colors = [
                vertex_data[c][mask]
                for c in ['red','green','blue']
                if c in vertex_data.dtype.names
            ]
            if colors:
                colors = np.stack(colors, axis=-1).astype(np.float32)
                # Normalize RGB
                if colors[..., :3].max() > 1.0:
                    colors[..., :3] /= 255.0
            else:
                colors = None

            if "scalar_nir" in vertex_data.dtype.names:
                nir = vertex_data["scalar_nir"][mask].astype(np.float32)
                nir /= 255.0
            else:
                nir = None

            features = np.concatenate(
                [f for f in [colors, nir[:, None] if nir is not None else None] if f is not None],
                axis=-1
            )

            return vertices, faces, features

        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return None, None, None
    
    def __len__(self):
        return len(self.mesh_files)
    
    @staticmethod
    def _compute_vertex_normals(verts, faces):
        """
        Compute per-vertex normals by averaging adjacent face normals.
        verts: [V,3]
        faces: [F,3]
        """

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0, axis=1)
        face_normals = face_normals / (           
            np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
        )

        vertex_normals = np.zeros_like(verts)

        # accumulate face normals onto vertices
        for i in range(3):
            #vertex_normals.index_add_(0, faces[:, i], face_normals)
            np.add.at(vertex_normals, faces[:, i], face_normals)
            
        vertex_normals = vertex_normals / (
            np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-8

        )

        return vertex_normals
    
    @staticmethod
    def _compute_face_centers(verts, faces):
        return verts[faces].mean(axis=1).astype(np.float32)

    @staticmethod
    def _compute_face_normals(verts, faces):
        v0 = verts[faces[:,0]]; v1 = verts[faces[:,1]]; v2 = verts[faces[:,2]]
        normals = np.cross(v1-v0, v2-v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(norms, 1e-8)
        return normals.astype(np.float32)

    @staticmethod
    def _compute_face_rings(faces):
        from collections import defaultdict
        F = faces.shape[0]
        edge_to_face = defaultdict(list)
        for fi, f in enumerate(faces):
            edges = [(f[i], f[(i+1)%3]) for i in range(3)]
            for a,b in edges: edge_to_face[tuple(sorted((a,b)))].append(fi)
        # 1-ring
        ring_1 = []
        for fi, f in enumerate(faces):
            neighbors = set()
            edges = [(f[i], f[(i+1)%3]) for i in range(3)]
            for a,b in edges: neighbors.update(edge_to_face[tuple(sorted((a,b)))])
            neighbors.discard(fi)
            neighbors = list(neighbors)
            while len(neighbors)<3: neighbors.append(fi)
            ring_1.append(neighbors[:3])
        ring_1 = np.array(ring_1, dtype=np.int64)
        # higher rings
        prev_ring = ring_1
        rings = {1: ring_1}
        for r, num in zip([2,3],[6,12]):
            ring_r = []
            for fi in range(F):
                neighbors = set(prev_ring[fi])
                for n in prev_ring[fi]: neighbors.update(prev_ring[n])
                neighbors.discard(fi)
                neighbors = list(neighbors)
                while len(neighbors)<num: neighbors.append(fi)
                ring_r.append(neighbors[:num])
            ring_r = np.array(ring_r, dtype=np.int64)
            rings[r] = ring_r
            prev_ring = ring_r
        return rings
    """
    def _build_mesh_pyramid(self, verts, faces):
        vertices = [verts]
        face_list = [faces]
        down_transforms = []

        # Create hierarchy
        for factor in self.ds_factors:
            v_hi, f_hi = vertices[-1], face_list[-1]
            v_lo, f_lo = mesh_downsample(v_hi, f_hi, factor)
            D = compute_downsample_matrix(v_hi, v_lo)

            vertices.append(v_lo)
            face_list.append(f_lo)
            down_transforms.append(D)

        # Spirals (only up to second-to-last level)
        spirals = [
            preprocess_spiral(face_list[i], self.seq_lengths[i], vertices[i])
            for i in range(len(self.seq_lengths))
        ]

        return vertices, face_list, down_transforms, spirals
    """
    @staticmethod
    def simplify_mesh(verts, faces, features, target_faces):
        """
        Simplify mesh to target face count while preserving vertex features.
        """
        if len(faces) > int(target_faces):
            #return verts, faces, features

            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(verts),
                triangles=o3d.utility.Vector3iVector(faces),
            )

            simplified = mesh.simplify_quadric_decimation(int(target_faces))
            new_verts = np.asarray(simplified.vertices).astype(np.float32)
            new_faces = np.asarray(simplified.triangles).astype(np.int64)

            used_verts = np.unique(new_faces.flatten())
            new_verts = new_verts[used_verts]
            
            # Map old vertex indices to new via nearest neighbor
            from scipy.spatial import cKDTree
            tree = cKDTree(verts)
            _, idx = tree.query(new_verts, k=3)
            new_features = features[idx].mean(axis=1)

            # remap faces
            mapping = {old: new for new, old in enumerate(used_verts)}
            new_faces = np.vectorize(mapping.get)(new_faces)
            
            verts, faces, features = new_verts, new_faces, new_features
        
        actual = len(faces)

        # Enforce exact count by trimming or padding with repeats
        if actual > int(target_faces):
            faces = faces[:int(target_faces)]
        elif actual < int(target_faces):
            # Pad by repeating faces
            pad = int(target_faces) - actual
            repeats = (pad // actual) + 1
            faces = np.concatenate([faces, np.tile(faces, (repeats, 1))[:pad]], axis=0)

        assert len(faces) == int(target_faces), f"Expected {int(target_faces)} faces, got {len(faces)}"
        return verts, faces, features
    
    @staticmethod
    def _normalize_point_count(verts, feats, normals, target_n):
        actual = verts.shape[0]
        if actual > int(target_n):
            # Random subsample
            idx = np.random.choice(actual, int(target_n), replace=False)
        else:
            # Pad with repeats
            idx = np.concatenate([
                np.arange(actual),
                np.random.choice(actual, int(target_n) - actual, replace=True)
            ])
        return verts[idx], feats[idx], normals[idx]
    
    def __getitem__(self, idx):
        fp = self.mesh_files[idx]

        verts, faces, feats = self.load_ply(fp)
        label = int(os.path.basename(fp).split("_")[1].lstrip("T"))
        label = torch.tensor(label, dtype=torch.long)
        
        if self.representation == "pcd":            
            normals = self._compute_vertex_normals(verts, faces)
            
            if self.target_faces is not None:
                verts, feats, normals = self._normalize_point_count(
                    verts, feats, normals, target_n=self.target_faces  # reuse same arg or add target_points
                )
    
            normals = torch.as_tensor(normals, dtype=torch.float32)
            
            verts = torch.tensor(verts, dtype=torch.float32)     # [N,3]
            feats = torch.tensor(feats, dtype=torch.float32)     # [N,C]
            points = torch.cat([verts, feats, normals], dim=1)   # [N, 3+C+3]

            return points, label
        
        elif self.representation == "mesh":
            if self.target_faces is not None:
                verts, faces, feats = self.simplify_mesh(
                    verts, faces, feats, self.target_faces
                )
            
            centers = self._compute_face_centers(verts, faces)
            normals = self._compute_face_normals(verts, faces)
            rings = self._compute_face_rings(faces)
            
            face_feats = feats[faces].mean(axis=1) 
            
            mesh = {
                "verts": torch.from_numpy(verts).float().clone(),
                "vert_feats": torch.from_numpy(feats).float().clone().permute(1, 0),
                "face_feats": torch.from_numpy(face_feats).float().clone().permute(1, 0),# average vertex feats per face
                "faces": torch.from_numpy(faces).long().clone(),
                "centers": torch.from_numpy(centers).float().clone().permute(1, 0),
                "normals": torch.from_numpy(normals).float().clone().permute(1, 0),
                "ring_1": torch.from_numpy(rings[1]).long().clone(),
                "ring_2": torch.from_numpy(rings[2]).long().clone(),
                "ring_3": torch.from_numpy(rings[3]).long().clone(),
            }
            return mesh, label
        
        elif self.representation == "graph":
            if self.target_faces is not None:
                verts, faces, feats = self.simplify_mesh(
                    verts, faces, feats, self.target_faces
                )
            
            x = np.concatenate([verts, feats], axis=-1)  # XYZ + RGB/NIR
            #x = verts   
                     
            data = mesh_to_graph(verts, faces, x)
            data.y = label
            return data
        """
        elif self.representation == "spiral":
            if self.target_faces is not None:
                verts, faces, feats = self.simplify_mesh(
                    verts, faces, feats, self.target_faces
                )
            
            try:
                vertices, face_list, down_transforms, spirals = self._build_mesh_pyramid(verts, faces)
            except Exception as e:
                print(f"Skipping mesh {fp} due to error: {e}")
                return self.__getitem__((idx+1) % len(self.mesh_files))  # skip to next
            
            top_feats = np.concatenate([vertices[0], feats], axis=-1)  # [V, 3+C]

            # Propagate features down through pyramid
            feat_levels = [top_feats]
            for D in down_transforms:
                # D: [n_lo, n_hi], feat: [n_hi, C] → [n_lo, C]
                feat_down = D.dot(feat_levels[-1])  # sparse matmul, stays numpy
                feat_levels.append(feat_down)
            
            return {
                "features": [
                    torch.tensor(f, dtype=torch.float32).unsqueeze(0)  # [1, V, C]
                    for f in feat_levels
                ],
                "spiral_indices": [torch.tensor(s, dtype=torch.long) for s in spirals],
                "down_transform": [to_sparse(D) for D in down_transforms],
                "label": label
            }"""