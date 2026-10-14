import os
import glob
import torch
import numpy as np
import pandas as pd
import hashlib

import open3d as o3d 
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

from data_utils.spiral_helpers import build_spiral_pyramid

classes = {
    0: 'control',
    1: 'bacterial_spot',
    2: 'septoria_leaf_spot',
    3: 'early_blight'
}

reg_scale = list(range(13))

# PhenoSpex PlantEye Scanner Angles
t1, t2 = np.deg2rad(20.1), np.deg2rad(-19.8)
VIEWS = np.array([[0.0, np.sin(t1), np.cos(t1)],
                  [0.0, np.sin(t2), np.cos(t2)]], dtype=np.float32)

# Model input options
FEATURE_SLICES = {"geom": slice(0, 0), "rgb": slice(0, 3), "rgbn": slice(0, 4)}
FEAT_CH = {"geom": 0, "rgb": 3, "rgbn": 4}
REPR = {"pointclouds": "pcd", "meshes": "mesh", "spirals": "spiral", "graph_meshes": "graph"}

def num_input_channels(dataset_name, features):
    f = FEAT_CH[features]
    rep = REPR[dataset_name]
    return {"pcd": 3 + f + 3, "graph": 3 + f, "spiral": 3 + f, "mesh": f}[rep]

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

    edge_index, _ = coalesce(edges.t().contiguous(), None)
    x = torch.as_tensor(features, dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index)

class TomatoTask3D(torch.utils.data.Dataset):
    """
    PLY mesh dataset supporting four representations:
 
      "pcd"    — point cloud  [N, 3 (+ C) + 3]
      "mesh"   — face-centric dict (centers, normals, face-rings)
      "graph"  — PyG Data with edge_index
      "spiral" — SpiralNet++ pyramid (multi-level spiral indices + pool matrices)
    """

    def __init__(
        self,
        root: str,
        split: str,
        representation: str="mesh", # "mesh", "spiral", "graph", "pcd",
        features: str="rgbn", # geom, rgb, rgbn
        seq_lengths=[21,21,21],
        ds_factors=[4, 4, 4],
        dilations=[1,1,1],
        target_faces=None,
        num_points=None,
        decimation_ratio=None,
        num_tasks: str="1", 
        task: str="classification",
        seed: int = 42,
        cache_dir: str="spiral_cache/",
    ):
        self.seed = seed
        self.split = split
        self.representation = representation
        self.features = features
        
        self.num_tasks = num_tasks
        self.task = task
        
        self.seq_lengths = seq_lengths
        self.ds_factors = ds_factors
        self.dilations = dilations        
        self.pool_mode = "mean"
        
        self.cache_dir = cache_dir
        self.cache_dir = cache_dir if representation == "spiral" else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # HortControl's merge resolution
        self.voxel_size_mm = 3.0
        
        # for coordinate normalization
        self.coord_scale = 446.75405273437497 # sub with global scaling
                
        if isinstance(target_faces, str):
            target_faces = None if target_faces.lower() == "none" else int(target_faces)
        self.target_faces = target_faces
        
        if isinstance(decimation_ratio, str):
            decimation_ratio = None if decimation_ratio.lower() == "none" else float(decimation_ratio)
        self.decimation_ratio = decimation_ratio
        
        if representation == "pcd" and num_points is None:
            if target_faces is not None:
                num_points = target_faces // 2     # F ≈ 2V: match the decimated vertex count
            else:
                raise ValueError("pcd without target_faces requires num_points for batching")
        self.num_points = num_points
            
        exclude_days=() #, "DAI22", "DAI25", "DAI28")     
        
        all_files = sorted(glob.glob(os.path.join(root, split, "*.ply"), recursive=True))

        self.mesh_files = [
            f for f in all_files
            if 'trian' not in os.path.basename(f)
            and os.path.basename(f).split("_")[2] not in exclude_days
        ]
        
        if self.task == "regression":
            severity_df = pd.read_csv(
                os.path.join(root, f"{split}_tomato_features.csv"),
                usecols=["polygon_file", "HB Score"],
            )

            def _key(path):
                # basename, no extension, lowercase -> "100x_t02_dai10_20250420"
                return os.path.splitext(os.path.basename(str(path).strip()))[0].lower()

            severity_df["key"] = severity_df["polygon_file"].map(_key)
            severity_df = severity_df.dropna(subset=["HB Score"]).drop_duplicates("key")
            score_map = dict(zip(severity_df["key"], severity_df["HB Score"]))
            
            missing = [os.path.basename(f) for f in self.mesh_files if _key(f) not in score_map]
            if missing:
                raise KeyError(f"[{split}] {len(missing)} meshes have no HB Score, e.g. {missing[:3]}")
            self.scores = [float(score_map[_key(f)]) for f in self.mesh_files]

    @staticmethod
    def load_ply(path):
        """
        Load a PLY mesh, apply a z > 6 mask, remap faces, and return
        (vertices, faces, per-vertex features). 
        """
        try:
            ply = PlyData.read(path)
        except Exception as e:
            raise RuntimeError(f"Could not read PLY file {path}") from e
        
        vertex_data = ply["vertex"].data
        x, y, z = vertex_data["x"], vertex_data["y"], vertex_data["z"]
        
        # Filter vertices
        mask = z > 6
        new_idx = np.cumsum(mask) - 1
        vertices = np.stack([x[mask], y[mask], z[mask]], axis=-1).astype(np.float32)

        face_data = ply["face"].data
        if "vertex_indices" in face_data.dtype.names:
            faces_raw = np.vstack(face_data["vertex_indices"])
        elif "vertex_index" in face_data.dtype.names:
            faces_raw = np.vstack(face_data["vertex_index"])
        else:
            raise ValueError("No face index field found in PLY.")

        # Keep faces where all referenced vertices survived masking
        face_mask = mask[faces_raw].all(axis=1)
        faces = new_idx[faces_raw[face_mask]]

        colors = [vertex_data[c][mask] for c in ("red", "green", "blue") if c in vertex_data.dtype.names]
        if colors:
            colors = np.stack(colors, axis=-1, dtype=np.float32)
            # Normalize RGB
            if colors[..., :3].max() > 1.0: colors[..., :3] /= 255.0
        else:
            colors = None

        if "scalar_nir" in vertex_data.dtype.names:
            nir = vertex_data["scalar_nir"][mask].astype(np.float32)[:, None] 
            if nir.max() > 1.0: nir = nir / 255.0
        else:
            nir = None

        feature_list = [f for f in (colors, nir) if f is not None]
        features = (
            np.concatenate(feature_list, axis=-1)
            if feature_list
            else np.empty((vertices.shape[0], 0), dtype=np.float32)
        )
        
        # clean up points
        used = np.unique(faces)
        remap = np.full(len(vertices), -1, dtype=np.int64); remap[used] = np.arange(len(used))
        vertices, faces, features = vertices[used], remap[faces], features[used]
        
        return vertices, faces, features
    
    def __len__(self):
        return len(self.mesh_files)
    
    def get_label(self, idx: int) -> int:
        if self.task == "classification":
            return int(os.path.basename(self.mesh_files[idx]).split("_")[1].lstrip("T"))
        elif self.task == "regression":
            return self.scores[idx]
            
    def __getitem__(self, idx):      
        # load file
        fp = self.mesh_files[idx]
        if self.representation == "spiral" and self.cache_dir:
            cp = self._cache_path(fp)
            if os.path.exists(cp):
                return torch.load(cp, weights_only=False)
        verts, faces, feats = self.load_ply(fp)
        if feats.shape[1] != 4:
            raise ValueError(f"{fp}: expected RGB+NIR (4 ch), got {feats.shape[1]}")
        feats = feats[:, FEATURE_SLICES[self.features]]
        
        # get label
        if self.task == "classification":
            label = int(os.path.basename(fp).split("_")[1].lstrip("T"))
            label = torch.tensor(label, dtype=torch.long)
        elif self.task == "regression":
            label = torch.tensor(self.scores[idx], dtype=torch.float32)
            
        # orient faces
        faces, _, _ = self._orient_faces(verts, faces, VIEWS)
        
        # normalize verts
        verts = self._normalize_coords(verts)
        
        # mesh decimation
        target = None
        if self.decimation_ratio is not None:
            target = max(4, int(round(len(faces) * self.decimation_ratio)))
        elif self.target_faces is not None:
            target = self.target_faces
        
        if target is not None: #and self.representation != "pcd":
            verts, faces, feats = self.simplify_mesh(
                verts, faces, feats, target,
                exact=(self.representation == "mesh")
                )
            faces, _, _ = self._orient_faces(verts, faces, VIEWS)

        # get data from specified representation
        if self.representation == "pcd":
            return self._build_pcd(verts, faces, feats, label)
        elif self.representation == "mesh":
            return self._build_mesh(verts, faces, feats, label)
        elif self.representation == "graph":
            return self._build_graph(verts, faces, feats, label)
        elif self.representation == "spiral":
            out = self._build_spiral(verts, faces, feats, label)
            if self.cache_dir:
                tmp = f"{cp}.tmp{os.getpid()}"            # atomic write; safe with DDP workers
                torch.save(out, tmp); os.replace(tmp, cp)
            return out
        else:
            raise ValueError(f"Unknown representation: {self.representation!r}")
    
    def _build_pcd(self, verts, faces, feats, label):
        """Point-cloud representation: [N, 3 (+ C) + 3]."""
        _, normals = self._compute_normals(verts, faces)
        
        if self.target_faces is None:
            size = self.voxel_size_mm / self.coord_scale
            verts, feats, normals = self.voxel_downsample(verts, feats, normals, size)
            
        rng = np.random.default_rng(None if self.split == "train" else self.seed)
        if self.num_points is not None:
            verts, feats, normals = self._normalize_point_count(
                verts, feats, normals, target_n=self.num_points, rng=rng
            )
 
        points = [torch.as_tensor(verts, dtype=torch.float32)]
        if self.features != "geom":
            points.append(torch.as_tensor(feats, dtype=torch.float32))
        points.append(torch.as_tensor(normals, dtype=torch.float32))
 
        return torch.cat(points, dim=1), label
    
    def _build_mesh(self, verts, faces, feats, label):
        """
        Mesh representation.  Spectral features are always included in
        vert_feats / face_feats; callers that don't want them can simply
        ignore those keys.
        """
        centers = self._compute_face_centers(verts, faces)
        normals, _ = self._compute_normals(verts, faces)
        rings = self._compute_face_rings(faces)
        face_feats = feats[faces].mean(axis=1)
 
        #if self.target_faces is not None and len(faces) != self.target_faces:
        #    raise ValueError(f"mesh has {len(faces)} faces, expected {self.target_faces}")
        return {
            "verts":      torch.from_numpy(verts).float(),
            "vert_feats": torch.from_numpy(feats).float().T,       # [C, V]
            "face_feats": torch.from_numpy(face_feats).float().T,  # [C, F]
            "faces":      torch.from_numpy(faces).long(),
            "centers":    torch.from_numpy(centers).float().T,     # [3, F]
            "normals":    torch.from_numpy(normals).float().T,     # [3, F]
            "ring_1":     torch.from_numpy(rings[0]).long(),
            "ring_2":     torch.from_numpy(rings[1]).long(),
            "ring_3":     torch.from_numpy(rings[2]).long(),
        }, label
    
    def _build_graph(self, verts, faces, feats, label):
        """Graph representation: node features are XYZ (+ spectral if requested)."""
        if self.features != "geom":
            x = np.concatenate([verts, feats], axis=-1)
        else:
            x = verts
 
        data = mesh_to_graph(verts, faces, x)
        data.y = label
        return data
    
    def _build_spiral(self, verts, faces, feats, label):
        use_spec = self.features != "geom"
        
        base = np.concatenate([verts, feats], axis=-1) if use_spec else verts
        
        out = build_spiral_pyramid(
            verts, faces, base,
            ds_factors=self.ds_factors, seq_lengths=self.seq_lengths, dilations=self.dilations,
            simplify_fn=self.simplify_mesh,
            orient_fn=lambda v, f: self._orient_faces(v, f, VIEWS)[0],
            normals_fn=lambda v, f: self._compute_normals(v, f)[1],
            pool_mode=self.pool_mode,
        )
        out["label"] = label
        
        return out

    def _normalize_coords(self, verts):
        """Normalize and center input mesh to fit in a sphere of radius 1 centered at (0,0,0)"""     
        lo, hi = verts.min(0), verts.max(0)
        c = (lo + hi) / 2
        c[2] = lo[2]                                   # keep height above crop plane
        verts = verts - c
        
        scale = self.coord_scale if self.coord_scale else np.linalg.norm(verts, axis=1).max()
        
        return (verts / scale).astype(np.float32)
    
    @staticmethod
    def _compute_normals(verts, faces):
        """
        Compute per-vertex normals by averaging adjacent face normals.
        verts: [V,3]
        faces: [F,3]
        """
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        vertex_normals = np.zeros_like(verts)

        # accumulate face normals onto vertices
        np.add.at(vertex_normals, faces[:, 0], face_normals)
        np.add.at(vertex_normals, faces[:, 1], face_normals)
        np.add.at(vertex_normals, faces[:, 2], face_normals)
        
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
        vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-8
        return face_normals.astype(np.float32), vertex_normals.astype(np.float32)
    
    @staticmethod
    def _compute_face_centers(verts, faces):
        return verts[faces].mean(axis=1).astype(np.float32)

    @staticmethod
    def _compute_face_rings(faces):
        F = faces.shape[0]
        
        # Build edge from face incidence using all three directed edges per face
        edges = np.concatenate([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ], axis=0)
        
        edges = np.sort(edges, axis=1)        # canonical ordering
        face_ids = np.repeat(np.arange(F), 3) # corresponding face ids
        
        # Group identical edges
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges = edges[order]
        face_ids = face_ids[order]

        # Find edge group boundaries
        diff = np.any(np.diff(edges, axis=0), axis=1)
        edge_id = np.concatenate([[0], np.cumsum(diff)]) # group label per half-edge
        
        # Two faces sharing an edge are adjacent
        E = edge_id[-1] + 1
        
        # Build edge x face incidence (E x F), then adj = inc.T @ inc - diag
        inc = coo_matrix(
            (np.ones(len(face_ids), dtype=bool), (edge_id, face_ids)),
            shape=(E, F)
        ).tocsr()
        adj = (inc.T @ inc).tocsr().astype(bool)
        adj.setdiag(False)
        adj = (adj > 0).astype(np.int32)

        def extract_ring(a, k):
            """Extracts exact k neighbor IDs for each row/face."""
            cx = a.tocoo()
            
            # Sort COO entries by row
            order_ = np.argsort(cx.row, kind="stable")
            rows_, cols_ = cx.row[order_], cx.col[order_]
            splits = np.searchsorted(rows_, np.arange(F + 1))
            
            rings = np.zeros((F, k), dtype=np.int64)
            
            for fi in range(F):
                nbrs = cols_[splits[fi]:splits[fi + 1]]
                n_nbrs = len(nbrs)
                
                if n_nbrs >= k:
                    rings[fi] = nbrs[:k]
                elif n_nbrs > 0:
                    rings[fi, :n_nbrs] = nbrs
                    rings[fi, n_nbrs:] = nbrs[-1]
                else:
                    rings[fi] = fi
            return rings

        I = coo_matrix((np.ones(F, dtype=np.int32), (np.arange(F), np.arange(F))), shape=(F, F)).tocsr()
        
        ring1 = extract_ring(adj, 3)

        reach2 = ((adj @ adj) > 0).astype(np.int32)
        adj2 = reach2 - reach2.multiply(adj + I)
        adj2.eliminate_zeros()
        ring2 = extract_ring(adj2, 6)
        
        reach3 = ((reach2 @ adj) > 0).astype(np.int32)
        adj3 = reach3 - reach3.multiply(adj + adj2 + I)
        adj3.eliminate_zeros()
        ring3 = extract_ring(adj3, 12)
 
        return ring1, ring2, ring3
        
    @staticmethod
    def voxel_downsample(verts, feats, normals, size):
        keys = np.floor(verts / size).astype(np.int64)
        
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        
        def avg(a):
            out = np.zeros((len(counts), a.shape[1]), np.float32)
            np.add.at(out, inv, a)
            return out / counts[:, None]
        
        n = avg(normals); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
        
        return avg(verts).astype(np.float32), avg(feats).astype(np.float32), n.astype(np.float32)
    
    @staticmethod
    def simplify_mesh(verts, faces, features, target_faces: int, exact: bool=True):
        """
        Simplify mesh to target face count while preserving vertex features.
        """
        # cast type
        verts = verts.astype(np.float64)
        faces = faces.astype(np.int32)
        
        # convert to mesh
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts),
            triangles=o3d.utility.Vector3iVector(faces),
        )
        
        # clean mesh
        mesh.remove_duplicated_vertices(); mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
        
        # decimate mesh
        if len(faces) >  int(target_faces):        
            simplified = mesh.simplify_quadric_decimation(int(target_faces), boundary_weight=5.0) # boundary weight will preserve leaf outlines
            simplified.remove_unreferenced_vertices() # clean decimated mesh
            
            mesh = simplified
            
        new_verts = np.asarray(mesh.vertices).astype(np.float32)
        new_faces = np.asarray(mesh.triangles).astype(np.int64)

        used_verts = np.unique(new_faces)
        new_verts = new_verts[used_verts]
        
        # Map old vertex indices to new via nearest neighbor, k=4
        d, idx = cKDTree(verts).query(new_verts, k=4)
        w = 1.0 / (d + 1e-8); w /= w.sum(1, keepdims=True) # inverse distance weighting
        new_features = (features[idx] * w[..., None]).sum(1).astype(np.float32) 

        # remap faces
        mapping = np.empty(used_verts.max() + 1, dtype=np.int64)
        mapping[used_verts] = np.arange(len(used_verts))
        new_faces = mapping[new_faces]
            
        # pad to exact face count (deterministic: repeat the first faces)
        if exact and len(new_faces) < int(target_faces):
            deficit = int(target_faces) - len(new_faces)
            reps = np.arange(deficit) % len(new_faces)
            new_faces = np.concatenate([new_faces, new_faces[reps]], axis=0)
        
        return new_verts, new_faces, new_features
    
    @staticmethod
    def _normalize_point_count(verts, feats, normals, target_n: int, rng):
        """Subsample or pad to exactly `target_n` points (reproducible)."""
        actual = len(verts)
                
        if actual >= int(target_n):
            # Random subsample
            idx = rng.choice(actual, int(target_n), replace=False)
        else:
            # Pad with repeats
            idx = np.concatenate([
                np.arange(actual),
                rng.choice(actual, int(target_n) - actual, replace=True),
            ])
            
        return verts[idx], feats[idx], normals[idx]
    
    @staticmethod
    def _orient_faces(verts, faces, views, fallback=np.array([0, 0, 1.0]), eps=0.05):
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        d = fn @ views.T                                  # [F, 2]
        pos, neg = (d > eps).any(1), (d < -eps).any(1)
        ambiguous = pos == neg                            # faces both sensors, or neither clearly
        flip = (~pos & neg) | (ambiguous & ((fn @ fallback) < 0))
        faces = faces.copy()
        faces[flip] = faces[flip][:, [0, 2, 1]]
        return faces, ambiguous, flip
    
    def _cache_path(self, fp):
        target = None
        if self.decimation_ratio is not None:
            target = self.decimation_ratio
        elif self.target_faces is not None:
            target = self.target_faces
        
        key = repr((target, self.ds_factors, self.seq_lengths, self.dilations,
                    self.features, self.coord_scale, self.pool_mode))
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        
        return os.path.join(self.cache_dir, f"{os.path.splitext(os.path.basename(fp))[0]}_{h}.pt")