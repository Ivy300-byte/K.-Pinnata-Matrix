import os
import re
import numpy as np
import torch
import easyocr
from PIL import Image
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
input_dir = "/home/fader/local_extracted_microscopy/CAM Diagram Schemes" 
output_dir = "/home/fader/automated_diagram_tensors"
os.makedirs(output_dir, exist_ok=True)

valid_extensions = ('.png', '.jpg', '.jpeg')
diagram_files = [os.path.join(root, f) for root, _, files in os.walk(input_dir) for f in files if f.lower().endswith(valid_extensions)]

master_node_names = []
master_node_coords = []
master_edge_list = []
name_to_global_idx = {}

# ==============================================================================
# 2. BATCH OCR NODE GATHERING (STABILIZED)
# ==============================================================================
if not diagram_files:
    print(f"⚠️ Pipeline Alert: No diagrams located in {input_dir}.")
else:
    print(f"Located {len(diagram_files)} raw diagrams. Building unified topology...")
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    for img_path in diagram_files:
        file_name = os.path.basename(img_path)
        print(f"Processing Text Layer from: {file_name}")
        
        try:
            # Open with PIL to force standard 8-bit unsigned integer channels
            pil_img = Image.open(img_path).convert("L")
            img_matrix = np.array(pil_img, dtype=np.uint8)
            
            ocr_results = reader.readtext(img_matrix)
            local_nodes = []

            for bbox, raw_text, confidence in ocr_results:
                clean_text = raw_text.strip()
                
                if len(clean_text) > 1 and confidence > 0.4:
                    # Safely extract bounding box anchors
                    tl_x, tl_y = bbox[0][0], bbox[0][1]
                    br_x, br_y = bbox[2][0], bbox[2][1]
                    
                    cx = (tl_x + br_x) / 2.0
                    cy = (tl_y + br_y) / 2.0
                    
                    if clean_text not in name_to_global_idx:
                        name_to_global_idx[clean_text] = len(master_node_names)
                        master_node_names.append(clean_text)
                        master_node_coords.append([cx, cy])
                        
                    local_nodes.append(name_to_global_idx[clean_text])
            
            # Record diagram layout sequence relationships
            if len(local_nodes) >= 2:
                for idx in range(len(local_nodes) - 1):
                    u = local_nodes[idx]
                    v = local_nodes[idx + 1]
                    master_edge_list.append([u, v])
                    master_edge_list.append([v, u])
                    
        except Exception as e:
            print(f"❌ Error bypass on {file_name}: {e}")
            continue

    # ==============================================================================
    # 3. AUTOMATED TOPOLOGY COMPILATION (SCIKIT-LEARN CONVERGENCE)
    # ==============================================================================
    if master_node_names:
        master_data = Data()
        master_data.x = torch.tensor(master_node_coords, dtype=torch.float)
        master_data.node_names = master_node_names
        
        # Pull text arrangement sequence indices
        if master_edge_list:
            base_edges = torch.tensor(master_edge_list, dtype=torch.long).t().contiguous()
        else:
            base_edges = torch.empty((2, 0), dtype=torch.long)
            
        # PURE PYTHON SCIKIT-LEARN KNN FALLBACK
        # Calculates spatial neighbor coordinates without C++ compilation requirements
        coords_np = np.array(master_node_coords)
        k_neighbors = min(4, len(coords_np)) # Guard check ensuring k doesn't exceed total node size
        
        knn_src = []
        knn_dst = []
        
        if len(coords_np) >= 2:
            nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(coords_np)
            distances, indices = nbrs.kneighbors(coords_np)
            
            for i in range(len(coords_np)):
                for neighbor_idx in indices[i]:
                    if i != neighbor_idx: # Prevent self-loops (cells pointing to themselves)
                        knn_src.extend([i, neighbor_idx])
                        knn_dst.extend([neighbor_idx, i])
                        
        knn_edges = torch.tensor([knn_src, knn_dst], dtype=torch.long)
        
        # Combine sequential file loops with geometric proximity calculations
        if base_edges.shape[1] > 0 and knn_edges.shape[1] > 0:
            master_data.edge_index = torch.cat([base_edges, knn_edges], dim=1)
        elif knn_edges.shape[1] > 0:
            master_data.edge_index = knn_edges
        else:
            master_data.edge_index = base_edges
            
        # Clean up duplicates
        if master_data.edge_index.shape[1] > 0:
            master_data.edge_index = torch.unique(master_data.edge_index, dim=1)

        final_output_path = os.path.join(output_dir, "unified_diagram_topology.pt")
        torch.save(master_data, final_output_path)
        
        print("\n🏆 Master Network Topology Successfully Formed!")
        print(f"Total Unique Nodes Logged (x): {master_data.x.shape} (Compounds discovered across your papers)")
        print(f"Total Combined Process Arrows (edge_index): {master_data.edge_index.shape} (Directed Edges constructed via Scikit-Learn)")
        print(f"Final Complete Node Registry: {master_data.node_names}")
        print(f"\nSaved completed file directly to: {final_output_path}")
