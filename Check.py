import os
import torch
import warnings
import torch_geometric
from torch_geometric.data import HeteroData

# 1. Mute core PyTorch script versioning logs to keep terminal clean
warnings.filterwarnings("ignore", category=FutureWarning)

# 2. Configure output parameters
output_dir = '/home/fader/automated_diagram_tensors'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'hetero_base_topology.pt')

# 3. Securely load your raw foundation topology
master_data = torch.load('/home/fader/automated_diagram_tensors/CAM Topology.pt', weights_only=False)
print("📦 Graph loaded successfully from disk!")

# 4. Instantiate a clean Heterogeneous Graph Container
hetero_data = HeteroData()

# 5. Categorize your nodes using your string name registry
compound_indices = []
process_indices = []

# Unified CAM pathway structural tracking signatures
target_keywords = [
    'coz', 'oaa', 'malate', 'storage', 'vacuole', 'chos', 'pep', 'pyr _', 'accoa', 
    'citrate', 'suc', 'atp (maintenance)', 'pyr', 'opo3-2', 'light', 'dark', 
    'phosphoenolpyruvate (pep)', 'amp', 'ppck', 'hzo', 'ppdk-rp', 'atp', 'ppdk', 
    'pepc', 'c `', 'pyruvate', 'oxaloacetate (oaa)', 'nadh', 'nadph', 'nadp-me', 
    'ho', 'mdh', 'nad+', 'ho_', 'nadp+', 'oh', 'almt', '(s)', 'malic acid', 
    'tdt', 'gz0', 'pyr_', 'ala', 'tcac', 'cbbc', '2-og+', 'glu', 'tp', 
    'transport to vein', '2-og', 'asp', 'citrate-'
]

for idx, name in enumerate(master_data.node_names):
    clean_name = name.lower()
    if any(keyword in clean_name for keyword in target_keywords):
        process_indices.append(idx)
    else:
        compound_indices.append(idx)

# 6. Slice your raw coordinate vectors into separate node types
hetero_data['compound'].x = master_data.x[compound_indices]
hetero_data['process'].x = master_data.x[process_indices]

# Save text metadata cleanly within each structural node type
hetero_data['compound'].names = [master_data.node_names[i] for i in compound_indices]
hetero_data['process'].names = [master_data.node_names[i] for i in process_indices]

# 7. EXPORT DATA: Explicitly write the structured heterogeneous graph object back to disk
torch.save(hetero_data, output_path)

# ==============================================================================
# VERIFICATION SCREEN PRINTOUTS
# ==============================================================================
print("\n🏆 Converted Multi-Layer Heterogeneous Base Created!")
print(hetero_data)
print(f"💾 Saved a distinct foundation file directly to: {output_path}")

print("\n--- Original Source Summary ---")
print(master_data)
print("Total Unique Global Nodes Recorded:", len(master_data.node_names))

