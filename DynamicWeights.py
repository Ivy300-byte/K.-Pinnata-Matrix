import sympy as sp, torch, torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, Linear

NODE_TYPES = ["T", "H2O", "CO2", "Salinity", "Nutrients", "hv"]
PATHWAY_SCHEMA = [
    ("T", "p1_stomata_transpiration", "H2O", "stomata"), ("H2O", "p1_stomata_transpiration", "T", "stomata"),
    ("hv", "p2_heating", "T", "heating"), ("H2O", "p3_osmotic_consequences", "Salinity", "osmotic"),
    ("Salinity", "p3_osmotic_consequences", "H2O", "osmotic"), ("Salinity", "p4_ion_balances", "Nutrients", "ion_balance"),
    ("Nutrients", "p4_ion_balances", "Salinity", "ion_balance"), ("hv", "p5_sun_shade", "Nutrients", "sun_shade"),
    ("H2O", "p6_stomata_light", "hv", "stomata"), ("hv", "p6_stomata_light", "H2O", "stomata"),
    ("H2O", "p7_stomata_co2", "CO2", "stomata"), ("CO2", "p7_stomata_co2", "H2O", "stomata"),
    ("hv", "p8_photorespiration", "CO2", "photoresp"), ("Salinity", "p9_metabolic_avoidance", "CO2", "metabolic_as"),
    ("H2O", "p10_xylem_flow", "Nutrients", "xylem"), ("Nutrients", "p10_xylem_flow", "H2O", "xylem"),
    ("CO2", "p11_cam_assimilation", "Nutrients", "cam_slope"), ("Nutrients", "p11_cn_signaling", "CO2", "cn_ratio"),
    ("T", "p12_membrane_metabolism", "CO2", "membrane"), ("CO2", "p12_membrane_metabolism", "T", "membrane"),
]

_safe_pow = lambda b, e: torch.pow(b, e) if isinstance(b, torch.Tensor) or isinstance(e, torch.Tensor) else b ** e

def build_torch_physics_equations():
    names = (
        "gs_max Cgs Wa Wi Ci Ca R Tl T0 Haj Hdj SvQ b_heat d_heat phi lam k2 Na h c_light "
        "Jmax0 Gamma0 Gamma1 Gamma2 Cx c2 c3 c4 w x y t dt f_y df_dy T_t F T_K p_H p_malate p_K "
        "H_o H_i malate_o malate_i Cl_o Cl_i V_cyto dC6_dt v_rubisco_C v_PPDK v_PEPC Rd "
        "psi_x psi_50 a_ws gp_max q_sr g_pw Z_r K_s s b_soil d_soil RAI_w c_as qw_max m_xylem "
        "alpha c6 c7 wr Lt Lk C_ext dCext_dt L_t dL_dt A_area v_SnRK1 vmax_SnRK1 Esps_act "
        "k_partial T6P Ki_T6P Km_ESPS SPS_act G1P Ki_G1P k_sps vmax_sps Esps_total F6P Km_F6P "
        "UDPG Km_UDPG gr Pmax S_t LB_t RB_t LA_t"
    )
    v = dict(zip(names.split(), sp.symbols(names, real=True)))
    w, wr, Lt, Lk = v["w"], v["wr"], v["Lt"], v["Lk"]

    gs = v["gs_max"] * v["Cgs"] * sp.sqrt(1 - (v["Ci"] / v["Ca"])**2) * (v["Wa"] / v["Wi"])
    tf = sp.exp((v["Haj"] / (v["R"] * v["T0"])) * (1 - v["T0"] / v["Tl"])) / (1 + sp.exp((v["SvQ"] * v["Tl"] - v["Hdj"]) / (v["R"] * v["Tl"])))
    g_star = v["Gamma0"] * (1 + v["Gamma1"] * (v["Tl"] - v["T0"]) + v["Gamma2"] * (v["Tl"] - v["T0"])**2)
    j_phi = (v["phi"] * v["lam"] * v["k2"]) / (2 * v["Na"] * v["h"] * v["c_light"])
    ws_sum = v["gp_max"] * (v["psi_x"] / v["psi_50"])**(-1) + (1 + (v["psi_x"] / v["psi_50"])**v["a_ws"])**(-1) + v["gp_max"] * ((1 / v["psi_50"])**v["a_ws"])**(-1) + v["gp_max"] * ((-v["psi_x"] / (v["psi_50"]**2))**v["a_ws"])**(-1)
    root = (((v["q_sr"] * sp.pi * v["g_pw"] * v["Z_r"]) / (v["K_s"] * v["s"]**(2 * v["b_soil"] + 3)))**2 / (v["s"]**(-v["d_soil"]))) / v["RAI_w"]
    cam = ((-1 + v["alpha"] * w) / sp.exp(v["alpha"] * w) - v["c6"] * v["L_t"] + (v["c7"] * Lk / (Lt + Lk)) * ((-wr - wr**2) / ((w + wr)**2))
           + (-v["c7"] * Lk * w - v["c7"] * Lk * w**2) / ((Lt + Lk) * (w + wr)**2) + (v["dCext_dt"] / sp.exp(v["alpha"] * w)) - v["c6"] * w * v["dL_dt"]
           + v["c7"] * (wr / (w + wr)) - (v["c7"] * Lk / (Lt**2)) * (wr / (w + wr)))
    a_cn = v["v_SnRK1"] - v["vmax_SnRK1"] * v["Esps_act"] * (1 + v["k_partial"] * (v["T6P"] / v["Ki_T6P"]))
    b_cn = v["Km_ESPS"] + v["SPS_act"] * (1 + v["T6P"] / v["Ki_T6P"] + v["G1P"] / v["Ki_G1P"])
    l_cn, p_cn, q_cn = v["vmax_sps"] / v["Esps_total"], v["F6P"] / (v["Km_F6P"] + v["F6P"]), v["UDPG"] / (v["Km_UDPG"] + v["UDPG"])
    c_fix = ((b_cn - a_cn) / b_cn**2) + v["k_sps"] + v["Esps_act"] + ((v["Esps_total"] - v["vmax_sps"]) / v["Esps_total"]**2) + (p_cn * q_cn + l_cn * (q_cn + p_cn))
    n_fix = sp.exp(v["gr"] * (sp.log(v["Pmax"] / v["S_t"]) + v["LB_t"] / v["RB_t"])) / (v["LA_t"] * v["Pmax"])
    ion = (v["R"] * v["T_K"] / v["F"]) * sp.log((v["p_H"] * v["H_o"] + v["p_malate"] * v["malate_o"] + v["p_K"] * v["Cl_o"]) / (v["H_i"] * v["malate_i"] * v["Cl_i"]))

    eqs = {
        "stomata": sum(sp.diff(gs, v[k]) for k in ("gs_max", "Cgs", "Wa", "Wi", "Ca", "Ci")),
        "heating": (-(v["b_heat"] * v["d_heat"]) / sp.sqrt((v["b_heat"] * v["d_heat"])**2)) * (-tf),
        "osmotic": -(v["c3"] * w) / (v["x"]**2) - 2 * v["c2"] * v["df_dy"] * v["T_t"] * v["y"] - v["f_y"] * v["T_t"] - v["c4"],
        "ion_balance": ion,
        "sun_shade": (sp.Min(v["Jmax0"] * tf, j_phi) / 4) * ((v["Cx"] - g_star) / (v["Cx"] + 2 * g_star)),
        "photoresp": -12 * (v["V_cyto"] * v["dC6_dt"] - sp.Rational(1, 6) * v["v_rubisco_C"] - sp.Rational(1, 2) * v["v_PPDK"] + sp.Rational(29, 54) * v["v_PEPC"] + sp.Rational(1, 6) * v["Rd"]),
        "metabolic_as": ws_sum * root + (w - 1) / v["c_as"],
        "xylem": v["qw_max"] * (w**v["m_xylem"]),
        "cam_slope": cam,
        "cn_ratio": c_fix / n_fix,
        "membrane": sp.integrate(cam * ion, (v["A_area"], 1, 12)) * v["dt"],
    }
    mods = [{"Min": torch.minimum, "pow": _safe_pow}, "torch"]
    return {k: (sp.lambdify(s := sorted(e.free_symbols, key=lambda x: x.name), e, modules=mods), [x.name for x in s]) for k, e in eqs.items()}

def create_physiological_heterodata(node_feats, state_tensors, compiled_eqs, edge_idx_dict=None, save_path="physiological_heterodata.pt"):
    data = HeteroData()
    for ntype, x in node_feats.items(): data[ntype].x = x
    data.state_vars = {k: v.detach().clone() for k, v in state_tensors.items()}
    for src, rel, dst, eq_key in PATHWAY_SCHEMA:
        etype, (fn, args) = (src, rel, dst), compiled_eqs[eq_key]
        w = fn(*(state_tensors[n] for n in args)).reshape(-1, 1)
        data[etype].edge_attr = w
        data[etype].edge_index = edge_idx_dict[etype] if edge_idx_dict and etype in edge_idx_dict else torch.zeros((2, w.size(0)), dtype=torch.long)
    torch.save(data, save_path)
    print(f"Saved HeteroData to {save_path}:\n{data}")
    return data

class EdgeModulatedConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin_src, self.lin_dst = Linear(in_ch, out_ch), Linear(in_ch, out_ch)
        self.edge_gate = nn.Sequential(Linear(1, out_ch), nn.Tanh())
    def forward(self, x, edge_index, edge_attr):
        (x_src, x_dst), (row, col) = x, edge_index
        return torch.zeros_like(self.lin_dst(x_dst)).index_add_(0, col, self.lin_src(x_src[row]) * self.edge_gate(edge_attr)) + self.lin_dst(x_dst)

class DynamicPhysicsHeteroGNN(nn.Module):
    def __init__(self, in_dims, hidden_dim, out_dim, compiled_eqs, learnable_names=None, init_params=None):
        super().__init__()
        self.compiled_eqs = compiled_eqs
        self.learnable_params = nn.ParameterDict({n: nn.Parameter(init_params[n].clone()) for n in (learnable_names or [])})
        self.encoders = nn.ModuleDict({k: Linear(v, hidden_dim) for k, v in in_dims.items()})
        self.conv1 = HeteroConv({(s, r, d): EdgeModulatedConv(hidden_dim, hidden_dim) for s, r, d, _ in PATHWAY_SCHEMA}, aggr="sum")
        self.decoders = nn.ModuleDict({k: Linear(hidden_dim, out_dim) for k in in_dims})
    def compute_dynamic_edges(self, state_vars):
        live = {**state_vars, **dict(self.learnable_params)}
        return {(s, r, d): self.compiled_eqs[k][0](*(live[n] for n in self.compiled_eqs[k][1])).reshape(-1, 1) for s, r, d, k in PATHWAY_SCHEMA}
    def forward(self, data):
        x = {k: torch.relu(self.encoders[k](v)) for k, v in data.x_dict.items()}
        x = {k: torch.relu(v) for k, v in self.conv1(x, data.edge_index_dict, self.compute_dynamic_edges(data.state_vars)).items()}
        return {k: self.decoders[k](v) for k, v in x.items()}

# Build graph, save, and verify live edge weight updates during training
eqs = build_torch_physics_equations()
state_tensors = {n: torch.rand(1) + 0.5 for n in sorted({n for _, args in eqs.values() for n in args})}
state_tensors["Ca"] = state_tensors["Ci"] + 1.0
hetero_data = create_physiological_heterodata({n: torch.randn(1, 4) for n in NODE_TYPES}, state_tensors, eqs)

model = DynamicPhysicsHeteroGNN({n: 4 for n in NODE_TYPES}, 16, 1, eqs, ["gs_max", "qw_max"], state_tensors)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
before_w = model.compute_dynamic_edges(hetero_data.state_vars)[("T", "p1_stomata_transpiration", "H2O")].item()
sum(p.pow(2).mean() for p in model(hetero_data).values()).backward()
opt.step()
after_w = model.compute_dynamic_edges(hetero_data.state_vars)[("T", "p1_stomata_transpiration", "H2O")].item()
print(f"\nDynamic Edge Weight (p1_stomata_transpiration): {before_w:.6f} -> {after_w:.6f}")
