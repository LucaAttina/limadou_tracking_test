import argparse
import re
import os
import sys
import numpy as np
import awkward as ak
import ROOT
import yaml
import math
import glob
import uproot
from efficiency_utils_new import (
    is_track_reconstructed,
    read_data_from_file,
    process_efficiency_event
)
import json

def read_data(file):
    """
    Read basic data structures.
    MCtruth and SelectedEvents have the same entry index = event index.
    """
    
    # Read trees
    trees = {
        "rec_m1": file["SelectedEvents"],
        "rec_m2": file["SelectedEvents_m2"],
        "mult": file["EventSummary"]
    }
    
    # Check all trees exist
    for name, tree in trees.items():
        if not tree:
            raise RuntimeError(f"TTree '{name}' not found in file")
        #print(f"[DEBUG] Tree '{name}' loaded successfully")
        
    mult_data = {
        "event_idx": trees["mult"]["event"].array(),
        "m1": trees["mult"]["mult_m1"].array(),
        "m2": trees["mult"]["mult_m2"].array()
    }
    
    n_events = len(mult_data["event_idx"])
    
    # ========================================
    # RECONSTRUCTED DATA - M1
    # ========================================
    
    #print(f"\n[DEBUG] M1 (SelectedEvents) tree:")
    
    #rec_m1_theta = trees["rec_m1"]["theta"].array()
    #rec_m1_phi = trees["rec_m1"]["phi"].array()
    rec_m1_event_idx = trees["rec_m1"]["event_idx"].array()
    
    # Cluster M1
    #rec_m1_cls_x = trees["rec_m1"]["cl_x"].array()
    #rec_m1_cls_y = trees["rec_m1"]["cl_y"].array()
    rec_m1_cls_z = trees["rec_m1"]["cl_z"].array()
    #d_sum_m1 = trees["rec_m1"]["d_sum"].array()
    res_x_m1 = trees["rec_m1"]["res_x"].array() 
    res_y_m1 = trees["rec_m1"]["res_y"].array() 
    
    
    # ========================================
    # RECONSTRUCTED DATA - M2
    # ========================================
        
    #rec_m2_theta = trees["rec_m2"]["theta_m2"].array()
    #rec_m2_phi = trees["rec_m2"]["phi_m2"].array()
    rec_m2_event_idx = trees["rec_m2"]["event_idx"].array()

    
    # Cluster M2
    #rec_m2_cls_x = trees["rec_m2"]["cl_x"].array()
    #rec_m2_cls_y = trees["rec_m2"]["cl_y"].array()
    rec_m2_cls_z = trees["rec_m2"]["cl_z"].array()
    #d_sum_m2 = trees["rec_m2"]["d_sum"].array()
    res_x_m2 = trees["rec_m2"]["res_x"].array()
    res_y_m2 = trees["rec_m2"]["res_y"].array()

    # Calcola n_tracks totali
    #n_tracks_m1 = len(rec_m1_theta)
    #n_tracks_m2 = len(rec_m2_theta)
    
    # ========================================
    # RETURN
    # ========================================
    
    return {
        "rec_data": {
            "m1": {
                #"theta": rec_m1_theta,
                #"phi": rec_m1_phi,
                "event_idx": rec_m1_event_idx,
                #"d_sum": d_sum_m1,
                "cls": {#"x": rec_m1_cls_x, "y": rec_m1_cls_y, 
                    "z": rec_m1_cls_z},
                "res": {"x": res_x_m1, "y": res_y_m1}
            },
            "m2": {
                #"theta": rec_m2_theta,
                #"phi": rec_m2_phi,
                "event_idx": rec_m2_event_idx,
                #"d_sum": d_sum_m2,
                "cls": {#"x": rec_m2_cls_x, "y": rec_m2_cls_y, 
                    "z": rec_m2_cls_z},
                "res": {"x": res_x_m2, "y": res_y_m2}
            }
        },
        "mult_data": mult_data,
        "n_events": n_events,
        #"n_tracks": {             
        #    "m1": n_tracks_m1,
        #    "m2": n_tracks_m2
        #}
    }


def res_plots(input_dir, output_dir, config, part="e", mc="False"):
    """
    Main function to calculate and plot residuals distributions.
    """
    os.makedirs(output_dir, exist_ok=True)

    root_files = sorted(glob.glob(os.path.join(input_dir, "*.root")))

    #r_val = 0.25
    #r_val = f"{r_val:.2f}".replace(".", "p")

    if mc:
        out_file = ROOT.TFile(os.path.join(output_dir, f"MC_{part}_residuals.root"), "RECREATE")
    else:
        out_file = ROOT.TFile(os.path.join(output_dir, f"{part}_residuals.root"), "RECREATE")

    #coord_map = {"x": 0, "y": 1}
    method_map = {"m1": 1, "m2": 2}
    layer_map = {"l1": 1, "l2": 2, "l3": 3}

    for file in root_files:
        
        print(f"Processing {file}")

        f = uproot.open(file)
        file_data = read_data(f)

        match = re.search(r'_(\d+\.?\d*)MeV_', file)
        if not match:
            continue

        energy = float(match.group(1))

        if part == "e":
            mass = 0.5109989
        elif part == "p":
            mass = 938.272
        elif part == "c":
            mass = 11174.9
            energy *= 12

        Etot = energy + mass
        p2 = Etot**2 - mass**2
        betap = p2 / Etot

        base = os.path.basename(file)

        prefix = base.split("MAIN")[0].rstrip("_")

        out_file.cd()

        tree_name = f"{part}_{energy:.0f}_MeV_res"

        tree = ROOT.TTree(f"{tree_name}", "residuals tree")

        energy_tree = np.zeros(1, dtype=np.float32)
        betap_tree = np.zeros(1, dtype=np.float32)
        method_tree = np.zeros(1, dtype=np.int32)
        layer_tree = np.zeros(1, dtype=np.int32)
        #coord_tree = np.zeros(1, dtype=np.int32)
        res_x_tree = np.zeros(1, dtype=np.float32)
        res_y_tree = np.zeros(1, dtype=np.float32)

        tree.Branch("energy", energy_tree, "energy/F")
        tree.Branch("betap", betap_tree, "betap/F")
        tree.Branch("method", method_tree, "method/I")
        tree.Branch("layer", layer_tree, "layer/I")
        #tree.Branch("coord", coord_tree, "coord/I")
        tree.Branch("res_x", res_x_tree, "res_x/F")
        tree.Branch("res_y", res_y_tree, "res_y/F")

        n_events = file_data["n_events"]
        
        for ev_idx in range(n_events):
            for method in ["m1","m2"]:

                mult = file_data['mult_data'][method][ev_idx]
                reco_idx = np.where(ev_idx == file_data["rec_data"][method]["event_idx"])[0]
                #
                #if reco_idx.tolist():
                #    mc_reconstructed, reco_to_gen = is_track_reconstructed(
                #        ev_idx, method, file_data["rec_data"][method]["cls"]["x"][reco_idx], 
                #        file_data["rec_data"][method]["cls"]["y"][reco_idx], file_data["rec_data"][method]["cls"]["z"][reco_idx], 
                #        file_data["gen_data"]["cls"]["x"][ev_idx], file_data["gen_data"]["cls"]["y"][ev_idx], 
                #        file_data["gen_data"]["cls"]["layer"][ev_idx], file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx], 
                #        n_tracks_per_event, reco_idx
                #    )   
                #    matched_reco_id = set(reco_to_gen.values())
                    #print(f"Event {ev_idx}, method {method}, matched reco IDs: {matched_reco_id}, mult: {mult}")

                if mult != 1:
                    continue
                #print(f"Event {ev_idx}, method {method}, mult: {mult} - processing for residuals")
                mult_id = int(0)
                n_cls = len([c for c in file_data["rec_data"][method]["cls"]["z"][reco_idx][mult_id] if c != -999])

                if n_cls == 3:
                    #print(f"Event {ev_idx}, method {method}, reco_id {mult_id} has 3 clusters - processing for residuals")
                    res_x = np.array(
                        file_data["rec_data"][method]["res"]["x"][reco_idx][mult_id]
                    ).flatten()

                    res_y = np.array(
                        file_data["rec_data"][method]["res"]["y"][reco_idx][mult_id]
                    ).flatten()

                    # Salva tutte le residuals per questa particella
                    for i in range(len(res_x)):
                        z = file_data["rec_data"][method]["cls"]["z"][reco_idx][mult_id][i]
                        if abs(z - 17.825) < 1e-3:
                            lay = "l1"
                        elif abs(z - 26.325) < 1e-3:
                            lay = "l2"
                        elif abs(z - 34.825) < 1e-3:
                            lay = "l3"

                        energy_tree[0] = energy
                        betap_tree[0] = betap
                        method_tree[0] = method_map[method]
                        layer_tree[0] = layer_map[lay]
                        res_x_tree[0] = res_x[i]
                        res_y_tree[0] = res_y[i]

                        tree.Fill() 
                        #print(f"Filled tree for Event {ev_idx}, method {method}, reco_id {mult_id}, layer {lay}, res_x {res_x[i]}, res_y {res_y[i]}")

                #else:
                #    print(f"Event {ev_idx}, method {method}, reco_id {mult_id} has {n_cls} clusters - skipping for residuals")

                '''         
                for mult_id in range(mult):
                    #if mult_id in matched_reco_id:
                        #print(f"Event {ev_idx}, method {method}, reco_id {reco_id} is matched")
                        n_cls = len([c for c in file_data["rec_data"][method]["cls"]["z"][reco_idx][mult_id] if c != -999])
                        if n_cls != 3:
                            #print(f"Event {ev_idx}, method {method}, reco_id {reco_id} has {n_cls} clusters - skipping for residuals")
                            continue  # Skip tracks that don't have 3 clusters

                        res_x = np.array(
                            file_data["rec_data"][method]["res"]["x"][reco_idx][mult_id]
                        ).flatten()

                        res_y = np.array(
                            file_data["rec_data"][method]["res"]["y"][reco_idx][mult_id]
                        ).flatten()

                        #res_x_flat = np.array(res_x).flatten() if hasattr(res_x, 'flatten') else res_x
                        #res_y_flat = np.array(res_y).flatten() if hasattr(res_y, 'flatten') else res_y

                        for i in range(len(res_x)):

                            z = file_data["rec_data"][method]["cls"]["z"][reco_idx][mult_id][i]
                            if abs(z - 17.825) < 1e-3:
                                lay = "l1"
                            elif abs(z - 26.325) < 1e-3:
                                lay = "l2"
                            elif abs(z - 34.825) < 1e-3:
                                lay = "l3"
                                
                            energy_tree[0] = energy
                            betap_tree[0] = betap
                            method_tree[0] = method_map[method]
                            layer_tree[0] = layer_map[lay]
                            #coord_tree[0] = coord_map["x"]
                            res_x_tree[0] = res_x[i]
                            res_y_tree[0] = res_y[i]

                            tree.Fill()
                            '''
        
        # Scrivi il tree subito dopo averlo riempito
        tree.Write()
                            
    out_file.cd()
    out_file.Close()
    print(f"residuals.root file created in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get residual distributions.")
    parser.add_argument("--input-dir", required=True, help="Use this directory containing .root files.")
    parser.add_argument("--output-dir", default=None, help="Define output directory.")
    parser.add_argument("--particle", required=True, help="Select particle configuration.")
    parser.add_argument("--mc", action="store_true", help="Define name.")
    args = parser.parse_args()
    
    if args.particle not in ["e", "p", "c"]:
        print(f"Error: Invalid particle type '{args.particle}'. Must be one of: e, p, c.")
        sys.exit(1)

    config_file = "config/config_res_dist.yaml"
    with open(config_file, "r") as f_cfg:
        config = yaml.safe_load(f_cfg)
    
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = config["output_dir"]
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    res_plots(args.input_dir, output_dir, config, args.particle, args.mc)