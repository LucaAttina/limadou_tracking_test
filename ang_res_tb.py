import argparse
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
    initialize_counters,
    #process_delta_event,
    read_data_from_file,
    compute_gen_mask, 
    is_track_reconstructed,
    process_efficiency_event
)
import re   
import json
import matplotlib.pyplot as plt

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
    
    rec_m1_theta = trees["rec_m1"]["theta"].array()
    #rec_m1_phi = trees["rec_m1"]["phi"].array()
    rec_m1_event_idx = trees["rec_m1"]["event_idx"].array()
    
    # Cluster M1
    #rec_m1_cls_x = trees["rec_m1"]["cl_x"].array()
    #rec_m1_cls_y = trees["rec_m1"]["cl_y"].array()
    #rec_m1_cls_z = trees["rec_m1"]["cl_z"].array()
    ##d_sum_m1 = trees["rec_m1"]["d_sum"].array()
    #res_x_m1 = trees["rec_m1"]["res_x"].array() 
    #res_y_m1 = trees["rec_m1"]["res_y"].array() 
    
    
    # ========================================
    # RECONSTRUCTED DATA - M2
    # ========================================
        
    rec_m2_theta = trees["rec_m2"]["theta_m2"].array()
    #rec_m2_phi = trees["rec_m2"]["phi_m2"].array()
    rec_m2_event_idx = trees["rec_m2"]["event_idx"].array()

    
    ## Cluster M2
    #rec_m2_cls_x = trees["rec_m2"]["cl_x"].array()
    #rec_m2_cls_y = trees["rec_m2"]["cl_y"].array()
    #rec_m2_cls_z = trees["rec_m2"]["cl_z"].array()
    ##d_sum_m2 = trees["rec_m2"]["d_sum"].array()
    #res_x_m2 = trees["rec_m2"]["res_x"].array()
    #res_y_m2 = trees["rec_m2"]["res_y"].array()

    # Calcola n_tracks totali
    #n_tracks_m1 = len(rec_m1_theta)
    #n_tracks_m2 = len(rec_m2_theta)
    
    # ========================================
    # RETURN
    # ========================================
    
    return {
        "rec_data": {
            "m1": {
                "theta": rec_m1_theta,
                #"phi": rec_m1_phi,
                "event_idx": rec_m1_event_idx,
                #"d_sum": d_sum_m1,
                #"cls": {#"x": rec_m1_cls_x, "y": rec_m1_cls_y, 
                #    "z": rec_m1_cls_z},
                #"res": {"x": res_x_m1, "y": res_y_m1}
            },
            "m2": {
                "theta": rec_m2_theta,
                #"phi": rec_m2_phi,
                "event_idx": rec_m2_event_idx,
                #"d_sum": d_sum_m2,
                #"cls": {#"x": rec_m2_cls_x, "y": rec_m2_cls_y, 
                #    "z": rec_m2_cls_z},
                #"res": {"x": res_x_m2, "y": res_y_m2}
            }
        },
        "mult_data": mult_data,
        "n_events": n_events,
        #"n_tracks": {             
        #    "m1": n_tracks_m1,
        #    "m2": n_tracks_m2
        #}
    }


def delta_data(input_dir, output_dir, config, particle="e", mc="False"):
    """
    Main function to calculate and plot delta distributions.
    """
    os.makedirs(output_dir, exist_ok=True)

    if particle == "e":
        part = "electron"
        mass = 0.5109989
    elif particle == "p":
        part = "proton"
        mass = 938.272
    elif particle == "c":
        part = "carbon"
        mass = 11174.9

    # ROOT output file
    if mc:
        out_file = ROOT.TFile(
            os.path.join(output_dir, f"MC_{particle}_theta.root"),
            "RECREATE"
        )
    else:
        out_file = ROOT.TFile(
            os.path.join(output_dir, f"{particle}_theta.root"),
            "RECREATE"
        )


    method_map = {
        "m1": 1,
        "m2": 2
    }

    '''
    tree = ROOT.TTree(
        "theta_signed",
        "Signed theta distribution"
    )


    energy_tree = np.zeros(1, dtype=np.float32)
    betap_tree = np.zeros(1, dtype=np.float32)
    method_tree = np.zeros(1, dtype=np.int32)
    theta_tree = np.zeros(1, dtype=np.float32)


    tree.Branch(
        "energy",
        energy_tree,
        "energy/F"
    )

    tree.Branch(
        "betap",
        betap_tree,
        "betap/F"
    )

    tree.Branch(
        "method",
        method_tree,
        "method/I"
    )

    tree.Branch(
        "theta_signed",
        theta_tree,
        "theta_signed/F"
    )
    '''

    root_files = sorted(glob.glob(os.path.join(input_dir, "*.root")))

    results = {
        "betap": [],
        "m1": {"mean": [], "std_dev": []},
        "m2": {"mean": [], "std_dev": []}
    }


    # ============================================================
    # DEBUG: Creo una directory per i plot di debug
    # ============================================================
    debug_dir = os.path.join(output_dir, "debug_plots")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"📁 Debug plots will be saved to: {debug_dir}")

    for file in root_files:

        print(f"Processing {file}")

        f = uproot.open(file)
        file_data = read_data(f)
        #if not mc:
        #    file_data = read_data(f)
        #else:
        #    file_data = read_data_mc(f)

        match = re.search(r'_(\d+\.?\d*)MeV_', file)
        if not match:
            continue

        energy = float(match.group(1))            

        Etot = energy + mass
        p2 = Etot**2 - mass**2
        betap = p2 / Etot   

        results["betap"].append(betap)

        tree_name = f"{part}_{energy:.0f}_MeV_theta"

        tree = ROOT.TTree(
            tree_name,
            "theta distribution"
        )

        energy_tree = np.zeros(1, dtype=np.float32)
        betap_tree = np.zeros(1, dtype=np.float32)
        method_tree = np.zeros(1, dtype=np.int32)
        theta_tree = np.zeros(1, dtype=np.float32)


        tree.Branch(
            "energy",
            energy_tree,
            "energy/F"
        )

        tree.Branch(
            "betap",
            betap_tree,
            "betap/F"
        )

        tree.Branch(
            "method",
            method_tree,
            "method/I"
        )

        tree.Branch(
            "theta",
            theta_tree,
            "theta/F"
        )
        
        # Initialize counters for delta analysis
        #counters = initialize_counters(["m1", "m2"], "delta")

        # Print track counts
        n_events = len(file_data["mult_data"]["event_idx"])
        print(f"Energy: {energy} MeV")
        print(f"beta*p: {betap:.4f}")
        print(f"Number of events: {n_events}")
        
        # ============================================================
        # DEBUG 1: Stampa statistiche di theta per questo file
        # ============================================================
        print(f"\n📊 THETA STATISTICS:")
        for method in ["m1", "m2"]:
            theta_vals = file_data["rec_data"][method]["theta"]
            if len(theta_vals) > 0:
                print(f"  {method.upper()}:")
                print(f"    N_tracks: {len(theta_vals)}")
                print(f"    Min: {np.min(theta_vals):.4f}°")
                print(f"    Max: {np.max(theta_vals):.4f}°")
                print(f"    Mean: {np.mean(theta_vals):.4f}°")
                print(f"    Std: {np.std(theta_vals):.4f}°")
                print(f"    Median: {np.median(theta_vals):.4f}°")
                # Prime 10 tracce
                print(f"    First 10 theta values: {theta_vals[:10]}")
        
        # ============================================================
        # DEBUG 2: Controlla molteplicità
        # ============================================================
        print(f"\n📊 MULTIPLICITY STATISTICS:")
        for method in ["m1", "m2"]:
            mult_vals = file_data["mult_data"][method]
            unique, counts = np.unique(mult_vals, return_counts=True)
            print(f"  {method.upper()}:")
            for u, c in zip(unique, counts):
                print(f"    mult={u}: {c} events ({c/n_events*100:.1f}%)")
        
        # Data containers for delta values
        delta_theta = {
            "m1": [],
            "m2": []
        }
    
        # Process all events
        for ev_idx in range(n_events):

            for method in ["m1","m2"]:
                mult = file_data['mult_data'][method][ev_idx]
                if mult != 1:
                    continue
                reco_idx = np.where(ev_idx == file_data["rec_data"][method]["event_idx"])[0]
                #print(f"\nEvent {ev_idx} - Method {method.upper()} - Multiplicity: {mult} - Reco tracks: {reco_idx}")
                if reco_idx.tolist():
                    #if mult == 1:
                    #if not mc:
                    theta_values = np.array(
                        file_data["rec_data"][method]["theta"][reco_idx]
                    )


                    for th in theta_values:

                        energy_tree[0] = energy
                        betap_tree[0] = betap
                        method_tree[0] = method_map[method]
                        theta_tree[0] = th

                        tree.Fill()

                        delta_theta[method].append(th)
        tree.Write()

                    #else:
                    #    delta_theta[method].extend(file_data["rec_data"][method]["theta"][reco_idx] - file_data["gen_data"]["theta"][ev_idx])  
                    #    #delta_data[method]["delta_phi"].extend(file_data["rec_data"][method]["phi"][reco_idx])    
                    #    print(f" delta_theta: {file_data['rec_data'][method]['theta'][reco_idx]} - {file_data["gen_data"]["theta"][ev_idx]}")  

        # ============================================================
        # DEBUG 3: Stampa statistiche di delta_theta
        # ============================================================
        print(f"\n📊 DELTA THETA STATISTICS (mult=1 only):")
        for method in ["m1", "m2"]:
            if len(delta_theta[method]) > 0:
                print(f"  {method.upper()}:")
                print(f"    N_events: {len(delta_theta[method])}")
                print(f"    Mean: {np.mean(delta_theta[method]):.4f}°")
                print(f"    Std: {np.std(delta_theta[method]):.4f}°")
                print(f"    Min: {np.min(delta_theta[method]):.4f}°")
                print(f"    Max: {np.max(delta_theta[method]):.4f}°")
            else:
                print(f"  {method.upper()}: NO EVENTS WITH MULT=1!")

        # ============================================================
        # DEBUG 4: Plot istogrammi
        # ============================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot theta distributions
        for i, method in enumerate(["m1", "m2"]):
            theta_vals = file_data["rec_data"][method]["theta"]
            ax = axes[0, i]
            ax.hist(theta_vals, bins=70, alpha=0.7, color='blue' if i==0 else 'red', edgecolor='black')
            ax.set_xlabel('Theta (degrees)')
            ax.set_ylabel('Entries')
            ax.set_title(f'{method.upper()} - All tracks\nMean = {np.mean(theta_vals):.2f}°, Std = {np.std(theta_vals):.2f}°')
            ax.axvline(0, color='green', linestyle='--', linewidth=2, label='θ=0 (beam)')
            ax.axvline(np.mean(theta_vals), color='red', linestyle='-', linewidth=2, label=f'Mean = {np.mean(theta_vals):.2f}°')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot delta theta distributions (mult=1)
        for i, method in enumerate(["m1", "m2"]):
            ax = axes[1, i]
            if len(delta_theta[method]) > 0:
                ax.hist(delta_theta[method], bins=70, alpha=0.7, color='blue' if i==0 else 'red', edgecolor='black')
                ax.set_xlabel('Delta Theta (degrees)')
                ax.set_ylabel('Entries')
                ax.set_title(f'{method.upper()} - Δθ (mult=1)\nMean = {np.mean(delta_theta[method]):.2f}°, Std = {np.std(delta_theta[method]):.2f}°')
                ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Δθ=0')
                ax.axvline(np.mean(delta_theta[method]), color='red', linestyle='-', linewidth=2, label=f'Mean = {np.mean(delta_theta[method]):.2f}°')
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'NO EVENTS WITH MULT=1', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{particle} - {energy} MeV  (beta*p = {betap:.4f})', fontsize=14)
        plt.tight_layout()
        if not mc:
            debug_plot = os.path.join(debug_dir, f'debug_{particle}_{int(energy)}MeV.png')
        else:
            debug_plot = os.path.join(debug_dir, f'debug_MC_{particle}_{int(energy)}MeV.png')

        plt.savefig(debug_plot, dpi=150)
        plt.close()
        print(f"  📊 Debug plot saved: {debug_plot}")
        
        # Convert to numpy arrays
        for method in ["m1","m2"]:
            results[method]["mean"].append(float(np.mean(delta_theta[method])))
            results[method]["std_dev"].append(float(np.std(delta_theta[method])))

    # ============================================================
    # DEBUG 6: Stampa tabella riassuntiva
    # ============================================================
    print("\n" + "="*80)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Energy (MeV)':>12} {'beta*p':>12} {'M1 mean (°)':>14} {'M1 std (°)':>14} {'M2 mean (°)':>14} {'M2 std (°)':>14}")
    print("-"*80)
    for i, (betap, m1_mean, m1_std, m2_mean, m2_std) in enumerate(zip(
        results["betap"], 
        results["m1"]["mean"], 
        results["m1"]["std_dev"],
        results["m2"]["mean"],
        results["m2"]["std_dev"]
    )):
        # Estrai energia dal nome del file (approssimativo)
        print(f"{i+1:>12} {betap:>12.4f} {m1_mean:>14.4f} {m1_std:>14.4f} {m2_mean:>14.4f} {m2_std:>14.4f}")
    print("="*80)

    #json_dir = "/home/lattina/limadou/data/testBeam/paper_plot_json/new_method"
    if mc:
        results_output = os.path.join(output_dir, f"MC_{particle}_dtheta_results.json")
    else:
        results_output = os.path.join(output_dir, f"{particle}_dtheta_results.json")

    out_file.cd()
    out_file.Close()
    
    #print(f"✅ ROOT theta file saved: {root_output}")

    #with open(results_output, 'w') as f:
    #    fin_data = {
    #        "particle": part,
    #        "results": results
    #    }     
#
    #    json.dump(fin_data, f, indent=4)
    #print(f"✅ Fit results saved to: {results_output}")    



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
    
    delta_data(args.input_dir, output_dir, config, args.particle, args.mc)