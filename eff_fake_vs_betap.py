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
    #print("\n" + "="*60)
    #print("READING DATA FROM FILE")
    #print("="*60)
    
    # Read trees
    trees = {
        "rec_m1": file["SelectedEvents"],
        "rec_m2": file["SelectedEvents_m2"],
        "gen": file["MCtruth"],
        "mult": file["EventSummary"]
    }
    
    # Check all trees exist
    for name, tree in trees.items():
        if not tree:
            raise RuntimeError(f"TTree '{name}' not found in file")
        #print(f"[DEBUG] Tree '{name}' loaded successfully")
    
    # Multiplicity data (from EventSummary)
    mult_data = {
        "event_idx": trees["mult"]["event"].array(),
        "m1": trees["mult"]["mult_m1"].array(),
        "m2": trees["mult"]["mult_m2"].array()
    }
    
    n_events = len(mult_data["event_idx"])
    n_tracks_mc = trees["gen"].num_entries
    #print(f"\n[DEBUG] EventSummary:")
    #print(f"  - n_events: {n_events}")
    #print(f"  - event_idx: {mult_data['event_idx'][:10]}... (first 10)")
    #print(f"  - mult_m1: {mult_data['m1'][:10]}... (first 10)")
    #print(f"  - mult_m2: {mult_data['m2'][:10]}... (first 10)")
    
    # ========================================
    # MC TRUTH DATA (per entry = evento)
    # ========================================
    
    #print(f"\n[DEBUG] MCtruth tree:")
    
    # Leggi i dati MC - ogni entry è un evento
    '''
    gen_theta = trees["gen"]["gen_theta"].array()
    gen_phi = trees["gen"]["gen_phi"].array()
    p_id = trees["gen"]["particle_id"].array()
    gen_energy = trees["gen"]["gen_energy"].array()
    mass = trees["gen"]["Mass"].array()
    '''
    
    #print(f"  - gen_theta type: {type(gen_theta)}")
    #print(f"  - gen_theta length: {len(gen_theta)}")
    #if len(gen_theta) > 0:
    #    print(f"  - gen_theta[0] type: {type(gen_theta[0])}")
    #    print(f"  - gen_theta[0] length: {len(gen_theta[0]) if hasattr(gen_theta[0], '__len__') else 'scalar'}")
    #    print(f"  - gen_theta[0]: {gen_theta[0]}")
    
    # Cluster MC
    '''
    gen_cls_x = trees["gen"]["x_abspos"].array()
    gen_cls_y = trees["gen"]["y_abspos"].array()
    gen_cls_layer = trees["gen"]["cl_layer"].array()
    gen_cls_to_trk = trees["gen"]["cls_to_trk_idx"].array()
    '''
    
    #print(f"  - gen_cls_x length: {len(gen_cls_x)}")
    #if len(gen_cls_x) > 0:
    #    print(f"  - gen_cls_x[0] type: {type(gen_cls_x[0])}")
    #    print(f"  - gen_cls_x[0] length: {len(gen_cls_x[0]) if hasattr(gen_cls_x[0], '__len__') else 'scalar'}")
    #    print(f"  - gen_cls_x[0][:5]: {gen_cls_x[0][:5] if hasattr(gen_cls_x[0], '__getitem__') else gen_cls_x[0]}")
    
    # Determina n_tracks_per_event
    '''
    n_tracks_per_event = 0
    if len(gen_theta) > 0:
        if hasattr(gen_theta[0], '__len__'):
            n_tracks_per_event = len(gen_theta[0])
        else:
            n_tracks_per_event = 1  # scalare = 1 traccia
    '''
    #print(f"  - n_tracks_per_event: {n_tracks_per_event}")
    
    # ========================================
    # RECONSTRUCTED DATA - M1
    # ========================================
    
    #print(f"\n[DEBUG] M1 (SelectedEvents) tree:")
    
    '''
    rec_m1_theta = trees["rec_m1"]["theta"].array()
    rec_m1_phi = trees["rec_m1"]["phi"].array()
    rec_m1_event_idx = trees["rec_m1"]["event_idx"].array()
    
    #print(f"  - rec_m1_theta type: {type(rec_m1_theta)}")
    #print(f"  - rec_m1_theta length: {len(rec_m1_theta)}")
    #if len(rec_m1_theta) > 0:
    #    print(f"  - rec_m1_theta[0] type: {type(rec_m1_theta[0])}")
    #    print(f"  - rec_m1_theta[0] length: {len(rec_m1_theta[0]) if hasattr(rec_m1_theta[0], '__len__') else 'scalar'}")
    #    print(f"  - rec_m1_theta[0]: {rec_m1_theta[0]}")
    #
    #print(f"  - rec_m1_event_idx length: {len(rec_m1_event_idx)}")
    #print(f"  - rec_m1_event_idx[:10]: {rec_m1_event_idx[:10]}")
    
    # Cluster M1
    rec_m1_cls_x = trees["rec_m1"]["cl_x"].array()
    rec_m1_cls_y = trees["rec_m1"]["cl_y"].array()
    rec_m1_cls_z = trees["rec_m1"]["cl_z"].array()
    d_sum_m1 = trees["rec_m1"]["d_sum"].array()
    res_x_m1 = trees["rec_m1"]["res_x"].array() 
    res_y_m1 = trees["rec_m1"]["res_y"].array() 
    
    #print(f"  - rec_m1_cls_x length: {len(rec_m1_cls_x)}")
    #if len(rec_m1_cls_x) > 0:
    #    print(f"  - rec_m1_cls_x[0] type: {type(rec_m1_cls_x[0])}")
    #    print(f"  - rec_m1_cls_x[0] length: {len(rec_m1_cls_x[0]) if hasattr(rec_m1_cls_x[0], '__len__') else 'scalar'}")
    
    # ========================================
    # RECONSTRUCTED DATA - M2
    # ========================================
    
    #print(f"\n[DEBUG] M2 (SelectedEvents_m2) tree:")
    
    rec_m2_theta = trees["rec_m2"]["theta_m2"].array()
    rec_m2_phi = trees["rec_m2"]["phi_m2"].array()
    rec_m2_event_idx = trees["rec_m2"]["event_idx"].array()
    
    #print(f"  - rec_m2_theta type: {type(rec_m2_theta)}")
    #print(f"  - rec_m2_theta length: {len(rec_m2_theta)}")
    #if len(rec_m2_theta) > 0:
    #    print(f"  - rec_m2_theta[0] type: {type(rec_m2_theta[0])}")
    #    print(f"  - rec_m2_theta[0] length: {len(rec_m2_theta[0]) if hasattr(rec_m2_theta[0], '__len__') else 'scalar'}")
    #    print(f"  - rec_m2_theta[0]: {rec_m2_theta[0]}")
    
    #print(f"  - rec_m2_event_idx length: {len(rec_m2_event_idx)}")
    #print(f"  - rec_m2_event_idx[:10]: {rec_m2_event_idx[:10]}")
    
    # Cluster M2
    rec_m2_cls_x = trees["rec_m2"]["cl_x"].array()
    rec_m2_cls_y = trees["rec_m2"]["cl_y"].array()
    rec_m2_cls_z = trees["rec_m2"]["cl_z"].array()
    d_sum_m2 = trees["rec_m2"]["d_sum"].array()
    res_x_m2 = trees["rec_m2"]["res_x"].array()
    res_y_m2 = trees["rec_m2"]["res_y"].array()

    
    #print(f"  - rec_m2_cls_x length: {len(rec_m2_cls_x)}")
    #if len(rec_m2_cls_x) > 0:
    #    print(f"  - rec_m2_cls_x[0] type: {type(rec_m2_cls_x[0])}")
    #    print(f"  - rec_m2_cls_x[0] length: {len(rec_m2_cls_x[0]) if hasattr(rec_m2_cls_x[0], '__len__') else 'scalar'}")
    
    # ========================================
    # VERIFICA CONSISTENZA
    # ========================================
    
    #print(f"\n[DEBUG] Consistency checks:")
    #print(f"  - n_events from EventSummary: {n_events}")
    #print(f"  - n_events from MCtruth: {len(gen_theta)}")
    #print(f"  - n_events from M1: {len(rec_m1_theta)}")
    #print(f"  - n_events from M2: {len(rec_m2_theta)}")
    
    if len(gen_theta) != n_events:
        print(f"  ⚠️ WARNING: MCtruth has {len(gen_theta)} events, EventSummary has {n_events}")
    
    # Controlla se gli eventi hanno lo stesso numero di tracce
    #if n_tracks_per_event > 0:
    #    all_same = all(len(theta) == n_tracks_per_event for theta in gen_theta[:min(10, len(gen_theta))])
    #    print(f"  - All events have same n_tracks? {all_same}")
    #
    #print("="*60 + "\n")
    

    # Calcola n_tracks totali
    n_tracks_mc = sum(len(theta) if hasattr(theta, '__len__') else 1 for theta in gen_theta)
    n_tracks_m1 = len(rec_m1_theta)
    n_tracks_m2 = len(rec_m2_theta)
    '''
    
    # ========================================
    # RETURN
    # ========================================
    
    return {
        "mult_data": mult_data,
        "n_events": n_events,
        #"n_tracks_per_event": n_tracks_per_event,
        "n_tracks": {             
            "mc": n_tracks_mc,
            #"m1": n_tracks_m1,
            #"m2": n_tracks_m2
        }
    }

def res_json(input_dir, output_dir, config, part="e"):
    """
    Main function to calculate and plot residuals distributions.
    """
    os.makedirs(output_dir, exist_ok=True)

    root_files = sorted(glob.glob(os.path.join(input_dir, "*_ALL*.root")))
    #coord_map = {"x": 0, "y": 1}
    #method_map = {"m1": 1, "m2": 2}
    #layer_map = {"l1": 1, "l2": 2, "l3": 3}

    results = {
        "particle": part,
        "betap" : [],
        "method1": {
            "efficiency": [],
            "efficiency_err": [],
            "fake_rate": [],
            "fake_rate_err": []
        },
        "method2": {
            "efficiency": [],
            "efficiency_err": [],
            "fake_rate": [],
            "fake_rate_err": []
        }
    }

    for file in root_files:
        
        print(f"Processing {file}")

        f = uproot.open(file)
        file_data = read_data(f)

        print(file_data.keys())

        n_reco_m1 = 0
        n_tot_reco_m1 = 0
        n_reco_m2 = 0
        n_tot_reco_m2 = 0

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
        results["betap"].append(betap)

        print(f"  Energy: {energy} MeV, mass: {mass:.3f} MeV, betap: {betap:.3f}")

        #base = os.path.basename(file)
        #prefix = base.split("MAIN")[0].rstrip("_")
        #n_tracks_per_event = len(file_data["gen_data"]["theta"][0])

        #out_file.cd()

        n_events = file_data["n_events"]
        n_gen = file_data["n_tracks"]["mc"]
        print(f"Events: {n_events}, Number of generated tracks: {n_gen}")

        stats = {
            "m1": {"n_single": 0, "n_multi": 0, "n_fake": 0, "n_tot_reco": 0},
            "m2": {"n_single": 0, "n_multi": 0, "n_fake": 0, "n_tot_reco": 0}
        }
        
        for ev_idx in range(n_events):
            for method in ["m1","m2"]:

                mult = file_data['mult_data'][method][ev_idx]
                #reco_idx = np.where(ev_idx == file_data["rec_data"][method]["event_idx"])[0]

                if mult == 0:
                    continue

                stats[method]["n_tot_reco"] += mult 
                stats[method]["n_single"] += 1
                stats[method]["n_multi"] += (mult - 1)

        # Calcola efficienza = eventi con 1 traccia ricostruita / totale eventi
        # OPPURE = tracce singole / tracce generate
        efficiency_m1 = stats["m1"]["n_single"] / n_gen if n_gen > 0 else 0
        efficiency_m2 = stats["m2"]["n_single"] / n_gen if n_gen > 0 else 0
        
        # Errore sull'efficienza (approssimazione gaussiana)
        eff_err_m1 = np.sqrt(efficiency_m1 * (1 - efficiency_m1) / n_gen) if n_gen > 0 else 0
        eff_err_m2 = np.sqrt(efficiency_m2 * (1 - efficiency_m2) / n_gen) if n_gen > 0 else 0
        
        # Fake rate = (tracce totali - tracce singole) / tracce totali
        fake_rate_m1 = (stats["m1"]["n_tot_reco"] - stats["m1"]["n_single"]) / stats["m1"]["n_tot_reco"] if stats["m1"]["n_tot_reco"] > 0 else 0
        fake_rate_m2 = (stats["m2"]["n_tot_reco"] - stats["m2"]["n_single"]) / stats["m2"]["n_tot_reco"] if stats["m2"]["n_tot_reco"] > 0 else 0
        
        # Errore sul fake rate
        fake_err_m1 = np.sqrt(fake_rate_m1 * (1 - fake_rate_m1) / stats["m1"]["n_tot_reco"]) if stats["m1"]["n_tot_reco"] > 0 else 0
        fake_err_m2 = np.sqrt(fake_rate_m2 * (1 - fake_rate_m2) / stats["m2"]["n_tot_reco"]) if stats["m2"]["n_tot_reco"] > 0 else 0

        # Stampa risultati
        print(f"  Method 1 - Efficiency: {stats["m1"]["n_single"]} / {n_gen} = {efficiency_m1:.4f} ± {eff_err_m1:.4f}, Fake Rate: ({stats["m1"]["n_tot_reco"] - stats["m1"]["n_single"]}) / {stats["m1"]["n_tot_reco"]} = {fake_rate_m1:.4f} ± {fake_err_m1:.4f}")
        print(f"  Method 2 - Efficiency: {stats["m2"]["n_single"]} / {n_gen} = {efficiency_m2:.4f} ± {eff_err_m2:.4f}, Fake Rate: ({stats["m2"]["n_tot_reco"] - stats["m2"]["n_single"]}) / {stats["m2"]["n_tot_reco"]} = {fake_rate_m2:.4f} ± {fake_err_m2:.4f}\n")

        # Salva risultati
        results["method1"]["efficiency"].append(float(efficiency_m1))
        results["method1"]["efficiency_err"].append(float(eff_err_m1))
        results["method1"]["fake_rate"].append(float(fake_rate_m1))
        results["method1"]["fake_rate_err"].append(float(fake_err_m1))
        
        results["method2"]["efficiency"].append(float(efficiency_m2))
        results["method2"]["efficiency_err"].append(float(eff_err_m2))
        results["method2"]["fake_rate"].append(float(fake_rate_m2))
        results["method2"]["fake_rate_err"].append(float(fake_err_m2))

    # Ordina per betap
    #for method in ["method1", "method2"]:
    #    sorted_indices = np.argsort(results["betap"])
    #    for key in results[method]:
    #        results[method][key] = np.array(results[method][key])[sorted_indices].tolist()

    # Salva in JSON
    output_json = os.path.join(output_dir, f"efficiency_fake_rate_{part}.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Saved results to: {output_json}")               

    '''
                    mc_reconstructed, reco_to_gen = is_track_reconstructed(
                        ev_idx, method, file_data["rec_data"][method]["cls"]["x"][reco_idx], 
                        file_data["rec_data"][method]["cls"]["y"][reco_idx], file_data["rec_data"][method]["cls"]["z"][reco_idx], 
                        file_data["gen_data"]["cls"]["x"][ev_idx], file_data["gen_data"]["cls"]["y"][ev_idx], 
                        file_data["gen_data"]["cls"]["layer"][ev_idx], file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx], 
                        n_tracks_per_event, reco_idx
                    )   
                    matched_reco_id = set(reco_to_gen.values())
                    #print(f"Event {ev_idx}, method {method}, matched reco IDs: {matched_reco_id}, mult: {mult}")
                    
                            
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
        
        # Scrivi il tree subito dopo averlo riempito
        tree.Write()
        '''
    '''
    c = ROOT.TCanvas(f"c_eff_fake_betap", "", 800, 600)
    c.Divie(2)
    c.cd(0)
    eff_vs_bp_m1 = ROOT.TGraphErrors(len(beta_p))
    eff_vs_bp_m2 = ROOT.TGraphErrors(len(beta_p))
    for i in range(len(beta_p)):
        eff_vs_bp_m1.SetPoint(i, beta_p[i], efficiency_m1[i])
        eff_vs_bp_m1.SetPointError(i, 0, eff_err_m1[i])
        eff_vs_bp_m2.SetPoint(i, beta_p[i], efficiency_m2[i])
        eff_vs_bp_m2.SetPointError(i, 0, eff_err_m2[i])

    eff_vs_bp_m1.SetTitle("Efficiency VS #betap")
    eff_vs_bp_m1.GetYAxis.SetTitle("Efficiency")
    eff_vs_bp_m1.GetXAxis.SetTitle("#betap [MeV]")
    eff_vs_bp_m1.SetMarkerColor(ROOT.kBlue)
    eff_vs_bp_m2.SetMarkerColor(ROOT.kRed)
    eff_vs_bp_m1.Draw("AP")
    eff_vs_bp_m2.Draw("P SAME")    
    
    leg = ROOT.TLegend(0.73, 0.79, 0.90, 0.90)
    leg.AddEntry(eff_vs_bp_m1, "Hough", "p")
    leg.AddEntry(h_purity_m2, "Comb.", "p")
    leg.SetTextSize(0.04)
    leg.SetFillStyle(1001)
    leg.SetFillColor(ROOT.kWhite)
    leg.Draw()
    
    c.Print(pdf_purity_comp)
    c.Close()                        
    #print(f"residuals.root file created in {output_dir}")
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get residual distributions.")
    parser.add_argument("--input-dir", required=True, help="Use this directory containing .root files.")
    parser.add_argument("--output-dir", default=None, help="Define output directory.")
    parser.add_argument("--particle", required=True, help="Select particle configuration.")
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
    
    res_json(args.input_dir, output_dir, config, args.particle)