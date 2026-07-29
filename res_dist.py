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
    is_track_reconstructed,
    read_data_from_file,
    process_efficiency_event
)
import json


def res_plots(input_file, output_dir, config, part="e"):
    """
    Main function to calculate and plot residuals distributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open file and read basic data
    f = uproot.open(input_file)
    p = f.get("radius")
    if not p:
        raise RuntimeError("TParameter 'radius' not found in file")
    r_val = p.value
    r_val = f"{r_val:.2f}".replace(".", "p")
    print(f"\nProcessing file: {input_file}")
    
    # Read basic data using utility function
    file_data = read_data_from_file(f)
    
    # Print track counts
    n_events = len(file_data["mult_data"]["event_idx"])
    n_tracks_per_event = len(file_data["gen_data"]["theta"][0])
    print(f"Number of events: {n_events}")
    print(f"Tracks per event: {n_tracks_per_event}")
    
    # Process all events
    if part == "e":
        particle = "electron"
        print(f"Selected particle: {particle}")
        emin = 1 # MeV
        emax = 150 # MeV
        nbin_dist = [35, 30, 30, 30, 30, 30, 30, 30, 35, 35]
        xlim = [0.5, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        fitlim_m1 = [0.4, 0.15, 0.12, 0.12, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1]
        fitlim_m2 = [0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.09, 0.09, 0.09]
    elif part == "p":
        particle = "proton"
        print(f"Selected particle: {particle}")        
        emin = 10 # MeV
        emax = 300 # MeV
        nbin_dist = [15, 18, 20, 20, 20, 20, 20, 20]
        xlim = [0.1, 0.1, 0.08, 0.08, 0.06, 0.04, 0.04, 0.03]
        fitlim = [0.05, 0.05, 0.03, 0.02, 0.02, 0.015, 0.015, 0.015]
    elif part == "c":
        particle = "carbon"
        print(f"Selected particle: {particle}")        
        emin = 100 # MeV
        emax = 4000 # MeV
    p_mass = file_data["gen_data"]["mass"][0][0]

    Etotmin = emin + p_mass
    Etotmax = emax + p_mass

    pmin2 = Etotmin**2 - p_mass**2
    pmax2 = Etotmax**2 - p_mass**2

    bpmin = pmin2 / Etotmin
    bpmax = pmax2 / Etotmax

    print(f"Beta*p range for {particle}: [{bpmin:.2f}, {bpmax:.2f}] MeV")
    betap_bins = np.linspace(bpmin, bpmax, len(xlim) + 1)  
    n_bins = len(betap_bins) - 1
    histograms = {
        "m1": {"l1": {"x": [], "y": []}, "l2": {"x": [], "y": []}, "l3": {"x": [], "y": []} },
        "m2": {"l1": {"x": [], "y": []}, "l2": {"x": [], "y": []}, "l3": {"x": [], "y": []} }
    }
    gaus_fits = {
        "m1": {"l1": {"x": [], "y": []}, "l2": {"x": [], "y": []}, "l3": {"x": [], "y": []} },
        "m2": {"l1": {"x": [], "y": []}, "l2": {"x": [], "y": []}, "l3": {"x": [], "y": []} }
    }


    for method in ["m1", "m2"]:
        for lay in ["l1", "l2", "l3"]:
            for coord in ["x", "y"]:
                for i in range(n_bins):
                    bin_low = betap_bins[i]
                    bin_high = betap_bins[i+1]
                    name = f"h_res_{method}_{lay}_{coord}_betap_{bin_low}_{bin_high}"
                    title = f"Residuals {coord} for {method.upper()} - #betap [{bin_low:.1f}, {bin_high:.1f}] MeV - Layer {lay.upper()}"
                    # Range iniziale, verrà aggiornato dopo
                    h = ROOT.TH1F(name, title, nbin_dist[i], -xlim[i], xlim[i])
                    histograms[method][lay][coord].append(h)

                    if part == "e":
                        if method == "m1":
                            fit_range = fitlim_m1[i]
                        else:
                            fit_range = fitlim_m2[i]

                    if lay == "l2" and part == "p":
                        fit_range += 0.01

                    if lay == "l2" and part == "e":
                        fit_range += 0.05

                    fit_name = f"gaus_{method}_{lay}_{coord}"
                    f = ROOT.TF1(fit_name, "gaus", -fit_range, fit_range)
                    gaus_fits[method][lay][coord].append(f)

    
    res_x_m1_list, res_y_m1_list = [], []
    res_x_m2_list, res_y_m2_list = [], []

    counts_per_bin = {
        "m1": {"l1": {"x": [0]*n_bins, "y": [0]*n_bins}, "l2": {"x": [0]*n_bins, "y": [0]*n_bins}, "l3": {"x": [0]*n_bins, "y": [0]*n_bins}},
        "m2": {"l1": {"x": [0]*n_bins, "y": [0]*n_bins}, "l2": {"x": [0]*n_bins, "y": [0]*n_bins}, "l3": {"x": [0]*n_bins, "y": [0]*n_bins}}
    }

    std_results = {
        "betap_bins": betap_bins.tolist(),
        "m1":{
            "l1": {
                "x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []
            },
            "l2": {
                "x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []
            },
            "l3": {
                "x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []
            },
        },
        "m2":{
            "l1": {
                "x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []
            },
            "l2": {
                "x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []
            },
            "l3": {
                "x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []
            }
        }
    }
    
    for ev_idx in range(n_events):
        for method in ["m1","m2"]:

            mult = file_data['mult_data'][method][ev_idx]
            reco_idx = np.where(ev_idx == file_data["rec_data"][method]["event_idx"])[0]
            
            if reco_idx.tolist():
                mc_reconstructed, reco_to_gen = is_track_reconstructed(
                    ev_idx, method, file_data["rec_data"][method]["cls"]["x"][reco_idx], 
                    file_data["rec_data"][method]["cls"]["y"][reco_idx], file_data["rec_data"][method]["cls"]["z"][reco_idx], 
                    file_data["gen_data"]["cls"]["x"][ev_idx], file_data["gen_data"]["cls"]["y"][ev_idx], 
                    file_data["gen_data"]["cls"]["layer"][ev_idx], file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx], 
                    n_tracks_per_event, reco_idx
                )   
                matched_reco_id = set(reco_to_gen.values())
                #print(f"Event {ev_idx}, method {method}, matched reco IDs: {matched_reco_id}, mult: {mult}")

                for trk_idx in range(n_tracks_per_event):
                    if mc_reconstructed[trk_idx]:
                        energy = file_data["gen_data"]["energy"][ev_idx][trk_idx]
                        mass = file_data["gen_data"]["mass"][ev_idx][trk_idx]
                        p2 = (energy + mass)**2 - mass**2
                        betap = p2 / (energy + mass)
                        
                        #print(f"Event {ev_idx}, method {method}, track {trk_idx} is reconstructed with energy {energy:.2f} MeV, mass {mass:.2f} MeV, beta*p {betap:.2f} MeV")
                        betap_bin_idx = -1
                        for i in range(n_bins):
                            if betap_bins[i] <= betap < betap_bins[i+1]:
                                betap_bin_idx = i
                                break

                        if betap_bin_idx == -1:
                            print(f"Warning: betap={betap:.2f} out of range [{betap_bins[0]}, {betap_bins[-1]}]")
                            continue
                        
                        for mult_id in range(mult):
                            if mult_id in matched_reco_id:
                                #print(f"Event {ev_idx}, method {method}, reco_id {reco_id} is matched")
                                n_cls = len([c for c in file_data["rec_data"][method]["cls"]["x"][reco_idx][mult_id] if c != -999])
                                if n_cls != 3:
                                    #print(f"Event {ev_idx}, method {method}, reco_id {reco_id} has {n_cls} clusters - skipping for residuals")
                                    continue  # Skip tracks that don't have 3 clusters

                                res_x = file_data["rec_data"][method]["res"]["x"][reco_idx][mult_id]
                                res_y = file_data["rec_data"][method]["res"]["y"][reco_idx][mult_id]

                                res_x_flat = np.array(res_x).flatten() if hasattr(res_x, 'flatten') else res_x
                                res_y_flat = np.array(res_y).flatten() if hasattr(res_y, 'flatten') else res_y
                                #print(f"{res_x_flat}, {res_y_flat}")
                                for i in range(len(res_x_flat)):
                                    if abs(file_data["rec_data"][method]["cls"]["z"][reco_idx][mult_id][i] - 17.825) < 1e-3:
                                        lay = "l1"
                                    elif abs(file_data["rec_data"][method]["cls"]["z"][reco_idx][mult_id][i] - 26.325) < 1e-3:
                                        lay = "l2"
                                    elif abs(file_data["rec_data"][method]["cls"]["z"][reco_idx][mult_id][i] - 34.825) < 1e-3:
                                        lay = "l3"
                                    
                                    if method == "m1":        
                                        histograms["m1"][lay]["x"][betap_bin_idx].Fill(res_x_flat[i])
                                        res_x_m1_list.append(res_x_flat[i])
                                        counts_per_bin["m1"][lay]["x"][betap_bin_idx] += 1
                                        histograms["m1"][lay]["y"][betap_bin_idx].Fill(res_y_flat[i])
                                        res_y_m1_list.append(res_y_flat[i])
                                        counts_per_bin["m1"][lay]["y"][betap_bin_idx] += 1
                                    else:
                                        histograms["m2"][lay]["x"][betap_bin_idx].Fill(res_x_flat[i])
                                        res_x_m2_list.append(res_x_flat[i])
                                        counts_per_bin["m2"][lay]["x"][betap_bin_idx] += 1
                                        histograms["m2"][lay]["y"][betap_bin_idx].Fill(res_y_flat[i])
                                        res_y_m2_list.append(res_y_flat[i])
                                        counts_per_bin["m2"][lay]["y"][betap_bin_idx] += 1


                                #print(f"Event {ev_idx}, method {method}, reco_id {reco_id}, n_cls {n_cls}, res_x {file_data['rec_data'][method]['res']['x'][reco_idx][reco_id]}, res_y {file_data['rec_data'][method]['res']['y'][reco_idx][reco_id]}")
                                #print(f"clusters x: {file_data['rec_data'][method]['cls']['x'][reco_idx][reco_id]}, clusters y: {file_data['rec_data'][method]['cls']['y'][reco_idx][reco_id]}, clusters z: {file_data['rec_data'][method]['cls']['z'][reco_idx][reco_id]}")
                            else:
                                #print(f"Event {ev_idx}, method {method}, reco_id {reco_id} is NOT matched to any gen track - skipping for residuals")
                                pass

    res_x_m1_array = np.array(res_x_m1_list)
    res_y_m1_array = np.array(res_y_m1_list)
    res_x_m2_array = np.array(res_x_m2_list)
    res_y_m2_array = np.array(res_y_m2_list)

    print(f"\n=== Statistiche globali - {particle} ===")
    print(f"Residuals m1 X: mean={res_x_m1_array.mean():.4f}, std={res_x_m1_array.std():.4f}")
    print(f"Residuals m1 Y: mean={res_y_m1_array.mean():.4f}, std={res_y_m1_array.std():.4f}")
    print(f"Residuals m2 X: mean={res_x_m2_array.mean():.4f}, std={res_x_m2_array.std():.4f}")
    print(f"Residuals m2 Y: mean={res_y_m2_array.mean():.4f}, std={res_y_m2_array.std():.4f}")
    
    # Stampa conteggi per bin
    print(f"\n=== Conteggi per bin di betap ===")
    for i in range(n_bins):
        print(f"Bin {i} [{betap_bins[i]}, {betap_bins[i+1]}] MeV:")
        for lay in ["l1", "l2", "l3"]:
            print(f"  {lay.upper()}: m1 X: {counts_per_bin['m1'][lay]['x'][i]}, m1 Y: {counts_per_bin['m1'][lay]['y'][i]}")
            print(f"  {lay.upper()}: m2 X: {counts_per_bin['m2'][lay]['x'][i]}, m2 Y: {counts_per_bin['m2'][lay]['y'][i]}")
    
    res_dir = os.path.join(output_dir, f"residuals_vs_betap")
    os.makedirs(res_dir, exist_ok=True)

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(111)



    for lay in ["l1", "l2", "l3"]:
        for i in range(n_bins):
            bin_low = betap_bins[i]
            bin_high = betap_bins[i+1]

            c = ROOT.TCanvas(f"c_betap_bin_{i}_{lay}", f"Betap bin [{bin_low:.1f}, {bin_high:.1f}] MeV - {lay}", 800, 600)
            c.Divide(2, 2)

            # m1 X
            c.cd(1)
            h = histograms["m1"][lay]["x"][i]
            f = gaus_fits["m1"][lay]["x"][i]
            if h.GetEntries() > 0:
                h.SetLineColor(ROOT.kBlue)
                h.SetLineWidth(2)
                h.GetXaxis().SetTitle("Res [mm]")  
                h.GetYaxis().SetTitle("Entries")
                h.Draw("HIST")
                #h.GetXaxis().SetRangeUser(h.GetMean() - 3 * h.GetStdDev(), h.GetMean() + 3 * h.GetStdDev())
                #c.Update()
                # Fit gaussiano
                h.Fit(f, "RQ")  # Q = quiet mode
                stats = h.FindObject("stats")
                if stats:
                    stats.SetTextSize(0.04)
                f.SetLineColor(ROOT.kRed)
                f.Draw("SAME")
                std_results["m1"][lay]["x_std_dev"].append(f.GetParameter(2))
                std_results["m1"][lay]["x_std_err"].append(f.GetParError(2))

            else:
                ROOT.TLatex(0.5, 0.5, "No entries").Draw()

            # m1 Y
            c.cd(2)
            h = histograms["m1"][lay]["y"][i]
            f = gaus_fits["m1"][lay]["y"][i]
            if h.GetEntries() > 0:
                h.SetLineColor(ROOT.kBlue)
                h.SetLineWidth(2)
                h.GetXaxis().SetTitle("Res [mm]")  
                h.GetYaxis().SetTitle("Entries")
                h.Draw("HIST")
                h.Fit(f, "RQ")  # Q = quiet mode
                stats = h.FindObject("stats")
                if stats:
                    stats.SetTextSize(0.04)
                f.SetLineColor(ROOT.kRed)
                f.Draw("SAME")
                std_results["m1"][lay]["y_std_dev"].append(f.GetParameter(2))
                std_results["m1"][lay]["y_std_err"].append(f.GetParError(2))

            else:
                ROOT.TLatex(0.5, 0.5, "No entries").Draw()

            # m2 X
            c.cd(3)
            h = histograms["m2"][lay]["x"][i]
            f = gaus_fits["m2"][lay]["x"][i]
            if h.GetEntries() > 0:
                h.SetLineColor(ROOT.kGreen)
                h.SetLineWidth(2)
                h.GetXaxis().SetTitle("Res [mm]")  
                h.GetYaxis().SetTitle("Entries")
                h.Draw("HIST")
                h.Fit(f, "RQ")  # Q = quiet mode
                stats = h.FindObject("stats")
                if stats:
                    stats.SetTextSize(0.04)
                f.SetLineColor(ROOT.kRed)
                f.Draw("SAME")
                std_results["m2"][lay]["x_std_dev"].append(f.GetParameter(2))
                std_results["m2"][lay]["x_std_err"].append(f.GetParError(2))

            else:
                ROOT.TLatex(0.5, 0.5, "No entries").Draw()

            # m2 Y
            c.cd(4)
            h = histograms["m2"][lay]["y"][i]
            f = gaus_fits["m2"][lay]["y"][i]
            if h.GetEntries() > 0:
                h.SetLineColor(ROOT.kGreen)
                h.SetLineWidth(2)
                h.GetXaxis().SetTitle("Res [mm]")  
                h.GetYaxis().SetTitle("Entries")
                h.Draw("HIST")
                h.Fit(f, "RQ")  # Q = quiet mode
                stats = h.FindObject("stats")
                if stats:
                    stats.SetTextSize(0.04)
                f.SetLineColor(ROOT.kRed)
                f.Draw("SAME")
                std_results["m2"][lay]["y_std_dev"].append(f.GetParameter(2))
                std_results["m2"][lay]["y_std_err"].append(f.GetParError(2))

            else:
                ROOT.TLatex(0.5, 0.5, "No entries").Draw()

            pdf_name = os.path.join(res_dir, f"residuals_betap_{bin_low:.0f}_{bin_high:.0f}_{lay}.pdf")
            c.Print(pdf_name)
            c.Close()
    
    print(f"PDF salvati in {res_dir}")

    # Save standard deviation results

    with open(os.path.join(output_dir, f"{part}_std_dev_results_r_{r_val}.json"), "w") as f:
        data = {
            "particle": particle,
            "results": std_results
        }     

        json.dump(data, f, indent=4)

    print(f"Standard deviation results saved in {os.path.join(output_dir, f"{particle}_std_dev_results_r_{r_val}.json")}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get residual distributions.")
    parser.add_argument("--input-file", required=True, help="Use this .root file.")
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
    
    res_plots(args.input_file, output_dir, config, args.particle)