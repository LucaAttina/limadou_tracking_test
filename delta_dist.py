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


def get_th2_projection(histo, min_val, max_val, name, x_name, y_name, method, output_dir):
    """
    Get projection of TH2 on X axis for a given Y range.
    """
    ROOT.gStyle.SetOptStat(1111)
    
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    
    ybin1 = histo.GetYaxis().FindBin(min_val)
    ybin2 = histo.GetYaxis().FindBin(max_val)
    h_proj = histo.ProjectionX(f"proj_{name}", ybin1, ybin2)
    
    c_proj = ROOT.TCanvas(f"c_proj_{name}", "", 800, 700)
    
    proj_title = (f"Projection of {y_name} on {x_name} in [{min_val}, {max_val}] - {method};{x_name} (deg);Entries")
    
    h_proj.SetTitle(proj_title)
    h_proj.SetLineWidth(2)
    h_proj.Draw("HIST")
    
    proj_dir = os.path.join(output_dir, "th2_projection")
    os.makedirs(proj_dir, exist_ok=True)
    
    pdf_proj = os.path.join(proj_dir, f"proj_{y_name}_on_{x_name}_{method}.pdf")
    
    c_proj.Print(pdf_proj)
    c_proj.Close()


def plot_correlation(x_vals, y_vals, x_name, y_name, method, output_dir, 
                     xmin, xmax, nbins_x, ymin, ymax, nbins_y, n_tracks_per_event, title):
    """
    Plot TH2F correlation histogram.
    
    Args:
        x_vals, y_vals: array of values on x/y axis
        x_name, y_name: names for plot saving
        method: M1 or M2
        output_dir: directory where the pdf file is saved
        xmin, xmax, nbins_x: x interval and number of bins on x axis
        ymin, ymax, nbins_y: y interval and number of bins on y axis
        title: title and axes labels in the format "Title;x_label;y_label"
    """
    if len(x_vals) == 0 or len(y_vals) == 0:
        print(f"No data for correlation {y_name} vs {x_name} ({method}) - skipping")
        return
    
    ROOT.gStyle.SetOptStat(0)
    
    hname = f"h2_{x_name}_{y_name}_{method}"
    h2 = ROOT.TH2F(hname, title, nbins_x, xmin, xmax, nbins_y, ymin, ymax)
    
    for x, y in zip(x_vals, y_vals):
        h2.Fill(x, y)
    
    c = ROOT.TCanvas(f"c_{hname}", "", 800, 700)
    h2.Draw("COLZ")
    
    corr_dir = os.path.join(output_dir, "correlations")
    os.makedirs(corr_dir, exist_ok=True)
    
    pdf_file = os.path.join(corr_dir, f"corr_{y_name}_vs_{x_name}_{method}_{n_tracks_per_event}_evs.pdf")
    c.Print(pdf_file)
    c.Close()
    
    # Get projection of TH2 on x axis for a given y range
    get_th2_projection(h2, ymin, ymax, hname, x_name, y_name, method, output_dir)
    
    print(f"PDF saved for correlation at {pdf_file}")


def plot_delta_hist(delta_vals, var_name, method, output_dir, color, 
                    xmin, xmax, nbins, n_tracks_per_event, setlog=False, fit=False):
    """
    Plot delta distribution histogram.
    """
    if len(delta_vals) == 0:
        print(f"No entries for delta {var_name} ({method}) - histogram skipped")
        return None, None
    
    ROOT.gStyle.SetOptStat(1111)
    ROOT.gStyle.SetStatFontSize(0.06)

    hname = f"h_delta_{var_name}_{method}"
    title = f"#Delta#{var_name} distribution - {method} - {n_tracks_per_event} tracks per event;#Delta#{var_name} (deg);Entries"
    h = ROOT.TH1F(hname, title, nbins, xmin, xmax)
    
    for val in delta_vals:
        h.Fill(val)
    
    c = ROOT.TCanvas(f"c_{hname}", "", 800, 600)
    
    if setlog:
        c.SetLogy()
    
    h.SetLineColor(color)
    h.SetLineWidth(2)
    h.Draw("HIST")

    ROOT.gStyle.SetStatX(0.45)
    ROOT.gStyle.SetStatY(0.85)
    
    if fit:
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptFit(1111)
        
        fit_range = 0.2 if var_name == "theta" else 0.4
        fit_f = ROOT.TF1(f"gaus_fit_{var_name}_{method}", "gaus", -fit_range, fit_range)
        fit_f.SetLineColor(ROOT.kRed)
        fit_f.SetLineWidth(2)
        h.Fit(fit_f, "R")
        fit_f.Draw("SAME")
    
    c.Update()
    
    delta_dir = os.path.join(output_dir, "delta_distributions")
    os.makedirs(delta_dir, exist_ok=True)
    
    if setlog:
        pdf_file = os.path.join(delta_dir, f"delta_{var_name}_{method}_log_{n_tracks_per_event}_evs.pdf")
    else:
        pdf_file = os.path.join(delta_dir, f"delta_{var_name}_{method}_{n_tracks_per_event}_evs.pdf")
    
    c.Print(pdf_file)
    c.Close()
    print(f"PDF saved for #Delta{var_name} - {method} at {pdf_file}")
    
    return h


def plot_overlapping_deltas(h_theta_m1, h_theta_m2, h_phi_m1, h_phi_m2, output_dir, n_tracks_per_event, setlog):
    """
    Plot overlapping delta distributions for M1 and M2.
    """
    proj_dir = os.path.join(output_dir, "delta_distributions")
    os.makedirs(proj_dir, exist_ok=True)
    ROOT.gStyle.SetOptStat(0)
    
    # Theta overlap
    if h_theta_m1 and h_theta_m2:
        c_th = ROOT.TCanvas("c_th", "", 800, 600)
        if setlog:
            c_th.SetLogy()
        ymax_th = max(h_theta_m1.GetMaximum(), h_theta_m2.GetMaximum())
        
        h_theta_m1.SetTitle(f"#Delta#theta distribution - {n_tracks_per_event} tracks per event")
        h_theta_m1.SetMaximum(1.15 * ymax_th)
        h_theta_m1.SetLineColor(ROOT.kBlue)
        h_theta_m2.SetLineColor(ROOT.kRed)
        
        h_theta_m1.Draw("HIST")
        h_theta_m2.Draw("HIST SAME")
        
        leg_th = ROOT.TLegend(0.78, 0.84, 0.95, 0.95)
        leg_th.AddEntry(h_theta_m1, "M1", "lep")
        leg_th.AddEntry(h_theta_m2, "M2", "lep")
        leg_th.SetTextSize(0.04)
        leg_th.SetFillStyle(1001)
        leg_th.SetFillColor(ROOT.kWhite)
        leg_th.Draw()
        
        pdf_th = os.path.join(proj_dir, "delta_theta_overlap.pdf")
        c_th.Print(pdf_th)
        c_th.Close()
    
    # Phi overlap
    if h_phi_m1 and h_phi_m2:
        c_ph = ROOT.TCanvas("c_ph", "", 800, 600)
        if setlog:
            c_ph.SetLogy()
        ymax_ph = max(h_phi_m1.GetMaximum(), h_phi_m2.GetMaximum())
        
        h_phi_m1.SetTitle(f"#Delta#phi distribution - {n_tracks_per_event} tracks per event")
        h_phi_m1.SetMaximum(1.15 * ymax_ph)
        h_phi_m1.SetLineColor(ROOT.kBlue)
        h_phi_m2.SetLineColor(ROOT.kRed)
        
        h_phi_m1.Draw("HIST")
        h_phi_m2.Draw("HIST SAME")
        
        leg_ph = ROOT.TLegend(0.78, 0.84, 0.95, 0.95)
        leg_ph.AddEntry(h_phi_m1, "M1", "lep")
        leg_ph.AddEntry(h_phi_m2, "M2", "lep")
        leg_ph.SetTextSize(0.04)
        leg_ph.SetFillStyle(1001)
        leg_ph.SetFillColor(ROOT.kWhite)
        leg_ph.Draw()
        
        pdf_ph = os.path.join(proj_dir, "delta_phi_overlap.pdf")
        c_ph.Print(pdf_ph)
        c_ph.Close()


def delta_plots(input_file, output_dir):
    """
    Main function to calculate and plot delta distributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open file and read basic data
    f = uproot.open(input_file)
    print(f"\nProcessing file: {input_file}")
    
    # Read basic data using utility function
    file_data = read_data_from_file(f)
    
    # Print track counts
    n_events = len(file_data["mult_data"]["event_idx"])
    n_tracks_per_event = len(file_data["gen_data"]["theta"][0])
    print(f"Number of events: {n_events}")
    print(f"Tracks per event: {n_tracks_per_event}")
    
    # Initialize counters for delta analysis
    #counters = initialize_counters(["m1", "m2"], "delta")
    
    # Data containers for delta values
    delta_data = {
        "m1": {"delta_theta": [], "delta_phi": [], "gen_theta": [], "gen_phi": []},
        "m2": {"delta_theta": [], "delta_phi": [], "gen_theta": [], "gen_phi": []}
    }

    eff_info = {
        "m1": {},
        "m2": {}
    }

    for m in ["m1", "m2"]:
        for ev_idx in range(n_events):
            eff_info[m][ev_idx] = {}
            for trk_idx in range(n_tracks_per_event):
                eff_info[m][ev_idx][trk_idx] = False
    
    # Process all events
    for ev_idx in range(n_events):
        gen_trk_idx = file_data["gen_data"]["trk_id"][ev_idx]        

        for method in ["m1","m2"]:
            mult = file_data['mult_data'][method][ev_idx]
            reco_idx = np.where(ev_idx == file_data["rec_data"][method]["event_idx"])[0]
            if reco_idx.tolist():

                # Process delta event using utility function
                result = process_efficiency_event(
                    n_tracks_per_event,
                    ev_idx,
                    reco_idx,
                    gen_trk_idx,
                    mult,
                    file_data["rec_data"][method]["cls"]["x"][reco_idx],
                    file_data["rec_data"][method]["cls"]["y"][reco_idx],
                    file_data["rec_data"][method]["cls"]["z"][reco_idx],
                    file_data["gen_data"]["cls"]["x"][ev_idx],
                    file_data["gen_data"]["cls"]["y"][ev_idx],
                    file_data["gen_data"]["cls"]["layer"][ev_idx],
                    file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx],
                    file_data["rec_data"][method]["theta"][reco_idx],
                    file_data["rec_data"][method]["phi"][reco_idx],
                    file_data["gen_data"]["theta"][ev_idx],
                    file_data["gen_data"]["phi"][ev_idx],
                    #counters["m2"],
                    eff_info,
                    theta_angle_threshold=5.0,
                    phi_angle_threshold=5.0,
                    return_good_flag=False,
                    return_delta=True,
                    method=method
                )

                if result is not None:
                    _,_,_,_,delta_theta_array, delta_phi_array, gen_theta_array, gen_phi_array = result

                    # Usa extend perché sono array (possono avere lunghezza >1)
                    delta_data[method]["delta_theta"].extend(delta_theta_array)
                    delta_data[method]["delta_phi"].extend(delta_phi_array)
                    delta_data[method]["gen_theta"].extend(gen_theta_array)
                    delta_data[method]["gen_phi"].extend(gen_phi_array)
    
    # Print summary
    #print("\nTotal reconstructed tracks (well reconstructed + angles mismatches):")
    #print(f"- M1: {counters['m1']['tracks']}")
    #print(f"- M2: {counters['m2']['tracks']}\n")
    #
    #print("Outliers for M1:")
    #print(f"theta: {counters['m1']['theta_great_out'] + counters['m1']['theta_low_out']} "
    #      f"(> 0.5 deg: {counters['m1']['theta_great_out']}, < -0.5 deg: {counters['m1']['theta_low_out']})")
    #print(f"phi: {counters['m1']['phi_great_out'] + counters['m1']['phi_low_out']} "
    #      f"(> 1.5 deg: {counters['m1']['phi_great_out']}, < -1.5 deg: {counters['m1']['phi_low_out']})")
    #
    #print("Outliers for M2:")
    #print(f"theta: {counters['m2']['theta_great_out'] + counters['m2']['theta_low_out']} "
    #      f"(> 0.5 deg: {counters['m2']['theta_great_out']}, < -0.5 deg: {counters['m2']['theta_low_out']})")
    #print(f"phi: {counters['m2']['phi_great_out'] + counters['m2']['phi_low_out']} "
    #      f"(> 1.5 deg: {counters['m2']['phi_great_out']}, < -1.5 deg: {counters['m2']['phi_low_out']})\n")
    
    # Convert to numpy arrays
    for method in ["m1","m2"]:
        for key in delta_data[method]:
            delta_data[method][key] = np.asarray(delta_data[method][key])
    
    # Plot delta histograms (commented out in original)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)
    
    # Create all correlation plots
    create_all_plots(delta_data, output_dir, n_tracks_per_event)


def create_all_plots(delta_data, output_dir, n_tracks_per_event):
    """
    Create all correlation plots.
    """
    # Delta theta dist m1
    h_theta_m1 = plot_delta_hist(
        delta_data["m1"]["delta_theta"],
        "theta", "m1", output_dir, ROOT.kBlue,
        delta_data["m1"]["delta_theta"].min(), 
        delta_data["m1"]["delta_theta"].max(),
        30, n_tracks_per_event, setlog=True, fit=False
    )

    # Delta theta dist m2
    h_theta_m2 = plot_delta_hist(
        delta_data["m2"]["delta_theta"],
        "theta", "m2", output_dir, ROOT.kRed,
        delta_data["m2"]["delta_theta"].min(), 
        delta_data["m2"]["delta_theta"].max(),
        30, n_tracks_per_event, setlog=True, fit=False
    )

    # Delta phi dist m1
    h_phi_m1 = plot_delta_hist(        
        delta_data["m1"]["delta_phi"],
        "phi", "m1", output_dir, ROOT.kBlue,
        delta_data["m1"]["delta_phi"].min(), 
        delta_data["m1"]["delta_phi"].max(),
        30, n_tracks_per_event, setlog=True, fit=False
    )

    # Delta phi dist m2
    h_phi_m2 = plot_delta_hist(        
        delta_data["m2"]["delta_phi"],
        "phi", "m2", output_dir, ROOT.kRed,
        delta_data["m2"]["delta_phi"].min(), 
        delta_data["m2"]["delta_phi"].max(),
        30, n_tracks_per_event, setlog=True, fit=False
    )

    plot_overlapping_deltas(h_theta_m1, h_theta_m2, h_phi_m1, h_phi_m2, output_dir, n_tracks_per_event, setlog=True)

    # Delta theta vs gen theta
    plot_correlation(
        delta_data["m1"]["gen_theta"], delta_data["m1"]["delta_theta"],
        "theta", "delta_theta", "M1", output_dir,
        delta_data["m1"]["gen_theta"].min() if len(delta_data["m1"]["gen_theta"]) > 0 else 0,
        delta_data["m1"]["gen_theta"].max() if len(delta_data["m1"]["gen_theta"]) > 0 else 180,
        60, -0.5, 0.5, 30, n_tracks_per_event,
        f"#Delta#theta vs #theta_gen - M1 - {n_tracks_per_event} tracks per event;#theta_gen (deg);#Delta#theta (deg)"
    )
    
    plot_correlation(
        delta_data["m2"]["gen_theta"], delta_data["m2"]["delta_theta"],
        "theta", "delta_theta", "M2", output_dir,
        delta_data["m2"]["gen_theta"].min() if len(delta_data["m2"]["gen_theta"]) > 0 else 0,
        delta_data["m2"]["gen_theta"].max() if len(delta_data["m2"]["gen_theta"]) > 0 else 180,
        60, -0.5, 0.5, 30, n_tracks_per_event,
        f"#Delta#theta vs #theta_gen - M2 - {n_tracks_per_event} tracks per event;#theta_gen (deg);#Delta#theta (deg)"
    )
    
    # Delta phi vs gen phi
    plot_correlation(
        delta_data["m1"]["gen_phi"], delta_data["m1"]["delta_phi"],
        "phi", "delta_phi", "M1", output_dir,
        delta_data["m1"]["gen_phi"].min() if len(delta_data["m1"]["gen_phi"]) > 0 else -180,
        delta_data["m1"]["gen_phi"].max() if len(delta_data["m1"]["gen_phi"]) > 0 else 180,
        60, -1.5, 1.5, 30, n_tracks_per_event,
        f"#Delta#phi vs #phi_fgen - M1 - {n_tracks_per_event} tracks per event;#phi_gen (deg);#Delta#phi (deg)"
    )
    
    plot_correlation(
        delta_data["m2"]["gen_phi"], delta_data["m2"]["delta_phi"],
        "phi", "delta_phi", "M2", output_dir,
        delta_data["m2"]["gen_phi"].min() if len(delta_data["m2"]["gen_phi"]) > 0 else -180,
        delta_data["m2"]["gen_phi"].max() if len(delta_data["m2"]["gen_phi"]) > 0 else 180,
        60, -1.5, 1.5, 30, n_tracks_per_event,
        f"#Delta#phi vs #phi_gen - M2 - {n_tracks_per_event} tracks per event;#phi_gen (deg);#Delta#phi (deg)"
    )
    
    # Delta phi vs gen theta
    plot_correlation(
        delta_data["m1"]["gen_theta"], delta_data["m1"]["delta_phi"],
        "theta", "delta_phi", "M1", output_dir,
        delta_data["m1"]["gen_theta"].min() if len(delta_data["m1"]["gen_theta"]) > 0 else 0,
        delta_data["m1"]["gen_theta"].max() if len(delta_data["m1"]["gen_theta"]) > 0 else 180,
        60, -1.5, 1.5, 30, n_tracks_per_event,
        f"#Delta#phi vs #theta_gen - M1 - {n_tracks_per_event} tracks per event;#theta_gen (deg);#Delta#phi (deg)"
    )
    
    plot_correlation(
        delta_data["m2"]["gen_theta"], delta_data["m2"]["delta_phi"],
        "theta", "delta_phi", "M2", output_dir,
        delta_data["m2"]["gen_theta"].min() if len(delta_data["m2"]["gen_theta"]) > 0 else 0,
        delta_data["m2"]["gen_theta"].max() if len(delta_data["m2"]["gen_theta"]) > 0 else 180,
        60, -1.5, 1.5, 30, n_tracks_per_event,
        f"#Delta#phi vs #theta_gen - M2 - {n_tracks_per_event} tracks per event;#theta_gen (deg);#Delta#phi (deg)"
    )
    
    # Delta phi vs delta theta
    plot_correlation(
        delta_data["m1"]["delta_theta"], delta_data["m1"]["delta_phi"],
        "delta_theta", "delta_phi", "M1", output_dir,
        -0.5, 0.5, 30, -1.5, 1.5, 30, n_tracks_per_event,
        f"#Delta#phi vs #Delta#theta - M1 - {n_tracks_per_event} tracks per event;#Delta#theta (deg);#Delta#phi (deg)"
    )
    
    plot_correlation(
        delta_data["m2"]["delta_theta"], delta_data["m2"]["delta_phi"],
        "delta_theta", "delta_phi", "M2", output_dir,
        -0.5, 0.5, 30, -1.5, 1.5, 30, n_tracks_per_event,
        f"#Delta#phi vs #Delta#theta - M2 - {n_tracks_per_event} tracks per event;#Delta#theta (deg);#Delta#phi (deg)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get delta distributions for theta and phi angles.")
    parser.add_argument("--input-file", required=True, help="Use this .root file.")
    parser.add_argument("--output-dir", default=None, help="Define output directory.")
    args = parser.parse_args()
    
    config_file = "config/config_delta.yaml"
    with open(config_file, "r") as f_cfg:
        config = yaml.safe_load(f_cfg)
    
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = config["output_dir"]
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    delta_plots(args.input_file, output_dir)