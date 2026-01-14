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
from utils import load_and_select_events, analyze_event
from geometry_utils import load_geometry

# === GLOBAL SETTINGS ===
tree_name = "L2"
#input_directory = "/home/lattina/limadou/test_L2/L2/" # Adjust as needed

# pairs, range and x label (ROOT style)
branch_pairs = {
    ("x0", "x0_m2"): {"range": (-90, 90), "xlabel": "x_{0} (mm)"},
    ("y0", "y0_m2"): {"range": (-90, 90), "xlabel": "y_{0} (mm)"},
    ("phi", "phi_m2"): {"range": (-180, 180), "xlabel": "#phi (#circ)"},
    ("theta", "theta_m2"): {
        "range": (0, 90),
        "xlabel": "#theta (#circ)",
    },
}

'''
# Keep the helper functions
def make_hist_from_jagged(name, arr_jagged, nbins, xmin, xmax):
    h = ROOT.TH1F(name, name, nbins, xmin, xmax)
    for evt in arr_jagged:
        for val in evt:
            h.Fill(float(val))
    return h
'''

# === GET LIST OF ROOT FILES ===

'''def find_root_files(input_directory):
    root_files = glob.glob(os.path.join(input_directory, "MC*LVL2.root"))
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in: {input_directory}")
    print(f"Found {len(root_files)} ROOT files to process")
    return root_files'''


def return_output_name(output_dir, input_file, suffix):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(output_dir, f"{base_name}{suffix}")

"""
# Keep the helper functions
def make_hist_from_jagged(name, arr_jagged, nbins, xmin, xmax):
    h = ROOT.TH1F(name, name, nbins, xmin, xmax)
    for evt in arr_jagged:
        for val in evt:
            h.Fill(float(val))
    return h
"""
'''
# Function to process a single file
def access_root_file(input_file_name):

    print(f"\nProcessing: {input_file_name}")

    output_file_name = return_output_name(
        output_dir, input_file_name, "_distributions.root"
    )

    out_file = ROOT.TFile(output_file_name, "RECREATE")

    # === READ TREE ===
    input_file = uproot.open(input_file_name)
    if tree_name not in input_file:
        print(f"⚠️ TTree '{tree_name}' not found in {input_file_name}, skipping...")
        return

    tree = input_file[tree_name]
    total_events = tree.num_entries
    print(f"✅ Read TTree '{tree_name}' with {total_events} events.")

    return tree, total_events
'''
'''
def cluster_multiplicity_per_event(arrays, output_dir, input_file_name, out_file):
    """Plot and save cluster multiplicity per selected event.
       -tree: uproot tree
    """
    nsel = len(arrays)

    if nsel == 0:
        print("⚠️ No selected events for cluster multiplicity plot. Skipping.")
        return

    cluster_multiplicities = ak.to_numpy(ak.num(arrays["L2Event/cls_size"]))

    # Create and fill multiplicity histogram
    cClsMult = ROOT.TCanvas("cClsMult", "cClsMult", 900, 700)
    h_cls_mult = ROOT.TH1F(
        "h_cls_mult",
        "; clusters per event; Entries",
        cluster_multiplicities.max() - cluster_multiplicities.min() + 1,
        cluster_multiplicities.min() - 0.5,
        cluster_multiplicities.max() + 0.5,
    )

    for mult in cluster_multiplicities:
        h_cls_mult.Fill(mult)

    # Print statistics
    print("Cluster multiplicity stats:")
    print(f"  Mean: {cluster_multiplicities.mean():.2f}")
    print(f"  Std Dev: {cluster_multiplicities.std():.2f}")
    print(f"  Min: {cluster_multiplicities.min()}")
    print(f"  Max: {cluster_multiplicities.max()}")

    # Draw and save
    h_cls_mult.Draw("HIST")
    pave_cls = ROOT.TPaveText(0.65, 0.80, 0.88, 0.88, "brNDC")
    pave_cls.AddText(f"Selected events: {nsel}")
    pave_cls.AddText(f"Events with no clusters: {np.sum(cluster_multiplicities == 0)}")   
    pave_cls.SetFillColor(0)
    pave_cls.Draw()

    out_file.cd()
    h_cls_mult.Write()  # Save cluster multiplicity histogram

    # Save PDF
    pdf_name = return_output_name(
        output_directory,
        input_file_name,
        "_cluster_multiplicity.pdf",
    )
    cClsMult.Print(pdf_name)

    print(f"✅ Cluster multiplicity plot saved in '{pdf_name}'")
'''

'''
# ====================================
# X0 vs X0_m2 multiplicity TH2
# ====================================
def make_multiplicity_th2(arrays, output_dir, base_name="multiplicity"):
    # compute multiplicities per event
    x0_mult = ak.num(arrays["L2Event/x0"])
    x0_m2_mult = ak.num(arrays["L2Event/x0_m2"])

    max_x0 = int(ak.max(x0_mult)) if len(x0_mult) > 0 else 0
    max_x0_m2 = int(ak.max(x0_m2_mult)) if len(x0_m2_mult) > 0 else 0

    nbx = max(1, max_x0 + 1)
    nby = max(1, max_x0_m2 + 1)

    # create TH2F with integer bins centered on integers
    hist_name = "h_x0_vs_x0m2"
    h2 = ROOT.TH2F(
        hist_name,
        "x0 multiplicity vs x0_m2 multiplicity; x0 multiplicity; x0_m2 multiplicity",
        nbx,
        -0.5,
        nbx - 0.5,
        nby,
        -0.5,
        nby - 0.5,
    )

    for xi, yi in zip(ak.to_numpy(x0_mult), ak.to_numpy(x0_m2_mult)):
        h2.Fill(float(xi), float(yi))

    # Save
    out_root = os.path.join(output_dir, f"{base_name}.root")
    f = ROOT.TFile(out_root, "RECREATE")
    h2.Write()
    f.Close()

    # Save a canvas as PDF
    c = ROOT.TCanvas("c_mult", "Multiplicity", 800, 600)
    h2.Draw("COLZ TEXT")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    c.SaveAs(pdf_path)
    print(f"Saved multiplicity TH2 to {out_root} and {pdf_path}")

    return h2
'''

'''
# ===================================
# Write text dump
# ===================================
def write_txt_dump(
    arrays, output_dir, base_name="cluster_details", x0_count=1, x0_m2_count=0
):
    """
    Write event details to a text file for events matching given multiplicities.

    Only outputs tracks that satisfy:
    - hit_tr == True
    - n_cls < 4
    - if n_cls == 2, then missing_in_acc == False

    Dumps all available info for all selected tracks and their clusters.
    Also prints a global summary of tracks before selection (console and TXT file).
    Adds intersection check for layer 1 for tracks with n_cls==3 and summarizes its fractions.
    """

    # --------------------------
    # Select events matching multiplicities
    # --------------------------
    x0_mult = ak.num(arrays["L2Event/x0"])
    x0_m2_mult = ak.num(arrays["L2Event/x0_m2"])

    sel_mask = (x0_mult == x0_count) & (x0_m2_mult == x0_m2_count)
    nevt = int(ak.sum(sel_mask)) #number of events matching x0_mult = 1 and x0_m2_mult = 0
    print(f"Events for dump (x0=={x0_count}, x0_m2=={x0_m2_count}): {nevt}")

    sel = arrays[sel_mask]
    if len(sel) == 0:
        print(
            f"No events matching x0=={x0_count} && x0_m2=={x0_m2_count}. Skipping dump."
        )
        return

    # --------------------------
    # Global summary counters (before filtering)
    # --------------------------
    total_counts = {"n_cls_2": 0, "n_cls_3": 0, "n_cls_gt3": 0}
    hit_tr_counts = {"n_cls_2": 0, "n_cls_3": 0, "n_cls_gt3": 0}
    missing_in_acc_counts = {"n_cls_2": 0}
    hit_and_not_missing_counts = {"n_cls_2": 0}

    # For n_cls==3: intersection counters
    n_cls3_intersection_yes = 0
    n_cls3_intersection_yes_and_hit = 0

    for evt in sel:
        track_list = analyze_event(evt)
        for trk in track_list:
            if trk.n_cls == 2: # check for 2-cluster tracks
                total_counts["n_cls_2"] += 1
                if trk.hit_tr:
                    hit_tr_counts["n_cls_2"] += 1
                if not trk.missing_in_acc:
                    missing_in_acc_counts["n_cls_2"] += 1
                if trk.hit_tr and not trk.missing_in_acc:
                    hit_and_not_missing_counts["n_cls_2"] += 1
            elif trk.n_cls == 3: # check for 3-cluster tracks
                total_counts["n_cls_3"] += 1
                if trk.hit_tr:
                    hit_tr_counts["n_cls_3"] += 1

                # Intersection check
                if len(trk.clusters) == 3:
                    clus0 = trk.clusters[0]
                    clus1 = trk.clusters[1]
                    clus2 = trk.clusters[2]
                    x1 = 0.5 * (clus0.mean_x + clus2.mean_x)
                    y1 = 0.5 * (clus0.mean_y + clus2.mean_y)
                    dx = abs(clus1.mean_x - x1)
                    dy = abs(clus1.mean_y - y1)
                    intersection = math.hypot(dx, dy) < 0.5 # distance between layer1 cluster and midpoint of layer0/2 clusters
                    if intersection:
                        n_cls3_intersection_yes += 1
                        if trk.hit_tr:
                            n_cls3_intersection_yes_and_hit += 1

            elif trk.n_cls > 3:
                total_counts["n_cls_gt3"] += 1
                if trk.hit_tr:
                    hit_tr_counts["n_cls_gt3"] += 1

    # --------------------------
    # Prepare output text file
    # --------------------------
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(txt_path, "w") as f:
        f.write("# Event dump for selected events\n")
        f.write(f"# x0 multiplicity: {x0_count}, x0_m2 multiplicity: {x0_m2_count}\n")
        f.write(
            "# Filters: hit_tr==True, n_cls<4, if n_cls==2 then missing_in_acc==False\n"
        )
        f.write(
            "# For n_cls==3, intersection_layer1 indicates if layer1 cluster intersects midpoint of layer0/2 clusters\n\n"
        )

        for evt_idx, evt in enumerate(sel):
            track_list = analyze_event(evt)
            total_clusters_all = len(evt["L2Event/cls_mean_x"])

            selected_tracks = [ # choose tracks satisfying the conditions
                trk
                for trk in track_list
                if trk.hit_tr
                and (trk.n_cls == 3 or (trk.n_cls == 2 and not trk.missing_in_acc))
            ]

            if len(selected_tracks) == 0:
                continue

            f.write(f"Event {evt_idx}\n")
            f.write(f"  n_tracks={len(selected_tracks)}\n")
            f.write(f"  n_clusters_total={total_clusters_all}\n")

            for trk in selected_tracks:
                intersection_layer1 = "N/A"
                if trk.n_cls == 3 and len(trk.clusters) == 3:
                    clus0 = trk.clusters[0]
                    clus1 = trk.clusters[1]
                    clus2 = trk.clusters[2]
                    x1 = 0.5 * (clus0.mean_x + clus2.mean_x)
                    y1 = 0.5 * (clus0.mean_y + clus2.mean_y)
                    dx = abs(clus1.mean_x - x1)
                    dy = abs(clus1.mean_y - y1)
                    intersection_layer1 = "Yes" if math.hypot(dx, dy) < 0.5 else "No"

                f.write(
                    f"  Track {trk.track_idx}: "
                    f"x0={trk.x0:.5f}, y0={trk.y0:.5f}, theta={trk.theta:.5f}, phi={trk.phi:.5f}, "
                    f"n_cls={trk.n_cls}, hit_tr={int(trk.hit_tr)}, missing_in_acc={int(trk.missing_in_acc)}, "
                    f"D_sum={trk.D_sum:.5f}, intersection_layer1={intersection_layer1}\n"
                )

                for i, c in enumerate(trk.clusters):
                    attrs = ", ".join(f"{k}={getattr(c, k)}" for k in vars(c))
                    f.write(f"    Cluster {i}: {attrs}\n")

            f.write("\n")

        # --------------------------
        # Global summary
        # --------------------------
        f.write("===== Global Track Summary (before filtering) =====\n")
        for key in ["n_cls_3", "n_cls_2", "n_cls_gt3"]:
            tot = total_counts.get(key, 0)
            hits = hit_tr_counts.get(key, 0)
            frac_hits = hits / tot * 100 if tot else 0
            if key == "n_cls_2":
                missing = missing_in_acc_counts.get(key, 0)
                frac_missing = missing / tot * 100 if tot else 0
                both = hit_and_not_missing_counts.get(key, 0)
                frac_both = both / tot * 100 if tot else 0
                f.write(
                    f"{key}: total={tot}, hit_tr={hits} ({frac_hits:.1f}%), "
                    f"missing_in_acc=False={missing} ({frac_missing:.1f}%), "
                    f"hit_tr and missing_in_acc=False={both} ({frac_both:.1f}%)\n"
                )
            elif key == "n_cls_3":
                frac_intersection = n_cls3_intersection_yes / tot * 100 if tot else 0
                frac_intersection_hit = n_cls3_intersection_yes_and_hit / tot * 100 if tot else 0
                f.write(
                    f"{key}: total={tot}, hit_tr={hits} ({frac_hits:.1f}%), "
                    f"intersection_layer1=Yes: {n_cls3_intersection_yes} ({frac_intersection:.1f}%), "
                    f"hit_tr AND intersection_layer1=Yes: {n_cls3_intersection_yes_and_hit} ({frac_intersection_hit:.1f}%)\n"
                )
            else:
                f.write(f"{key}: total={tot}, hit_tr={hits} ({frac_hits:.1f}%)\n")

    # --------------------------
    # Print same summary in console
    # --------------------------
    print("\n===== Global Track Summary (before filtering) =====")
    for key in ["n_cls_3", "n_cls_2", "n_cls_gt3"]:
        tot = total_counts.get(key, 0)
        hits = hit_tr_counts.get(key, 0)
        frac_hits = hits / tot * 100 if tot else 0
        if key == "n_cls_2":
            missing = missing_in_acc_counts.get(key, 0)
            frac_missing = missing / tot * 100 if tot else 0
            both = hit_and_not_missing_counts.get(key, 0)
            frac_both = both / tot * 100 if tot else 0
            print(
                f"{key}: total={tot}, hit_tr: {hits} ({frac_hits:.1f}%), "
                f"missing_in_acceptance=False: {missing} ({frac_missing:.1f}%), "
                f"hit_tr and missing_in_acceptace=False: {both} ({frac_both:.1f}%)"
            )
        elif key == "n_cls_3":
            frac_intersection = n_cls3_intersection_yes / tot * 100 if tot else 0
            frac_intersection_hit = n_cls3_intersection_yes_and_hit / tot * 100 if tot else 0
            print(
                f"{key}: total={tot}, hit_tr: {hits} ({frac_hits:.1f}%), "
                f"intersection_layer1=Yes: {n_cls3_intersection_yes} ({frac_intersection:.1f}%), "
                f"hit_tr AND intersection_layer1=Yes: {n_cls3_intersection_yes_and_hit} ({frac_intersection_hit:.1f}%)"
            )
        else:
            print(f"{key}: total: {tot}, hit_tr: {hits} ({frac_hits:.1f}%)")

    print(f"Saved event dump to {txt_path}")
'''

# ==================================
# Theta, Phi, X0 and Y0 comparison
# ==================================
def compare_params_distributions(
    arrays, output_dir, base_name="params_compare", x0_count=None, x0_m2_count=None, nbins=60
):
    """Compare theta/theta_m2, phi/phi_m2, x0/x0_m2 and y0/y0_m2 for events matching given multiplicities.

    Parameters:
      - arrays: awkward arrays returned by load_and_select_events (may already be filtered)
      - x0_count, x0_m2_count: multiplicity requirements used for the selection inside this function
    """
    
    norm_hist = False # set to True to normalize histograms to unit area

    # select events where x0 multiplicity == x0_count and x0_m2 multiplicity == x0_m2_count
    if x0_count is not None and x0_m2_count is not None:
        x0_mult = ak.num(arrays["L2Event/x0"])
        x0_m2_mult = ak.num(arrays["L2Event/x0_m2"])
        sel_mask = (x0_mult == x0_count) & (x0_m2_mult == x0_m2_count)
        sel = arrays[sel_mask]
        nevt = int(ak.sum(sel_mask))
        print(f"Events with x0=={x0_count} and x0_m2=={x0_m2_count}: {nevt}")
        # use all events in arrays if no multiplicity requirements provided
    else:
        sel = arrays
        nevt = len(sel)
    
    if len(sel) == 0:
        print(
            f"No events matching x0=={x0_count} && x0_m2=={x0_m2_count}. Skipping comparisons."
        )
        return 
    
    if nevt == 0:
        print("⚠️ No selected events for comparison. Skipping.")
        return 
    
    ROOT.gStyle.SetOptStat(0)
    ROOT.gROOT.SetBatch(True)

    c = ROOT.TCanvas("c_compare", "Parameter comparison", 900, 700)

    out_root = os.path.join(output_dir, f"{base_name}.root")
    fout = ROOT.TFile(out_root, "RECREATE")

    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    c.Print(pdf_path + "[")

    # Loop over parameter pairs
    for (b1, b2), cgf in branch_pairs.items():
        xmin, xmax = cgf["range"]
        xlabel = cgf.get("xlabel", b1)

        h1 = ROOT.TH1F(f"h_{b1}_cmp", f"{xlabel}_hist;{xlabel};Entries", int(nbins), float(xmin), float(xmax))
        h2 = ROOT.TH1F(f"h_{b2}_cmp", f"{xlabel}_hist;{xlabel};Entries", int(nbins), float(xmin), float(xmax))

        # TRACK-LEVEL filling
        for evt in sel:
            for v in evt[f"L2Event/{b1}"]:
                h1.Fill(float(v))
            for v in evt[f"L2Event/{b2}"]:
                h2.Fill(float(v))

        if h1.Integral() == 0 or h2.Integral() == 0:
            print(f"⚠️ No entries for {b1} or {b2}. Skipping.")
            continue

        # Normalization
        if norm_hist:
            h1.Scale(1.0 / h1.Integral(), "width")
            h2.Scale(1.0 / h2.Integral(), "width")

        # Style
        h1.SetLineColor(ROOT.kBlue)
        h2.SetLineColor(ROOT.kRed)
        h1.SetLineWidth(2)
        h2.SetLineWidth(2)

        ymax = max(h1.GetMaximum(), h2.GetMaximum()) * 1.2
        h1.SetMaximum(ymax)

        # Draw
        h1.Draw("HIST")
        h2.Draw("HIST SAME")

        leg = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
        leg.AddEntry(h1, b1, "l")
        leg.AddEntry(h2, b2, "l")
        leg.SetBorderSize(0)
        leg.Draw()

        c.Print(pdf_path)

        fout.cd()
        h1.Write()
        h2.Write()

    c.Print(pdf_path + "]")
    fout.Close()

    print(f"✅ Saved comparison plots to {pdf_path}")
    print(f"✅ Saved histograms to {out_root}")



    '''
    # extract theta and theta_m2 arrays (take first element if per-event array)
    theta = ak.to_numpy(ak.fill_none(ak.flatten(sel["L2Event/theta"]), np.nan))
    theta_m2 = ak.to_numpy(ak.fill_none(ak.flatten(sel["L2Event/theta_m2"]), np.nan))

    phi = ak.to_numpy(ak.fill_none(ak.flatten(sel["L2Event/phi"]), np.nan))
    phi_m2 = ak.to_numpy(ak.fill_none(ak.flatten(sel["L2Event/phi_m2"]), np.nan))

    # remove NaNs
    theta = theta[~np.isnan(theta)]
    theta_m2 = theta_m2[~np.isnan(theta_m2)]
    phi = phi[~np.isnan(phi)]
    phi_m2 = phi_m2[~np.isnan(phi_m2)]

    # create histograms with automatic ranges
    def auto_hist(name, data, nbins=80):
        if len(data) == 0:
            return ROOT.TH1F(name, name, nbins, -1.0, 1.0)
        mn = float(np.min(data))
        mx = float(np.max(data))
        if math.isclose(mn, mx):
            mn -= 0.001
            mx += 0.001
        return ROOT.TH1F(name, name, nbins, mn, mx)

    # build histograms using combined ranges so overlays match
    h_theta = auto_hist("h_theta", np.concatenate([theta, theta_m2]))
    h_theta_m2 = auto_hist("h_theta_m2", np.concatenate([theta, theta_m2]))

    # Fill
    for v in theta:
        h_theta.Fill(v)
    for v in theta_m2:
        h_theta_m2.Fill(v)

    # style and save overlay
    h_theta.SetLineColor(ROOT.kBlue)
    h_theta.SetLineWidth(2)
    h_theta_m2.SetLineColor(ROOT.kRed)
    h_theta_m2.SetLineWidth(2)

    c_th = ROOT.TCanvas("c_theta", "Theta comparison", 800, 600)
    h_theta.Draw()
    h_theta_m2.Draw("SAME")
    leg = ROOT.TLegend(0.7, 0.75, 0.88, 0.88)
    leg.AddEntry(h_theta, "theta", "l")
    leg.AddEntry(h_theta_m2, "theta_m2", "l")
    leg.Draw()
    pdf_theta = os.path.join(output_dir, f"{base_name}_theta.pdf")
    c_th.SaveAs(pdf_theta)

    # phi
    h_phi = auto_hist("h_phi", np.concatenate([phi, phi_m2]))
    h_phi_m2 = auto_hist("h_phi_m2", np.concatenate([phi, phi_m2]))
    for v in phi:
        h_phi.Fill(v)
    for v in phi_m2:
        h_phi_m2.Fill(v)
    h_phi.SetLineColor(ROOT.kBlue)
    h_phi_m2.SetLineColor(ROOT.kRed)
    h_phi.SetLineWidth(2)
    h_phi_m2.SetLineWidth(2)

    c_ph = ROOT.TCanvas("c_phi", "Phi comparison", 800, 600)
    h_phi.Draw()
    h_phi_m2.Draw("SAME")
    leg2 = ROOT.TLegend(0.7, 0.75, 0.88, 0.88)
    leg2.AddEntry(h_phi, "phi", "l")
    leg2.AddEntry(h_phi_m2, "phi_m2", "l")
    leg2.Draw()
    pdf_phi = os.path.join(output_dir, f"{base_name}_phi.pdf")
    c_ph.SaveAs(pdf_phi)

    # Save histograms to ROOT file
    out_root = os.path.join(output_dir, f"{base_name}.root")
    f = ROOT.TFile(out_root, "RECREATE")
    h_theta.Write()
    h_theta_m2.Write()
    h_phi.Write()
    h_phi_m2.Write()
    f.Close()

    print(f"Saved angle comparison histograms to {out_root} and {pdf_theta}, {pdf_phi}")


    return {
        "h_theta": h_theta,
        "h_theta_m2": h_theta_m2,
        "h_phi": h_phi,
        "h_phi_m2": h_phi_m2,
    }
'''

#===================================
# Main execution
#===================================
if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Apply selection on L2 events and get histograms and dumps. Masks and multiplicity requirements can be set via CLI or YAML config."
    )

    # Input options
    parser.add_argument(
        "--config",
        help="YAML config file with options (keys: output_dir, mask_trig, mask_trig_count, apply_multiplicity, x0_count, x0_m2_count or nested 'masks')",
        default=None,
    )
    parser.add_argument(
        "--input-dir", 
        help="Input directory (overrides config)", 
        required=True,
        default=None
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all ROOT files in the input directory",
    )
    parser.add_argument(
        "--file", 
        type=str, 
        help="Process a single ROOT file in the input directory (insert filename)", 
    )

    # Output directory
    parser.add_argument(
        "--output-dir", 
        default=None, 
        help="Output directory (overrides config)"
    )

    # Masks and multiplicity options
    # Use default=None for boolean flags so we can detect whether CLI provided them
    parser.add_argument(
        "--mask-trig",
        action="store_true",
        default=None,
        help="Apply trig mask (overrides config)",
    )
    parser.add_argument(
        "--mask-trig-count",
        action="store_true",
        default=None,
        help="Apply trig_count mask (overrides config)",
    )
    parser.add_argument(
        "--x0-count",
        type=int,
        default=None,
        help="x0 multiplicity requirement (overrides config)",
    )
    parser.add_argument(
        "--x0-m2-count",
        type=int,
        default=None,
        help="x0_m2 multiplicity requirement (overrides config)",
    )
    parser.add_argument(
        "--apply-multiplicity",
        action="store_true",
        default=None,
        help="Apply multiplicity masks inside load_and_select_events (overrides config)",
    )

    # Output histograms and dumps
    parser.add_argument(
        "--compare-params",
        action="store_true",
        help="Compare parameters distributions for selected events (theta vs theta_m2, phi vs phi_m2, x0 vs x0_m2, y0 vs y0_m2)",
    )
    '''parser.add_argument(
        "--multi-th2",
        action="store_true",
        help="Create multiplicity TH2 x0 vs x0_m2",
    )'''
    '''parser.add_argument(
        "--cls-mult-per-event",
        action="store_true",
        help="Get cluster multiplicity per event histogram",
    )'''
    args = parser.parse_args()

    # Load YAML config if provided
    cfg = {}
    if args.config:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to use --config but 'yaml' could not be imported. Install pyyaml."
            )
        with open(args.config, "r") as cf:
            cfg = yaml.safe_load(cf) or {}

    def cfg_get(key, default=None):
        return cfg.get(key, default)
    
    if args.input_dir is None:
        print("⚠️ Please specify an input directory via --input-dir.")
        sys.exit(1)
    input_dir = args.input_dir

    # Determine final values (CLI overrides config)
    '''input_directory = args.file if args.file is not None #else cfg_get("input_dir")
    if input_directory is None:
        parser.error("Input file must be specified via --file or in the config file")'''

    if args.all and args.file:
        print("⚠️ Please specify only one option: either --all or --file.")
        sys.exit(1)

    if not args.all and not args.file:
        print("❗ Please specify an option: --all or --file")
        sys.exit(1)


    # =========================
    # Build list of ROOT files
    # =========================

    # if --all is chosen, add all ROOT files in root_files
    if args.all:
        root_files = glob.glob(os.path.join(input_dir, "MC*.root"))
        if not root_files:
            raise FileNotFoundError(f"No ROOT files found in: {input_dir}")
    # if --file is chosen, process that single file
    else:
        root_file = os.path.join(input_dir, args.file)
        if not os.path.exists(root_file):
            raise FileNotFoundError(f"File not found: {root_file}")
        root_files = [root_file]

    print(f"Found {len(root_files)} ROOT files to process")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else cfg_get("output_dir", "/home/lattina/limadou/test_L2/output")
    )

    # masks: allow top-level keys or nested 'masks' mapping in the config
    masks_cfg = cfg_get("masks", {}) or {}

    mask_trig = (
        args.mask_trig
        if args.mask_trig is not None
        else cfg_get("mask_trig", masks_cfg.get("trig", False))
    )
    mask_trig_count = (
        args.mask_trig_count
        if args.mask_trig_count is not None
        else cfg_get("mask_trig_count", masks_cfg.get("trig_count", False))
    )

    apply_multiplicity = (
        args.apply_multiplicity
        if args.apply_multiplicity is not None
        else cfg_get("apply_multiplicity", masks_cfg.get("apply_multiplicity", False))
    )

    x0_count = args.x0_count if args.x0_count is not None else cfg_get("x0_count", None)
    x0_m2_count = (
        args.x0_m2_count if args.x0_m2_count is not None else cfg_get("x0_m2_count", None)
    )

    # Normalize booleans
    mask_trig = bool(mask_trig)
    mask_trig_count = bool(mask_trig_count)
    apply_multiplicity = bool(apply_multiplicity)

    os.makedirs(output_dir, exist_ok=True)

    masks = {
        "trig": mask_trig,
        "trig_count": mask_trig_count,
        "x0_multiplicity": bool(apply_multiplicity),
        "x0_m2_multiplicity": bool(apply_multiplicity),
    }

    multiplicity_config = {
        "x0_count": int(x0_count) if x0_count is not None else None,
        "x0_m2_count": int(x0_m2_count) if x0_m2_count is not None else None
    }

    load_geometry()

    # suffix outputs with multiplicity combo
    suffix = f"_x0_{x0_count}_x0m2_{x0_m2_count}"

    # access each ROOT file and get tree
    arrays_list = []
    for root_file in root_files:
        arrays_tmp, counters = load_and_select_events(
            root_file, 
            masks_to_apply={"trig": False, "trig_count": False, "x0_multiplicity": False, "x0_m2_multiplicity": False},
            multiplicity_config=multiplicity_config
        )
        arrays_list.append(arrays_tmp)

    if len(arrays_list) > 0:
        if len(arrays_list) == 1:
            arrays = arrays_list[0]
        else:
            arrays = ak.concatenate(arrays_list)
    else:
        print("⚠️ No events loaded from any file. Exiting.")
        sys.exit(1)

    # Build multiplicity TH2 (for the selected events)
    '''
    if args.multi_th2:
        h2 = make_multiplicity_th2(
            arrays, 
            output_dir, 
            base_name="multiplicity_selected" + suffix
        )

    # Write cluster details for events matching the multiplicities
    write_txt_dump(
        arrays,
        output_dir,
        base_name="cluster_details" + suffix,
        x0_count=x0_count,
        x0_m2_count=x0_m2_count,
    )
    '''

    # Compare angle distributions for the selected events
    if args.compare_params:
        compare_params_distributions(
            arrays, 
            output_dir, 
            base_name="param_compare" + suffix, 
            x0_count=x0_count, 
            x0_m2_count=x0_m2_count
        )

    ''' 
    # Cluster multiplicity per event
    if args.cls_mult_per_event:
        cluster_multiplicity_per_event(
            arrays,
            output_dir,
            input_file_name,
            out_file,
        )'''