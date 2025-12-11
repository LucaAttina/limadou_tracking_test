import ROOT
import uproot
import awkward as ak
import math
import argparse
import os
import numpy as np

from geometry_utils import (
    load_geometry,
    track_hit_TR,
    is_in_acceptance,
    handle_two_cluster_track,
)

ROOT.gStyle.SetOptStat(0)

from utils import safe_first, load_and_select_events, analyze_event


# ============================================================
# Text output and histogram filling
# ============================================================
def write_txt_dump(
    txt_output,
    arrays,
    h_ncls,
    h_samez,
    h_resx,
    h_resy,
    h_dsum_vs_ncls,
    counters,
):
    """Write event details into a TXT file and fill histograms."""
    print(f"📝 Writing detailed event dump to {txt_output}")
    with open(txt_output, "w") as f:

        for i, evt in enumerate(arrays):
            tracks = analyze_event(evt)

            f.write(f"=== Event {i} ===\n")

            for trk in tracks:

                # --- Fill histograms ---
                n_cls = trk.n_cls
                h_ncls.Fill(n_cls)

                # Dsum now taken from trk.D_sum (data member)
                Dsum = trk.D_sum
                h_dsum_vs_ncls.Fill(Dsum, n_cls)

                for c in trk.clusters:
                    h_resx.Fill(c.res_x)
                    h_resy.Fill(c.res_y)

                # Issues detection (compute locally from clusters to avoid dependency
                # on analyze_event populating trk.issues)
                issue_meanx = any(c.mean_x == -999 for c in trk.clusters)
                same_z_count = len(trk.clusters) - len(
                    set(c.mean_z for c in trk.clusters)
                )

                if issue_meanx:
                    counters["bad_meanx"] += 1
                if not trk.hit_tr:
                    counters["no_TR_hit"] += 1
                if same_z_count >= 2:
                    counters["same_z_tracks"] += 1

                # --- Write track header ---
                f.write(
                    f"Track {trk.track_idx}: "
                    f"x0={trk.x0:.5f}, y0={trk.y0:.5f}, theta={trk.theta:.5f}, phi={trk.phi:.5f}\n"
                )
                f.write(
                    f"  n_cls={trk.n_cls}, Dsum={Dsum:.5f}, "
                    f"hit_TR={int(trk.hit_tr)}, missing_in_acc={int(trk.missing_in_acc)}\n"
                )

                # --- Issues ---
                if issue_meanx or same_z_count >= 2 or not trk.hit_tr:
                    f.write("  ⚠️ Issues:\n")
                    if issue_meanx:
                        f.write("    - mean_x = -999\n")
                    if same_z_count >= 2:
                        f.write("    - ≥2 clusters same Z\n")
                    if not trk.hit_tr:
                        f.write("    - no_TR_hit\n")

                # --- Clusters ---
                for j, c in enumerate(trk.clusters):
                    f.write(
                        f"  Cluster {j}: "
                        f"x={c.mean_x:.3f}, y={c.mean_y:.3f}, z={c.mean_z:.3f}, "
                        f"size={c.size}, res_x={c.res_x:.3f}, res_y={c.res_y:.3f}\n"
                    )

                f.write("\n")


# ============================================================
# ROOT tree output
# ============================================================
def create_ttree(root_file, arrays):
    """Create and write a ROOT TTree with selected event-level info."""

    root_file.cd()
    tree_out = ROOT.TTree("SelectedEvents", "Selected Events after filtering")

    x0_val = np.zeros(1, dtype=np.float32)
    y0_val = np.zeros(1, dtype=np.float32)
    theta_val = np.zeros(1, dtype=np.float32)
    phi_val = np.zeros(1, dtype=np.float32)
    n_cls = np.zeros(1, dtype=np.int32)
    hit_tr = np.zeros(1, dtype=np.int32)
    in_acc = np.zeros(1, dtype=np.int32)

    tree_out.Branch("x0", x0_val, "x0/F")
    tree_out.Branch("y0", y0_val, "y0/F")
    tree_out.Branch("theta", theta_val, "theta/F")
    tree_out.Branch("phi", phi_val, "phi/F")
    tree_out.Branch("n_cls", n_cls, "n_cls/I")
    tree_out.Branch("track_hit_TR", hit_tr, "track_hit_TR/I")
    tree_out.Branch("missing_in_acceptance", in_acc, "missing_in_acceptance/I")

    for evt in arrays:
        tracks = analyze_event(evt)

        # One TTree entry per *track*
        for trk in tracks:
            x0_val[0] = trk.x0
            y0_val[0] = trk.y0
            theta_val[0] = trk.theta
            phi_val[0] = trk.phi
            n_cls[0] = trk.n_cls
            hit_tr[0] = int(trk.hit_tr)
            in_acc[0] = int(trk.missing_in_acc)
            tree_out.Fill()

    tree_out.Write()
    print("✅ TTree 'SelectedEvents' written.")


# ============================================================
# Summary histogram creation
# ============================================================
def make_summary_hist(arrays, output_file, output_dir):

    print("\n📊 Building summary histogram...")

    h_summary = ROOT.TH1F("h_summary", "Event summary counts;;Counts", 9, 0.5, 9.5)
    labels = [
        "Tracks after masks",
        "n_cls = 2",
        "n_cls > 3",
        "track_hit_TR",
        "n_cls = 2 + TR + out of acc",
        "mean_x = -999",
        "Dsum > 10",
        "clusters same z",
        "Good tracks",
    ]

    for i, lab in enumerate(labels, start=1):
        h_summary.GetXaxis().SetBinLabel(i, lab)

    total_after_masks = 0
    ncls_eq2 = ncls_gt3 = track_hit_tr_count = 0
    ncls2_tr_missing_out = 0
    bad_meanx = dsum_gt10 = samez_ge2 = good_tracks = 0

    for evt in arrays:
        tracks = analyze_event(evt)

        total_after_masks += len(tracks)

        for trk in tracks:

            n_cls = trk.n_cls
            # use D_sum data member
            Dsum = trk.D_sum
            issue_meanx = any(c.mean_x == -999 for c in trk.clusters)
            same_z_count = len(trk.clusters) - len(set(c.mean_z for c in trk.clusters))

            # Basic counts
            if n_cls == 2:
                ncls_eq2 += 1
            if n_cls > 3:
                ncls_gt3 += 1
            if trk.hit_tr:
                track_hit_tr_count += 1
            if n_cls == 2 and trk.hit_tr and not trk.missing_in_acc:
                ncls2_tr_missing_out += 1

            if issue_meanx:
                bad_meanx += 1
            if Dsum > 10:
                dsum_gt10 += 1
            if same_z_count >= 2:
                samez_ge2 += 1

            # Good track
            if (
                n_cls < 4
                and not issue_meanx
                and trk.hit_tr
                and (n_cls != 2 or not trk.missing_in_acc)
                and Dsum < 10
                and same_z_count == 0
            ):
                good_tracks += 1

    # Fill histogram bins
    values = [
        total_after_masks,
        ncls_eq2,
        ncls_gt3,
        track_hit_tr_count,
        ncls2_tr_missing_out,
        bad_meanx,
        dsum_gt10,
        samez_ge2,
        good_tracks,
    ]

    for i, v in enumerate(values, start=1):
        h_summary.SetBinContent(i, v)

    # Print summary
    print("\n===== Summary Counts =====")
    for lab, val in zip(labels, values):
        perc = val / total_after_masks * 100 if total_after_masks else 0
        print(f"{lab:<45}: {val:6d} ({perc:5.2f}%)")

    # Save
    output_file.cd()
    h_summary.Write()

    c_summary = ROOT.TCanvas("c_summary", "Summary", 1000, 600)
    c_summary.SetBottomMargin(0.28)
    h_summary.SetFillColor(ROOT.kAzure - 4)
    h_summary.Draw("hist text0")

    pdf_path = os.path.join(output_dir, "summary_counts.pdf")
    c_summary.SaveAs(pdf_path)
    c_summary.Write()

    print(f"\n💾 Saved summary canvas to: {pdf_path}")
    print("📂 Stored inside ROOT file.")
    print("✅ Summary histogram creation complete.\n")


# ============================================================
# Main orchestrator
# ============================================================
def extract_selected_info(input_file, output_dir, save_tree=False):
    load_geometry()

    masks_to_apply = {
        "trig": False,
        "trig_count": True,
        "x0_multiplicity": True,
        "x0_m2_multiplicity": True,
    }

    multiplicity_config = {
        "x0_count": 1,
        "x0_m2_count": 0,
    }

    arrays, counters = load_and_select_events(
        input_file,
        masks_to_apply=masks_to_apply,
        multiplicity_config=multiplicity_config,
    )

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_file))[0]
    txt_output = os.path.join(output_dir, f"{base}_selected.txt")
    root_output = os.path.join(output_dir, f"{base}_selected.root")

    # Histograms
    h_ncls = ROOT.TH1F(
        "h_ncls", "Number of Clusters per Event;N_{cls};Entries", 16, -0.5, 15.5
    )
    h_samez = ROOT.TH1F(
        "h_samez", "Clusters with same Z per Event;Count;Entries", 16, -0.5, 15.5
    )
    h_resx = ROOT.TH1F("h_resx", "Residual X;res_x;Entries", 100, -0.5, 0.5)
    h_resy = ROOT.TH1F("h_resy", "Residual Y;res_y;Entries", 100, -0.5, 0.5)

    dsum_edges = np.concatenate(
        [
            np.linspace(0, 50, 20, endpoint=False),
            np.linspace(50, 200, 15, endpoint=False),
            np.linspace(200, 1000, 10, endpoint=False),
            np.linspace(1000, 2000, 6),
        ]
    ).astype(np.float64)

    h_dsum_vs_ncls = ROOT.TH2F(
        "h_dsum_vs_ncls",
        "Dsum vs Ncls;Dsum;N_{cls}",
        len(dsum_edges) - 1,
        dsum_edges,
        16,
        -0.5,
        15.5,
    )

    counters = {"bad_meanx": 0, "no_TR_hit": 0, "same_z_tracks": 0}

    write_txt_dump(
        txt_output, arrays, h_ncls, h_samez, h_resx, h_resy, h_dsum_vs_ncls, counters
    )

    root_file = ROOT.TFile(root_output, "RECREATE")
    h_ncls.Write()
    h_samez.Write()
    h_resx.Write()
    h_resy.Write()
    h_dsum_vs_ncls.Write()

    if save_tree:
        create_ttree(root_file, arrays)

    make_summary_hist(arrays, root_file, output_dir)
    root_file.Close()

    print("✅ Done.\n")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and analyze L2 events.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--save-tree", action="store_true")
    args = parser.parse_args()

    extract_selected_info(args.input, args.output_dir, args.save_tree)
