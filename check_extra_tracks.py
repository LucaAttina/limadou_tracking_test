import sys
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

from utils import safe_first, load_and_select_events, analyze_event, get_all_clusters, remove_duplicate_tracks

# ============================================================
# Text output and histogram filling (single method)
# ============================================================
def write_txt_dump(
    txt_output,
    results,
    h_ncls,
    h_samez,
    h_resx,
    h_resy,
    h_dsum_vs_ncls,
    counters,
    method="",
):
    """Write event details into a TXT file and fill histograms for the selected method."""
    print(f"📝 Writing detailed event dump to {txt_output}")
    with open(txt_output, "w") as f:

        good_tracks = 0
        trk_num = 0

        # txt header
        if method == "":
            f.write(f"{'='*20} Method M1 (Hough) {'='*20}\n\n")
        elif  method == "_m2":
            f.write(f"{'='*20} Method M2 (Combinatorial) {'='*20}\n\n")

        for i, evt in enumerate(results):
            if method == "":
                m_tr = evt["m1"]
            elif method == "_m2":
                m_tr = evt["m2"]   
            all_clusters = evt["clusters"]         

            # skip event if no tracks
            if len(m_tr) == 0:
                continue 

            trk_num += len(m_tr)
            f.write(f"=== Event {i} ===\n")
            for trk in m_tr:

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

                # conditions for good tracks
                is_good_track = (
                        n_cls < 4
                        and not issue_meanx
                        and trk.hit_tr
                        and (n_cls != 2 or not trk.missing_in_acc)
                        and Dsum < 10
                        and same_z_count == 0
                )

                # --- Write track header ---
                cluster_ids = [str(c.cluster_idx) for c in trk.clusters]
                if is_good_track:
                    f.write("### GOOD TRACK ###\n")
                    good_tracks += 1
                f.write(
                    f"* Track {trk.track_idx}: trk_idx={trk.track_idx}\n"
                    f"x0{method}={trk.x0:.5f}, y0{method}={trk.y0:.5f}, theta{method}={trk.theta:.5f}, phi{method}={trk.phi:.5f}\n"
                )
                f.write(
                    f"  n_cls={trk.n_cls}, Dsum={Dsum:.5f}, "
                    f"hit_TR={int(trk.hit_tr)}, missing_in_acc={int(trk.missing_in_acc)}\n"
                )
                f.write(f"clusters used: {', '.join(cluster_ids)}\n")

                # --- Issues ---
                if issue_meanx or same_z_count > 0 or not trk.hit_tr or trk.missing_in_acc:
                    f.write("  ⚠️ === Issues: === \n")
                    if issue_meanx:
                        f.write("    - mean_x = -999\n")
                    if same_z_count >= 2:
                        f.write("    - ≥2 clusters same Z\n")
                    if not trk.hit_tr:
                        f.write("    - no_TR_hit\n")
                    if trk.missing_in_acc:
                        f.write("  - missing point in acceptance\n")
                    if same_z_count > 0:
                        f.write("  - clusters with same Z\n")

                # --- Clusters ---
                for j, c in enumerate(trk.clusters):
                    f.write(
                        f"- Cluster {j}: cls_idx={c.cluster_idx}\n"
                        f"x={c.mean_x:.3f}, y={c.mean_y:.3f}, z={c.mean_z:.3f}, "
                        f"size={c.size}, res_x={c.res_x:.3f}, res_y={c.res_y:.3f}\n"   #TBM
                    )
                
                f.write("\n")

            # --- ALL CLUSTERS at the end of the event ---
            f.write("#" * 50 + "\n")
            f.write("# ALL CLUSTERS FOR THIS EVENT\n")
            f.write("#" * 50 + "\n")
            for j, c in enumerate(all_clusters):
                if c.cluster_idx == -999:
                    continue
                f.write(
                    f"- Cluster {j}: ID={c.cluster_idx}\n"
                    f"  x={c.mean_x:.3f}, y={c.mean_y:.3f}, z={c.mean_z:.3f}, size={c.size}\n"
                )
            f.write("\n" + "=" * 50 + "\n\n")
        
        f.write("\n" + "#" * 60)
        f.write(f"\nTracks number = {trk_num}")
        f.write(f"\nGood tracks = {good_tracks}")
        f.write(f"\nGood tracks ratio = {good_tracks/trk_num*100:.2f}%" if trk_num > 0 else "\nGood tracks ratio = 0.00%")


# ============================================================
# Text output and histogram filling (comparison m1 vs m2)
# ============================================================
def compare_txt(
    txt_output,
    results,
    h_ncls_m1, h_ncls_m2,
    h_samez_m1, h_samez_m2,
    h_resx_m1, h_resx_m2,
    h_resy_m1, h_resy_m2,
    h_dsum_vs_ncls_m1, h_dsum_vs_ncls_m2,
    counters_m1, counters_m2,
    debug_all=False,
):
    """Write event details into a TXT file comparing m1 and m2 side by side.
       If debug_all is False, skip events with same multiplicity and good tracks in both methods."""
    print(f"📝 Writing detailed event dump (m1 vs m2) to {txt_output}")

    trk_num_m1 = 0
    trk_num_m2 = 0
    good_tracks_m1 = 0
    good_tracks_m2 = 0
    
    with open(txt_output, "w") as f:
        for i, evt in enumerate(results):

            all_clusters = evt["clusters"]        
            tracks_m1 = evt["m1"]     
            tracks_m2 = evt["m2"]
            
            # Skip events with no tracks
            if len(tracks_m1) == 0 and len(tracks_m2) == 0:
                continue

            # check if we should skip this event (if not in debug_all mode)
            if not debug_all:
                should_skip = True
                # condition 1: different multiplicity
                if len(tracks_m1) != len(tracks_m2):
                    should_skip = False
                # condition 2: difference in good track
                else:
                    for track_num in range(len(tracks_m1)):
                        if track_num >= len(tracks_m2):
                            continue

                        trk_m1 = tracks_m1[track_num]
                        trk_m2 = tracks_m2[track_num]

                        # check if tracks are good
                        is_good_m1 = (
                            trk_m1.n_cls < 4
                            and not any(c.mean_x == -999 for c in trk_m1.clusters)
                            and trk_m1.hit_tr
                            and (trk_m1.n_cls != 2 or not trk_m1.missing_in_acc)
                            and trk_m1.D_sum < 10
                            and (len(trk_m1.clusters) - len(set(c.mean_z for c in trk_m1.clusters))) == 0
                        )

                        is_good_m2 = (
                            trk_m2.n_cls < 4
                            and not any(c.mean_x == -999 for c in trk_m2.clusters)
                            and trk_m2.hit_tr
                            and (trk_m2.n_cls != 2 or not trk_m2.missing_in_acc)
                            and trk_m2.D_sum < 10
                            and (len(trk_m2.clusters) - len(set(c.mean_z for c in trk_m2.clusters))) == 0
                        )

                        # if one is not good while the other is, print details
                        if not (is_good_m1 and is_good_m2):
                            should_skip = False
                            break
                if should_skip:
                    continue

            # count printed tracks only
            trk_num_m1 += len(tracks_m1)
            trk_num_m2 += len(tracks_m2)
            
            f.write(f"{'='*60}\n")
            f.write(f"EVENT {i}\n")
            f.write(f"{'='*60}\n\n")
            
            # Print tracks info side by side
            f.write(f"{f'M1 METHOD - mult_x0 = {len(tracks_m1)}':<45} | {f'M2 METHOD - mult_x0_m2 = {len(tracks_m2)}':<45}\n")
            f.write(f"{'-'*45} | {'-'*45}\n\n")
            
            # Set max number of tracks
            max_tracks = max(len(tracks_m1), len(tracks_m2))
            
            for track_num in range(max_tracks):
                # Prepare string m1
                m1_str = ""
                if track_num < len(tracks_m1):
                    trk = tracks_m1[track_num]
                    
                    # Fill histos for m1
                    n_cls = trk.n_cls
                    h_ncls_m1.Fill(n_cls)
                    Dsum = trk.D_sum
                    h_dsum_vs_ncls_m1.Fill(Dsum, n_cls)
                    
                    for c in trk.clusters:
                        h_resx_m1.Fill(c.res_x)
                        h_resy_m1.Fill(c.res_y)
                    
                    # Issues detection
                    issue_meanx = any(c.mean_x == -999 for c in trk.clusters)
                    same_z_count = len(trk.clusters) - len(
                        set(c.mean_z for c in trk.clusters)
                    )
                    
                    if issue_meanx:
                        counters_m1["bad_meanx"] += 1
                    if not trk.hit_tr:
                        counters_m1["no_TR_hit"] += 1
                    if same_z_count >= 2:
                        counters_m1["same_z_tracks"] += 1

                    # check if good track
                    is_good_track_m1 = (
                        n_cls < 4
                        and not issue_meanx
                        and trk.hit_tr
                        and (n_cls != 2 or not trk.missing_in_acc)
                        and Dsum < 10
                        and same_z_count == 0
                    )
                    
                    # Build string m1
                    cluster_ids = [str(c.cluster_idx) for c in trk.clusters]
                    if is_good_track_m1:
                        m1_str += "### GOOD TRACK ###\n"
                        good_tracks_m1 += 1
                    m1_str += f"* Track {trk.track_idx}: trk_idx={trk.track_idx}\n"
                    m1_str += f"x0={trk.x0:.5f}, y0={trk.y0:.5f}\n"
                    m1_str += f"theta={trk.theta:.5f}, phi={trk.phi:.5f}\n"
                    m1_str += f"n_cls={trk.n_cls}, Dsum={Dsum:.5f}\n"
                    m1_str += f"hit_TR={int(trk.hit_tr)}, missing_in_acc={int(trk.missing_in_acc)}\n"
                    m1_str += f"clusters used: {', '.join(cluster_ids)}\n"
                    
                    # Issues
                    if issue_meanx or same_z_count > 0 or not trk.hit_tr or trk.missing_in_acc:
                        m1_str += "⚠️ Issues:\n"
                        if issue_meanx:
                            m1_str += "  - mean_x = -999\n"
                        if same_z_count >= 2:
                            m1_str += f"  - ≥2 clusters same Z\n"
                        if not trk.hit_tr:
                            m1_str += "  - no_TR_hit\n"
                        if trk.missing_in_acc:
                            m1_str += "  - missing point in acceptance\n"
                        if same_z_count > 0:
                            m1_str += "  - clusters with same Z\n"
                    
                    # Clusters details
                    for j, c in enumerate(trk.clusters):
                        m1_str += f"- Cluster {j}: cls_idx={c.cluster_idx}\n"
                        m1_str += f"  x={c.mean_x:.3f}, y={c.mean_y:.3f}\n"
                        m1_str += f"  z={c.mean_z:.3f}, size={c.size}\n"
                        m1_str += f"  res_x={c.res_x:.3f}, res_y={c.res_y:.3f}\n"
                    
                    
                else:
                    m1_str = "NO TRACK\n"
                
                # Prepare string m2
                m2_str = ""
                if track_num < len(tracks_m2):
                    trk = tracks_m2[track_num]
                    
                    # Fill histos for m2
                    n_cls = trk.n_cls
                    h_ncls_m2.Fill(n_cls)
                    Dsum = trk.D_sum
                    h_dsum_vs_ncls_m2.Fill(Dsum, n_cls)
                    
                    for c in trk.clusters:
                        h_resx_m2.Fill(c.res_x)
                        h_resy_m2.Fill(c.res_y)
                    
                    # Issues detection
                    issue_meanx = any(c.mean_x == -999 for c in trk.clusters)
                    same_z_count = len(trk.clusters) - len(
                        set(c.mean_z for c in trk.clusters)
                    )
                    
                    if issue_meanx:
                        counters_m2["bad_meanx"] += 1
                    if not trk.hit_tr:
                        counters_m2["no_TR_hit"] += 1
                    if same_z_count >= 2:
                        counters_m2["same_z_tracks"] += 1
                    
                    # check if good track
                    is_good_track_m2 = (
                        n_cls < 4
                        and not issue_meanx
                        and trk.hit_tr
                        and (n_cls != 2 or not trk.missing_in_acc)
                        and Dsum < 10
                        and same_z_count == 0
                    )

                    # Build string M2
                    cluster_ids = [str(c.cluster_idx) for c in trk.clusters]
                    if is_good_track_m2:
                        m2_str += "### GOOD TRACK ###\n"
                        good_tracks_m2 += 1
                    m2_str += f"* Track {trk.track_idx}: trk_idx={trk.track_idx}\n"
                    m2_str += f"x0={trk.x0:.5f}, y0={trk.y0:.5f}\n"
                    m2_str += f"theta={trk.theta:.5f}, phi={trk.phi:.5f}\n"
                    m2_str += f"n_cls={trk.n_cls}, Dsum={Dsum:.5f}\n"
                    m2_str += f"hit_TR={int(trk.hit_tr)}, missing_in_acc={int(trk.missing_in_acc)}\n"
                    m2_str += f"clusters used: {', '.join(cluster_ids)}\n"
                    
                    # Issues
                    if issue_meanx or same_z_count > 0 or not trk.hit_tr or trk.missing_in_acc:
                        m2_str += "⚠️ Issues:\n"
                        if issue_meanx:
                            m2_str += "  - mean_x = -999\n"
                        if same_z_count >= 2:
                            m2_str += f"  - ≥2 clusters same Z\n"
                        if not trk.hit_tr:
                            m2_str += "  - no_TR_hit\n"
                        if trk.missing_in_acc:
                            m2_str += "  - missing point in acceptance\n"
                        if same_z_count > 0:
                            m2_str += "  - clusters with same Z\n"
                    
                    # Clusters details
                    for j, c in enumerate(trk.clusters):
                        m2_str += f"- Cluster {j}: cls_idx={c.cluster_idx}\n"
                        m2_str += f"  x={c.mean_x:.3f}, y={c.mean_y:.3f}\n"
                        m2_str += f"  z={c.mean_z:.3f}, size={c.size}\n"
                        m2_str += f"  res_x={c.res_x:.3f}, res_y={c.res_y:.3f}\n"
                else:
                    m2_str = "NO TRACK\n"
                
                # format output
                m1_lines = m1_str.strip().split('\n')
                m2_lines = m2_str.strip().split('\n')
                max_lines = max(len(m1_lines), len(m2_lines))
                
                m1_lines += [''] * (max_lines - len(m1_lines))
                m2_lines += [''] * (max_lines - len(m2_lines))
                
                for line_m1, line_m2 in zip(m1_lines, m2_lines):
                    f.write(f"{line_m1:<45} | {line_m2:<45}\n")
                
                f.write(f"{'-'*45} | {'-'*45}\n\n")
            
            # get full cluster list for each event            
            f.write("#" * 50 + "\n")
            f.write("# ALL CLUSTERS FOR THIS EVENT\n")
            f.write("#" * 50 + "\n")

            for j, c in enumerate(all_clusters): 
                if c.cluster_idx == -999:
                    continue               
                f.write(
                    f"- Cluster {j}: ID={c.cluster_idx}\n"
                    f"  x={c.mean_x:.3f}, y={c.mean_y:.3f}, z={c.mean_z:.3f}, size={c.size}\n"
                )
            
            f.write("\n" + "=" * 60 + "\n\n")

        f.write("\n" + "#" * 60)
        f.write(f"\nTracks number (m1) = {trk_num_m1}")
        f.write(f"\nGood tracks (m1) = {good_tracks_m1}")
        f.write(f"\nTracks number (m2) = {trk_num_m2}")
        f.write(f"\nGood tracks (m2) = {good_tracks_m2}")
        f.write(f"\nGood tracks ratio (m1) = {good_tracks_m1/trk_num_m1*100:.2f}%" if trk_num_m1 > 0 else "\nGood tracks ratio (m1) = 0.00%")
        f.write(f"\nGood tracks ratio (m2) = {good_tracks_m2/trk_num_m2*100:.2f}%" if trk_num_m2 > 0 else "\nGood tracks ratio (m2) = 0.00%")


# ============================================================
# ROOT trees output
# ============================================================
def create_event_ttree(root_file, results):
    """
    Save event multiplicities for m1 and m2 into a Tree.
    """

    root_file.cd()

    tree_ev = ROOT.TTree("EventSummary", "Event summary (track multiplicities)")
    
    event   = np.zeros(1, dtype=np.int32)
    mult_m1 = np.zeros(1, dtype=np.int32)
    mult_m2 = np.zeros(1, dtype=np.int32)

    tree_ev.Branch("event", event, "event/I")
    tree_ev.Branch("mult_m1", mult_m1, "mult_m1/I")
    tree_ev.Branch("mult_m2", mult_m2, "mult_m2/I")

    for i, r in enumerate(results): # save multiplicities for each method
        event[0] = i
        mult_m1[0] = len(r["m1"])
        mult_m2[0] = len(r["m2"])
        tree_ev.Fill()

    tree_ev.Write()
    print("✅ TTree 'EventSummary' written.")


def create_ttree(root_file, tracks, method=""):
    """Create and write a ROOT TTree with selected event-level info."""

    root_file.cd()
    tree_out = ROOT.TTree(f"SelectedEvents({method})", f"Selected Events after filtering({method})")

    x0_val = np.zeros(1, dtype=np.float32)
    y0_val = np.zeros(1, dtype=np.float32)
    theta_val = np.zeros(1, dtype=np.float32)
    phi_val = np.zeros(1, dtype=np.float32)
    n_cls = np.zeros(1, dtype=np.int32)
    hit_tr = np.zeros(1, dtype=np.int32)
    in_acc = np.zeros(1, dtype=np.int32)
    event_idx = np.zeros(1, dtype=np.int32)

    tree_out.Branch(f"x0{method}", x0_val, "x0/F")
    tree_out.Branch(f"y0{method}", y0_val, "y0/F")
    tree_out.Branch(f"theta{method}", theta_val, "theta/F")
    tree_out.Branch(f"phi{method}", phi_val, "phi/F")
    tree_out.Branch("n_cls", n_cls, "n_cls/I")
    tree_out.Branch("track_hit_TR", hit_tr, "track_hit_TR/I")
    tree_out.Branch("missing_in_acceptance", in_acc, "missing_in_acceptance/I")
    tree_out.Branch(f"event_idx", event_idx, "event_idx/I")
    
    # One TTree entry per track
    for trk in tracks:
        x0_val[0] = trk.x0
        y0_val[0] = trk.y0
        theta_val[0] = trk.theta
        phi_val[0] = trk.phi
        n_cls[0] = trk.n_cls
        hit_tr[0] = int(trk.hit_tr)
        in_acc[0] = int(trk.missing_in_acc)
        event_idx[0] = trk.event
        tree_out.Fill()

    tree_out.Write()
    print("✅ TTree 'SelectedEvents' written.")


# ============================================================
# Summary histogram creation
# ============================================================
def make_summary_hist(tracks, output_file, output_dir, method=""):

    print("\n📊 Building summary histogram...")

    h_summary = ROOT.TH1F("h_summary", f"Event summary counts({method});;Counts", 9, 0.5, 9.5)
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
    c_summary = ROOT.TCanvas("c_summary", f"Summary{method}", 1000, 600)
    c_summary.SetBottomMargin(0.28)
    h_summary.SetFillColor(ROOT.kAzure - 4)
    h_summary.Draw("hist text0")
    pdf_path = os.path.join(output_dir, f"summary_counts{method}.pdf")
    c_summary.SaveAs(pdf_path)
    c_summary.Write()
    print(f"\n💾 Saved summary canvas to: {pdf_path}")
    print("📂 Stored inside ROOT file.")
    print("✅ Summary histogram creation complete.\n")


# ============================================================
# Main orchestrator
# ============================================================
def extract_selected_info(input_file, output_dir, save_tree=False, masks_to_apply = None, multiplicity_config = None, method="", debug_all=False):
    load_geometry()

    if masks_to_apply is None:
        masks_to_apply = {
            "trig": False,
            "trig_count": False,
            "x0_multiplicity": False,
            "x0_m2_multiplicity": False,
        }
    if multiplicity_config is None:
        multiplicity_config = {
            "x0_count": None, 
            "x0_m2_count": None
        }

    arrays, _ = load_and_select_events(
        input_file,
        masks_to_apply=masks_to_apply,
        multiplicity_config=multiplicity_config,
    )

    # process analyze_event once and store into results
    results = [] # results is a list of dicts
    for evt in arrays:
        tracks_m1, clusters = analyze_event(evt, method="")
        tracks_m2, _ = analyze_event(evt, method="_m2") # clusters are the same for both m1 and m2

        results.append({
            "m1": tracks_m1,
            "m2": tracks_m2,
            "clusters": clusters
        })

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_file))[0]
    
    # separe methods 
    # if method == "both" -> execute both m1 and m2 
    if(method == "both"):
        if(debug_all):
            txt_output = os.path.join(output_dir, f"{base}_selected_m1_vs_m2_all.txt")
            root_output = os.path.join(output_dir, f"{base}_selected_m1_vs_m2_all.root")
        else: 
            txt_output = os.path.join(output_dir, f"{base}_selected_m1_vs_m2.txt")
            root_output = os.path.join(output_dir, f"{base}_selected_m1_vs_m2.root")
    else:
        txt_output = os.path.join(output_dir, f"{base}_selected{method}.txt")
        root_output = os.path.join(output_dir, f"{base}_selected{method}.root")

    # Histograms
    dsum_edges = np.concatenate(
    [
        np.linspace(0, 50, 20, endpoint=False),
        np.linspace(50, 200, 15, endpoint=False),
        np.linspace(200, 1000, 10, endpoint=False),
        np.linspace(1000, 2000, 6),
    ]
    ).astype(np.float64)
        
    if(method == "" or method == "both"):
        h_ncls_m1 = ROOT.TH1F("h_ncls_m1", "Number of Clusters per Event (M1);N_{cls};Entries", 16, -0.5, 15.5)
        h_samez_m1 = ROOT.TH1F("h_samez_m1", "Clusters with same Z per Event (M1);Count;Entries", 16, -0.5, 15.5)
        h_resx_m1 = ROOT.TH1F("h_resx_m1", "Residual X (M1);res_x;Entries", 100, -0.5, 0.5)
        h_resy_m1 = ROOT.TH1F("h_resy_m1", "Residual Y (M1);res_y;Entries", 100, -0.5, 0.5)

        h_dsum_vs_ncls_m1 = ROOT.TH2F(
            "h_dsum_vs_ncls_m1",
            "Dsum vs Ncls (M1);Dsum;N_{cls}",
            len(dsum_edges) - 1,
            dsum_edges,
            16,
            -0.5,
            15.5,
        )

        counters_m1 = {"bad_meanx": 0, "no_TR_hit": 0, "same_z_tracks": 0}
        all_tracks_m1 = []

        for i, rslt in enumerate(results):
            for tr in rslt["m1"]:
                tr.event = i # add event index to track object
            remove_duplicate_tracks(rslt["m1"], event_idx=i)
            all_tracks_m1.extend(rslt["m1"])


    if(method == "_m2" or method == "both"):
        h_ncls_m2 = ROOT.TH1F("h_ncls_m2", "Number of Clusters per Event (M2);N_{cls};Entries", 16, -0.5, 15.5)
        h_samez_m2 = ROOT.TH1F("h_samez_m2", "Clusters with same Z per Event (M2);Count;Entries", 16, -0.5, 15.5)
        h_resx_m2 = ROOT.TH1F("h_resx_m2", "Residual X (M2);res_x;Entries", 100, -0.5, 0.5)
        h_resy_m2 = ROOT.TH1F("h_resy_m2", "Residual Y (M2);res_y;Entries", 100, -0.5, 0.5)

        h_dsum_vs_ncls_m2 = ROOT.TH2F(
            "h_dsum_vs_ncls_m2",
            "Dsum vs Ncls (M2);Dsum;N_{cls}",
            len(dsum_edges) - 1,
            dsum_edges,
            16,
            -0.5,
            15.5,
        )

        counters_m2 = {"bad_meanx": 0, "no_TR_hit": 0, "same_z_tracks": 0}
        all_tracks_m2 = []

        for i, rslt in enumerate(results):
            for tr in rslt["m2"]:
                tr.event = i # add event index to track object
            remove_duplicate_tracks(rslt["m2"], event_idx=i)
            all_tracks_m2.extend(rslt["m2"])          

    if method == "both":
        compare_txt(
            txt_output,
            results,
            h_ncls_m1, h_ncls_m2,
            h_samez_m1, h_samez_m2,
            h_resx_m1, h_resx_m2,
            h_resy_m1, h_resy_m2,
            h_dsum_vs_ncls_m1, h_dsum_vs_ncls_m2,
            counters_m1, counters_m2,
            debug_all,
        )

        h_ncls_m1.Write()
        h_ncls_m2.Write()
        h_samez_m1.Write()
        h_samez_m2.Write()
        h_resx_m1.Write()
        h_resx_m2.Write()
        h_resy_m1.Write()
        h_resy_m2.Write()
        h_dsum_vs_ncls_m1.Write()
        h_dsum_vs_ncls_m2.Write()
    
    elif method == "":
        write_txt_dump(
            txt_output,
            results,
            h_ncls_m1,
            h_samez_m1,
            h_resx_m1,
            h_resy_m1,
            h_dsum_vs_ncls_m1,
            counters_m1,
            method="",
        )
        h_ncls_m1.Write()
        h_samez_m1.Write()
        h_resx_m1.Write()
        h_resy_m1.Write()
        h_dsum_vs_ncls_m1.Write()


    elif method == "_m2":
        write_txt_dump(
            txt_output,
            results,
            h_ncls_m2,
            h_samez_m2,
            h_resx_m2,
            h_resy_m2,
            h_dsum_vs_ncls_m2,
            counters_m2,
            method="_m2",
        )
        h_ncls_m2.Write()
        h_samez_m2.Write()
        h_resx_m2.Write()
        h_resy_m2.Write()
        h_dsum_vs_ncls_m2.Write()


    root_file = ROOT.TFile(root_output, "RECREATE")

    # create tree for selected method
    if save_tree:
        create_event_ttree(root_file, results)
        if method == "both":
            create_ttree(root_file, all_tracks_m1, method="") 
            create_ttree(root_file, all_tracks_m2, method="_m2")
        elif method == "":
            create_ttree(root_file, all_tracks_m1, method) 
        elif method == "_m2":
            create_ttree(root_file, all_tracks_m2, method)    

    # make summary hist for selected method
    if method == "both":
        make_summary_hist(all_tracks_m1, root_file, output_dir, method="")
        make_summary_hist(all_tracks_m2, root_file, output_dir, method="_m2")
    elif method == "":
        make_summary_hist(all_tracks_m1, root_file, output_dir, method)
    elif method == "_m2":
        make_summary_hist(all_tracks_m2, root_file, output_dir, method)

    root_file.Close()

    print("✅ Done.\n")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":

    # select masks to apply
    masks_to_apply = {
        "trig": False,
        "trig_count": False,
        "x0_multiplicity": False,
        "x0_m2_multiplicity": False,
    }

    multiplicity_config = {
        "x0_count": None,
        "x0_m2_count": None,
    }

    parser = argparse.ArgumentParser(description="Extract and analyze L2 events.")
    parser.add_argument("--input", required=True, help="Input file.")
    parser.add_argument("--output-dir", default="./output", help="Output directory. Default is ./output")
    parser.add_argument("--save-tree", action="store_true", help="Build TTree with tracks info.")
    parser.add_argument("--debug-all", action="store_true", help="Print all tracks into txt dump. Default: print differences between m1 and m2 methods.")

    
    method_group = parser.add_mutually_exclusive_group() # either choose m1 or m2, no argument for both
    method_group.add_argument("--use-m2", action="store_true", help="Check for good tracks using only m2 method (default: use both).")
    method_group.add_argument("--use-m1", action="store_true", help="Check for good tracks using only m1 method (default: use both).")
    args = parser.parse_args()

    # choose applied method
    if (not args.use_m1) and (not args.use_m2):
        print("\n --- Running both methods (default option) --- ")
        extract_selected_info(args.input, args.output_dir, args.save_tree, masks_to_apply, multiplicity_config, method="both", debug_all=args.debug_all)
    elif args.use_m1:
        print(f"\n --- Running m1 method ---")
        extract_selected_info(args.input, args.output_dir, args.save_tree, masks_to_apply, multiplicity_config, method="", debug_all=args.debug_all)
    elif args.use_m2:
        print(f"\n --- Running m2 method ---")
        extract_selected_info(args.input, args.output_dir, args.save_tree, masks_to_apply, multiplicity_config, method="_m2", debug_all=args.debug_all)



    

