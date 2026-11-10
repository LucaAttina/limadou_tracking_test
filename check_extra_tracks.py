import sys
import array
import yaml
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


def init_stats():
    return {
        'total_tracks': 0,
        'n_cls_2': 0,
        'n_cls_3': 0,
        'n_cls_gt3': 0,
        'hit_tr_true': 0,
        'hit_tr_false': 0,
        'missing_in_acc_true': 0,
        'missing_in_acc_false': 0,
        'dsum_lt100': 0,
        'dsum_ge100': 0,
        'passed_all': 0,
        'rejected_by_tr': 0,
        'rejected_by_acc': 0,
        'rejected_by_dsum': 0,
        'rejected_by_same_z': 0,
        'rejected_by_multiple': 0,
        'n_cls_2_hit_tr_true': 0,
        'n_cls_2_hit_tr_false': 0,
        'n_cls_2_missing_acc_false': 0,
        'n_cls_2_missing_acc_true': 0,
        'n_cls_2_good': 0,
        'n_cls_3_hit_tr_true': 0,
        'n_cls_3_hit_tr_false': 0,
        'n_cls_3_dsum_lt100': 0,
        'n_cls_3_dsum_ge100': 0,
        'n_cls_3_good': 0,
    }

def print_stats(stats, method_name):
    """Stampa le statistiche per un metodo."""
    total = stats['total_tracks']
    
    if total == 0:
        print(f"\n⚠️ No tracks for method {method_name}")
        return
    
    print("\n" + "="*70)
    print(f"📊 TRACK SELECTION STATISTICS - {method_name}")
    print("="*70)
    print(f"\nTotal tracks processed: {total}")
    
    print("\n🔴 BY NUMBER OF CLUSTERS:")
    print(f"  • n_cls = 2:  {stats['n_cls_2']:6d} ({100*stats['n_cls_2']/total:5.1f}%)")
    print(f"  • n_cls = 3:  {stats['n_cls_3']:6d} ({100*stats['n_cls_3']/total:5.1f}%)")
    print(f"  • n_cls > 3:  {stats['n_cls_gt3']:6d} ({100*stats['n_cls_gt3']/total:5.1f}%)")
    
    if stats['n_cls_2'] > 0:
        print("\n🟠 TRACKS WITH n_cls = 2:")
        print(f"  • hit_TR = True:   {stats['n_cls_2_hit_tr_true']:6d} ({100*stats['n_cls_2_hit_tr_true']/stats['n_cls_2']:5.1f}% of n_cls=2)")
        print(f"  • hit_TR = False:  {stats['n_cls_2_hit_tr_false']:6d} ({100*stats['n_cls_2_hit_tr_false']/stats['n_cls_2']:5.1f}% of n_cls=2)")
        print(f"  • missing_in_acc = False: {stats['n_cls_2_missing_acc_false']:6d} ({100*stats['n_cls_2_missing_acc_false']/stats['n_cls_2']:5.1f}% of n_cls=2)")
        print(f"  • missing_in_acc = True:  {stats['n_cls_2_missing_acc_true']:6d} ({100*stats['n_cls_2_missing_acc_true']/stats['n_cls_2']:5.1f}% of n_cls=2)")
    
    if stats['n_cls_3'] > 0:
        print("\n🟡 TRACKS WITH n_cls = 3:")
        print(f"  • hit_TR = True:   {stats['n_cls_3_hit_tr_true']:6d} ({100*stats['n_cls_3_hit_tr_true']/stats['n_cls_3']:5.1f}% of n_cls=3)")
        print(f"  • hit_TR = False:  {stats['n_cls_3_hit_tr_false']:6d} ({100*stats['n_cls_3_hit_tr_false']/stats['n_cls_3']:5.1f}% of n_cls=3)")
        print(f"  • D_sum < 100:     {stats['n_cls_3_dsum_lt100']:6d} ({100*stats['n_cls_3_dsum_lt100']/stats['n_cls_3']:5.1f}% of n_cls=3)")
        print(f"  • D_sum >= 100:    {stats['n_cls_3_dsum_ge100']:6d} ({100*stats['n_cls_3_dsum_ge100']/stats['n_cls_3']:5.1f}% of n_cls=3)")
    
    print("\n🟢 REJECTION BREAKDOWN:")
    print(f"  • Rejected by hit_TR = False:    {stats['rejected_by_tr']:6d} ({100*stats['rejected_by_tr']/total:5.1f}%)")
    print(f"  • Rejected by missing_in_acc:    {stats['rejected_by_acc']:6d} ({100*stats['rejected_by_acc']/total:5.1f}%)")
    print(f"  • Rejected by D_sum >= 100:      {stats['rejected_by_dsum']:6d} ({100*stats['rejected_by_dsum']/total:5.1f}%)")
    print(f"  • Rejected by same_z:            {stats['rejected_by_same_z']:6d} ({100*stats['rejected_by_same_z']/total:5.1f}%)")
    print(f"  • Rejected by multiple reasons:  {stats['rejected_by_multiple']:6d} ({100*stats['rejected_by_multiple']/total:5.1f}%)")
    
    print("\n✅ FINAL RESULT:")
    print(f"  • Passed all cuts:  {stats['passed_all']:6d} ({100*stats['passed_all']/total:5.1f}%)")
    if stats['n_cls_2'] > 0:
        print(f"  • n_cls=2 good:     {stats['n_cls_2_good']:6d} ({100*stats['n_cls_2_good']/stats['n_cls_2']:5.1f}% of n_cls=2)")
    if stats['n_cls_3'] > 0:
        print(f"  • n_cls=3 good:     {stats['n_cls_3_good']:6d} ({100*stats['n_cls_3_good']/stats['n_cls_3']:5.1f}% of n_cls=3)")
    print("="*70)

ROOT.gStyle.SetOptStat(0)

from utils import safe_first, load_and_select_events, analyze_event, get_all_clusters, remove_duplicate_tracks, is_good_track

'''
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

                # conditions for good tracks
                is_good_tr, issue_meanx, same_z_count = is_good_track(trk)

                if issue_meanx:
                    counters["bad_meanx"] += 1
                if not trk.hit_tr:
                    counters["no_TR_hit"] += 1
                if same_z_count >= 2:
                    counters["same_z_tracks"] += 1

                # --- Write track header ---
                cluster_ids = [str(c.cluster_idx) for c in trk.clusters]
                if is_good_tr:
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
                        f.write("    - missing point in acceptance\n")
                    if same_z_count > 0:
                        f.write("    - clusters with same Z\n")
                    if Dsum > 10:
                        f.write("    - Dsum > 10\n")

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
    '''


# ============================================================
# Text output and histogram filling (comparison)
# ============================================================
def compare_txt(
    txt_output,
    results,
    h_ncls_m1, h_ncls_m2,
    h_samez_m1, h_samez_m2,
    h_resx_m1, h_resx_m2,
    h_resy_m1, h_resy_m2,
    h_dsum_vs_ncls_m1, h_dsum_vs_ncls_m2,
    counters_hough, counters_comb,
    debug_all=False,
):
    """Write event details into a TXT file comparing Hough and Comb side by side.
       If debug_all is False, skip events with same multiplicity and good tracks in both methods."""
    print(f"📝 Writing detailed event dump (Hough vs Comb) to {txt_output}")

    trk_num_hough = 0
    trk_num_comb = 0
    good_tracks_hough = 0
    good_tracks_comb = 0
    
    with open(txt_output, "w") as f:
        for i, evt in enumerate(results):

            all_clusters = evt["clusters"]        
            tracks_hough = evt["hough"]     
            tracks_comb = evt["comb"]
            
            # Skip events with no tracks
            if len(tracks_hough) == 0 and len(tracks_comb) == 0:
                continue

            # check if we should skip this event (if not in debug_all mode)
            if not debug_all:
                should_skip = True
                # condition 1: different multiplicity
                if len(tracks_hough) != len(tracks_comb):
                    should_skip = False
                # condition 2: difference in good track
                else:
                    for track_num in range(len(tracks_hough)):
                        if track_num >= len(tracks_comb):
                            continue

                        trk_hough = tracks_hough[track_num]
                        trk_comb = tracks_comb[track_num]

                        # check if tracks are good
                        is_good_hough , _ , _  = is_good_track(trk_hough)
                        is_good_comb , _ , _ = is_good_track(trk_comb)

                        # if one is not good while the other is, print details
                        if not (is_good_hough and is_good_comb):
                            should_skip = False
                            break
                if should_skip:
                    continue

            # count printed tracks only
            trk_num_hough += len(tracks_hough)
            trk_num_comb += len(tracks_comb)
            
            f.write(f"{'='*60}\n")
            f.write(f"EVENT {i}\n")
            f.write(f"{'='*60}\n\n")
            
            # Print tracks info side by side
            f.write(f"{f'HOUGH METHOD - mult_tracks = {len(tracks_hough)}':<45} | {f'COMB METHOD - mult_tracks = {len(tracks_comb)}':<45}\n")
            f.write(f"{'-'*45} | {'-'*45}\n\n")
            
            # Set max number of tracks
            max_tracks = max(len(tracks_hough), len(tracks_comb))
            
            for track_num in range(max_tracks):
                # Prepare string hough
                hough_str = ""
                if track_num < len(tracks_hough):
                    trk = tracks_hough[track_num]
                    
                    # Fill histos for hough
                    n_cls = trk.n_cls
                    h_ncls_m1.Fill(n_cls)
                    Dsum = trk.D_sum
                    h_dsum_vs_ncls_m1.Fill(Dsum, n_cls)
                    
                    for c in trk.clusters:
                        h_resx_m1.Fill(c.res_x)
                        h_resy_m1.Fill(c.res_y)
                    
                    # Issues detection
                    #issue_meanx = any(c.mean_x == -999 for c in trk.clusters)
                    #same_z_count = len(trk.clusters) - len(
                    #    set(c.mean_z for c in trk.clusters)
                    #)

                    # check if good track
                    is_good_track_hough, issue_meanx, same_z_count = is_good_track(trk)

                    if issue_meanx:
                        counters_hough["bad_meanx"] += 1
                    if not trk.hit_tr:
                        counters_hough["no_TR_hit"] += 1
                    if same_z_count >= 2:
                        counters_hough["same_z_tracks"] += 1

                    # Build string m1
                    cluster_ids = [str(c.cluster_idx) for c in trk.clusters]
                    if is_good_track_hough:
                        hough_str += "### GOOD TRACK ###\n"
                        good_tracks_hough += 1
                    hough_str += f"* Track {trk.track_idx}: trk_idx={trk.track_idx}\n"
                    hough_str += f"x0={trk.x0:.5f}, y0={trk.y0:.5f}\n"
                    hough_str += f"theta={trk.theta:.5f}, phi={trk.phi:.5f}\n"
                    hough_str += f"n_cls={trk.n_cls}, Dsum={Dsum:.5f}\n"
                    hough_str += f"hit_TR={int(trk.hit_tr)}, missing_in_acc={int(trk.missing_in_acc)}\n"
                    hough_str += f"clusters used: {', '.join(cluster_ids)}\n"
                    
                    # Issues
                    if issue_meanx or same_z_count > 0 or not trk.hit_tr or trk.missing_in_acc or Dsum > 10:
                        hough_str += "⚠️ Issues:\n"
                        if issue_meanx:
                            hough_str += "  - mean_x = -999\n"
                        if same_z_count >= 2:
                            hough_str += f"  - ≥2 clusters same Z\n"
                        if not trk.hit_tr:
                            hough_str += "  - no_TR_hit\n"
                        if trk.missing_in_acc:
                            hough_str += "  - missing point in acceptance\n"
                        if same_z_count > 0:
                            hough_str += "  - clusters with same Z\n"
                        if Dsum > 10:
                            hough_str += "  - Dsum > 10\n"
                    
                    # Clusters details
                    for j, c in enumerate(trk.clusters):
                        hough_str += f"- Cluster {j}: cls_idx={c.cluster_idx}\n"
                        hough_str += f"  x={c.mean_x:.3f}, y={c.mean_y:.3f}\n"
                        hough_str += f"  z={c.mean_z:.3f}, size={c.size}\n"
                        hough_str += f"  res_x={c.res_x:.3f}, res_y={c.res_y:.3f}\n"
                    
                    
                else:
                    hough_str = "NO TRACK\n"
                
                # Prepare string comb
                comb_str = ""
                if track_num < len(tracks_comb):
                    trk = tracks_comb[track_num]
                    
                    # Fill histos for comb
                    n_cls = trk.n_cls
                    h_ncls_m2.Fill(n_cls)
                    Dsum = trk.D_sum
                    h_dsum_vs_ncls_m2.Fill(Dsum, n_cls)
                    
                    for c in trk.clusters:
                        h_resx_m2.Fill(c.res_x)
                        h_resy_m2.Fill(c.res_y)
                    
                    # Issues detection
                    #issue_meanx = any(c.mean_x == -999 for c in trk.clusters)
                    #same_z_count = len(trk.clusters) - len(
                    #    set(c.mean_z for c in trk.clusters)
                    #)
                    

                    
                    # check if good track
                    is_good_track_comb, issue_meanx, same_z_count = is_good_track(trk)
                    
                    if issue_meanx:
                        counters_comb["bad_meanx"] += 1
                    if not trk.hit_tr:
                        counters_comb["no_TR_hit"] += 1
                    if same_z_count >= 2:
                        counters_comb["same_z_tracks"] += 1

                    # Build string M2
                    cluster_ids = [str(c.cluster_idx) for c in trk.clusters]
                    if is_good_track_comb:
                        comb_str += "### GOOD TRACK ###\n"
                        good_tracks_comb += 1
                    comb_str += f"* Track {trk.track_idx}: trk_idx={trk.track_idx}\n"
                    comb_str += f"x0={trk.x0:.5f}, y0={trk.y0:.5f}\n"
                    comb_str += f"theta={trk.theta:.5f}, phi={trk.phi:.5f}\n"
                    comb_str += f"n_cls={trk.n_cls}, Dsum={Dsum:.5f}\n"
                    comb_str += f"hit_TR={int(trk.hit_tr)}, missing_in_acc={int(trk.missing_in_acc)}\n"
                    comb_str += f"clusters used: {', '.join(cluster_ids)}\n"
                    
                    # Issues
                    if issue_meanx or same_z_count > 0 or not trk.hit_tr or trk.missing_in_acc:
                        comb_str += "⚠️ Issues:\n"
                        if issue_meanx:
                            comb_str += "  - mean_x = -999\n"
                        if same_z_count >= 2:
                            comb_str += f"  - ≥2 clusters same Z\n"
                        if not trk.hit_tr:
                            comb_str += "  - no_TR_hit\n"
                        if trk.missing_in_acc:
                            comb_str += "  - missing point in acceptance\n"
                        if same_z_count > 0:
                            comb_str += "  - clusters with same Z\n"
                        if Dsum > 10:
                            comb_str += "  - Dsum > 10\n"
                    
                    # Clusters details
                    for j, c in enumerate(trk.clusters):
                        comb_str += f"- Cluster {j}: cls_idx={c.cluster_idx}\n"
                        comb_str += f"  x={c.mean_x:.3f}, y={c.mean_y:.3f}\n"
                        comb_str += f"  z={c.mean_z:.3f}, size={c.size}\n"
                        comb_str += f"  res_x={c.res_x:.3f}, res_y={c.res_y:.3f}\n"
                else:
                    comb_str = "NO TRACK\n"
                
                # format output
                hough_lines = hough_str.strip().split('\n')
                comb_lines = comb_str.strip().split('\n')
                max_lines = max(len(hough_lines), len(comb_lines))
                
                hough_lines += [''] * (max_lines - len(hough_lines))
                comb_lines += [''] * (max_lines - len(comb_lines))
                
                for line_hough, line_comb in zip(hough_lines, comb_lines):
                    f.write(f"{line_hough:<45} | {line_comb:<45}\n")
                
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
        f.write(f"\nTracks number (hough) = {trk_num_hough}")
        f.write(f"\nGood tracks (hough) = {good_tracks_hough}")
        f.write(f"\nTracks number (comb) = {trk_num_comb}")
        f.write(f"\nGood tracks (comb) = {good_tracks_comb}")
        f.write(f"\nGood tracks ratio (hough) = {good_tracks_hough/trk_num_hough*100:.2f}%" if trk_num_hough > 0 else "\nGood tracks ratio (hough) = 0.00%")
        f.write(f"\nGood tracks ratio (comb) = {good_tracks_comb/trk_num_comb*100:.2f}%" if trk_num_comb > 0 else "\nGood tracks ratio (comb) = 0.00%")


# ============================================================
# ROOT trees output
# ============================================================
def create_event_ttree(root_file, results):
    """
    Save event multiplicities for hough and comb into a Tree.
    """

    root_file.cd()

    tree_ev = ROOT.TTree("EventSummary", "Event summary (track multiplicities)")
    
    event   = np.zeros(1, dtype=np.int32)
    mult_hough = np.zeros(1, dtype=np.int32)
    mult_comb  = np.zeros(1, dtype=np.int32)

    tree_ev.Branch("event", event, "event/I")
    tree_ev.Branch("mult_hough", mult_hough, "mult_hough/I")
    tree_ev.Branch("mult_comb", mult_comb, "mult_comb/I")

    for i, r in enumerate(results): # save multiplicities for each method
        event[0] = i
        mult_hough[0] = len(r["hough"])
        mult_comb[0] = len(r["comb"])
        tree_ev.Fill()

    tree_ev.Write()
    print("✅ TTree 'EventSummary' written.")


def create_ttree(root_file, tracks, method=""):
    """Create and write a ROOT TTree with selected event-level info."""

    root_file.cd()
    tree_out = ROOT.TTree(f"SelectedEvents_{method}", f"Selected Events after filtering ({method})")

    x0_val = np.zeros(1, dtype=np.float32)
    y0_val = np.zeros(1, dtype=np.float32)
    theta_val = np.zeros(1, dtype=np.float32)
    phi_val = np.zeros(1, dtype=np.float32)
    n_cls = np.zeros(1, dtype=np.int32)
    hit_tr = np.zeros(1, dtype=np.int32)
    in_acc = np.zeros(1, dtype=np.int32)
    event_idx = np.zeros(1, dtype=np.int32)
    d_sum = np.zeros(1, dtype=np.float32)

    # new: clusters coordinates are now saved
    #cl_x = array.array('f', [0.]*max_clusters)
    #cl_y = array.array('f', [0.]*max_clusters)
    #cl_z = array.array('f', [0.]*max_clusters)
    cl_x = ROOT.vector('float')()
    cl_y = ROOT.vector('float')()
    cl_z = ROOT.vector('float')()
    res_x = ROOT.vector('float')()
    res_y = ROOT.vector('float')()

    tree_out.Branch(f"x0", x0_val, "x0/F")
    tree_out.Branch(f"y0", y0_val, "y0/F")
    tree_out.Branch(f"theta", theta_val, "theta/F")
    tree_out.Branch(f"phi", phi_val, "phi/F")
    tree_out.Branch("n_cls", n_cls, "n_cls/I")
    tree_out.Branch("track_hit_TR", hit_tr, "track_hit_TR/I")
    tree_out.Branch("missing_in_acceptance", in_acc, "missing_in_acceptance/I")
    tree_out.Branch(f"event_idx", event_idx, "event_idx/I")
    tree_out.Branch(f"d_sum", d_sum, "d_sum/F")
    #tree_out.Branch("cl_x", cl_x, f"cl_x[{max_clusters}]/F")
    #tree_out.Branch("cl_y", cl_y, f"cl_y[{max_clusters}]/F")
    #tree_out.Branch("cl_z", cl_z, f"cl_z[{max_clusters}]/F")
    tree_out.Branch("cl_x", cl_x)
    tree_out.Branch("cl_y", cl_y)
    tree_out.Branch("cl_z", cl_z)
    tree_out.Branch("res_x", res_x)
    tree_out.Branch("res_y", res_y)
    
    # One TTree entry per track
    for trk in tracks:
        # only save good tracks
        is_good_tr , _ , _ = is_good_track(trk)
        if not is_good_tr:
            continue

        x0_val[0] = trk.x0
        y0_val[0] = trk.y0
        theta_val[0] = trk.theta
        phi_val[0] = trk.phi
        n_cls[0] = trk.n_cls
        hit_tr[0] = int(trk.hit_tr)
        in_acc[0] = int(trk.missing_in_acc)
        event_idx[0] = trk.event
        d_sum[0] = trk.D_sum

        # clear cluster vectors
        #for i in range(max_clusters):
        #    cl_x[i] = cl_y[i] = cl_z[i] = -999.
        cl_x.clear()
        cl_y.clear()
        cl_z.clear()
        res_x.clear()
        res_y.clear()
        for c in trk.clusters:
            cl_x.push_back(c.mean_x)
            cl_y.push_back(c.mean_y)
            cl_z.push_back(c.mean_z)
            res_x.push_back(c.res_x)
            res_y.push_back(c.res_y)

        tree_out.Fill()

    tree_out.Write()
    print(f"✅ TTree 'SelectedEvents_{method}' written.")


# ============================================================
# Summary histogram creation
# ============================================================
def make_summary_hist(tracks, output_file, output_dir, base, method=""):

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
    pdf_path = os.path.join(output_dir, f"{base}_summary_counts{method}.pdf")
    #c_summary.SaveAs(pdf_path)
    #c_summary.Write()
    print(f"\n💾 Saved summary canvas to: {pdf_path}")
    print("📂 Stored inside ROOT file.")
    print("✅ Summary histogram creation complete.\n")


# ============================================================
# Main orchestrator
# ============================================================
def extract_selected_info(input_file, output_dir, save_tree=False, masks_to_apply = None, multiplicity_config = None, method="", debug_all=False, mc_copy=False):
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

    stats_hough_total = init_stats()
    stats_comb_total = init_stats()

    results = []

    for i, evt in enumerate(arrays):
        tracks_hough, clusters, stats_hough = analyze_event(evt, i, method="hough")
        tracks_comb, _, stats_comb = analyze_event(evt, i, method="comb")

        # Stats for hough
        for key in stats_hough_total:
            stats_hough_total[key] += stats_hough.get(key, 0)

        # Stats for comb
        for key in stats_comb_total:
            stats_comb_total[key] += stats_comb.get(key, 0)

        results.append({
            "hough": tracks_hough,
            "comb": tracks_comb,
            "clusters": clusters
        })

    print_stats(stats_hough_total, "Hough")
    print_stats(stats_comb_total, "Combinatorial")

    # Opzionale: stampa anche un confronto diretto
    print("\n" + "="*70)
    print("📊 COMPARISON: Hough vs Comb")
    print("="*70)
    print(f"{'Metric':<35} {'Hough':>10} {'Comb':>10} {'Diff':>10}")
    print("-"*70)
    print(f"{'Total tracks':<35} {stats_hough_total['total_tracks']:>10} {stats_comb_total['total_tracks']:>10} {stats_comb_total['total_tracks']-stats_hough_total['total_tracks']:>+10}")
    print(f"{'Passed all cuts':<35} {stats_hough_total['passed_all']:>10} {stats_comb_total['passed_all']:>10} {stats_comb_total['passed_all']-stats_hough_total['passed_all']:>+10}")
    print(f"{'n_cls=2 good':<35} {stats_hough_total['n_cls_2_good']:>10} {stats_comb_total['n_cls_2_good']:>10} {stats_comb_total['n_cls_2_good']-stats_hough_total['n_cls_2_good']:>+10}")
    print(f"{'n_cls=3 good':<35} {stats_hough_total['n_cls_3_good']:>10} {stats_comb_total['n_cls_3_good']:>10} {stats_comb_total['n_cls_3_good']-stats_hough_total['n_cls_3_good']:>+10}")
    print("="*70)
        #print(f"Event {evt}: m1 tracks = {len(tracks_m1)}, {tracks_m1} - m2 tracks = {len(tracks_m2)}, {tracks_m2}")
            
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_file))[0]
    
    # separe methods 
    # if method == "both" -> execute both m1 and m2 
    #if(method == "both"):
    if(debug_all):
        txt_output = os.path.join(output_dir, f"{base}_selected_tracks_all.txt")
    else: 
        txt_output = os.path.join(output_dir, f"{base}_selected_tracks_diff.txt")
    root_output = os.path.join(output_dir, f"{base}_selected_tracks.root")
    #else:
    #    txt_output = os.path.join(output_dir, f"{base}_selected{method}.txt")
    #    root_output = os.path.join(output_dir, f"{base}_selected{method}.root")

    root_file = ROOT.TFile(root_output, "RECREATE")
    root_file.cd()

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

        counters_hough = {"bad_meanx": 0, "no_TR_hit": 0, "same_z_tracks": 0}
        all_tracks_hough = []

        for i, rslt in enumerate(results):
            for tr in rslt["hough"]:
                tr.event = i # add event index to track object
            remove_duplicate_tracks(rslt["hough"], event_idx=i)
            all_tracks_hough.extend(rslt["hough"])


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

        counters_comb = {"bad_meanx": 0, "no_TR_hit": 0, "same_z_tracks": 0}
        all_tracks_comb = []

        for i, rslt in enumerate(results):
            for tr in rslt["comb"]:
                tr.event = i # add event index to track object
            remove_duplicate_tracks(rslt["comb"], event_idx=i)
            all_tracks_comb.extend(rslt["comb"])          

    if method == "both":
        compare_txt(
            txt_output,
            results,
            h_ncls_m1, h_ncls_m2,
            h_samez_m1, h_samez_m2,
            h_resx_m1, h_resx_m2,
            h_resy_m1, h_resy_m2,
            h_dsum_vs_ncls_m1, h_dsum_vs_ncls_m2,
            counters_hough, counters_comb,
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

    '''
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
    '''


    # --- Copy MCtruth and radius from input file ---
    if mc_copy:
        input_root = ROOT.TFile(input_file, "READ")
        root_file.cd()
        print(f"\n📂 Copying MCTruth tree from input file to output ROOT file...")
        mc_tree = input_root.Get("MCtruth")
        if mc_tree is None:
            raise RuntimeError(f"No MCtruth tree found in input file {input_file}")

        if mc_tree:
            mc_clone = mc_tree.CloneTree(-1, "fast")  # -1 = all the entries
            mc_clone.Write()
            print("✅ MCTruth tree copied to output ROOT file.")

        radius_val = 0.25

        #radius_param = input_root.Get("radius")
        #if not radius_param:
        #    raise RuntimeError("TParameter 'radius' not found in input file")
        #else:
        #    radius_val = radius_param.GetVal()
        #    root_file.cd()
        #    radius_out = ROOT.TParameter('float')("radius", radius_val)
        #    radius_out.Write()
        #    print(f"✅ Radius {radius_val} saved into output file")

        input_root.Close()


    # create tree for selected method
    if save_tree:
        create_event_ttree(root_file, results)
        if method == "both":
            create_ttree(root_file, all_tracks_hough, method="hough") 
            create_ttree(root_file, all_tracks_comb, method="comb")
        '''
        elif method == "":
            create_ttree(root_file, all_tracks_m1, method) 
        elif method == "_m2":
            create_ttree(root_file, all_tracks_m2, method)    
        '''

    
    # make summary hist for selected method
    if method == "both":
        make_summary_hist(all_tracks_hough, root_file, output_dir, base, method="_hough")
        make_summary_hist(all_tracks_comb, root_file, output_dir, base, method="_comb")
    '''
    elif method == "":
        make_summary_hist(all_tracks_m1, root_file, output_dir, base, method)
    elif method == "_m2":
        make_summary_hist(all_tracks_m2, root_file, output_dir, base, method)
    '''
    

    root_file.Close()

    print("✅ Done.\n")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract and analyze L2 events.")
    parser.add_argument("--input", required=True, help="Input file.")
    parser.add_argument("--output-dir", default="./output", help="Output directory. Default is ./output")
    parser.add_argument("--save-tree", action="store_true", help="Build TTree with tracks info.")
    parser.add_argument("--debug-all", action="store_true", help="Print all tracks into txt dump. Default: print differences between Hough and Comb methods.")
    parser.add_argument("--mc",  action="store_true", help="If true, process MC data.")

    '''
    method_group = parser.add_mutually_exclusive_group() # either choose m1 or m2, no argument for both
    method_group.add_argument("--use-m2", action="store_true", help="Check for good tracks using only m2 method (default: use both).")
    method_group.add_argument("--use-m1", action="store_true", help="Check for good tracks using only m1 method (default: use both).")
    '''
    args = parser.parse_args()

    # read masks from config
    with open("config/config_check_hough.yaml", "r") as f_cfg:
        config = yaml.safe_load(f_cfg)
    masks_to_apply = config["masks_to_apply"]
    multiplicity_config = config["multiplicity_config"]

    # choose applied method
    #if (not args.use_m1) and (not args.use_m2):
    print("\n --- Running both methods --- ")
    extract_selected_info(args.input, args.output_dir, args.save_tree, masks_to_apply, multiplicity_config, method="both", debug_all=args.debug_all, mc_copy=args.mc)
    '''
    elif args.use_m1:
        print(f"\n --- Running m1 method ---")
        extract_selected_info(args.input, args.output_dir, args.save_tree, masks_to_apply, multiplicity_config, method="", debug_all=args.debug_all, mc_copy=args.mc)
    elif args.use_m2:
        print(f"\n --- Running m2 method ---")
        extract_selected_info(args.input, args.output_dir, args.save_tree, masks_to_apply, multiplicity_config, method="_m2", debug_all=args.debug_all, mc_copy=args.mc)
    '''


    

