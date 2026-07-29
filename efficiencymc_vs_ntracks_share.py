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
    process_efficiency_event,
    check_angle_mismatch,
    read_data_from_file,
    write_counters,
    print_counters,
    compute_gen_mask, 
    is_track_reconstructed
)


def calculate_efficiency(fname, calculate_m1=False, theta_angle_threshold=5.0, phi_angle_threshold=5.0):
    """
    Calculate efficiency for M1 and M2 for a single file.
    
    Args:
        fname: ROOT file path
        gen_mask: pre-computed gen_mask array
        calculate_m1: whether to calculate M1 efficiency (only for first file)
    
    Returns:
        tuple: (radius, eff_m1, eff_m2, counters_m1, counters_m2)
    """
    sh_str = "not_shared" if "not_shared" in fname else "shared"

    tot_fake_m1 = 0
    tot_fake_m2 = 0
    # Open file and read radius
    f = uproot.open(fname)
    p = f.get("radius")
    if not p:
        raise RuntimeError("TParameter 'radius' not found in file")
    r_val = p.value
    
    # Read basic data using utility function
    file_data = read_data_from_file(f)

    n_events = len(file_data["mult_data"]["event_idx"])
    n_tracks_per_event = len(file_data["gen_data"]["theta"][0])
    print(f"\n{'='*100}\n")
    print(f"Processing file: {fname}, tracks per event: {n_tracks_per_event}\n")
    print(f"Number of events: {n_events}")
    print(f"Tracks per event: {n_tracks_per_event}")

    eff_info, rec_info, eff_counters = initialize_counters(n_events, n_tracks_per_event, file_data)    

    # Process all events
    for ev_idx in range(n_events):
        # Check if generated event is good (denominator)
        #is_gen_good = gen_mask[ev_idx] if ev_idx < len(gen_mask) else False

        for ntr in range(n_tracks_per_event):
            mc_cls_n = 0
            mc_cls_n = sum(1 for x in file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx] if x == ntr)
            if mc_cls_n == 2:
                eff_counters["mc"]["2cl_gen"] += 1
            elif mc_cls_n == 3:
                eff_counters["mc"]["3cl_gen"] += 1
            else:
                print(f"ev_ {ev_idx}: WARNING: {mc_cls_n}")

        eff_counters["mc"]["tot_gen"] += n_tracks_per_event
        
        # Get multiplicities
        mult_m1 = file_data['mult_data']['m1'][ev_idx]
        mult_m2 = file_data['mult_data']['m2'][ev_idx]
        gen_trk_idx = file_data["gen_data"]["trk_id"][ev_idx]

        results = {
            "efficiency": {
                "tot": 0,
                "tot_err": 0,
                "3cl": 0,
                "3cl_err": 0,
                "2cl": 0,
                "2cl_err": 0,
            },
            "fake": {
                "tot": 0,
                "tot_err": 0,
                "3cl": 0,
                "3cl_err": 0,
                "2cl": 0,
                "2cl_err": 0,
            }
                    
        }
        
        # Process M1 (only if requested - first file only)
        #reco_m1_idx = np.where(ev_idx == file_data["rec_data"]["m1"]["event_idx"])[0]
        #if reco_m1_idx.tolist():
        #    good_for_eff_m1, reco_to_gen_m1, n_fake_m1, fake_id_m1, diff_theta_m1, diff_phi_m1, gen_theta, gen_phi = process_efficiency_event(
        #        n_tracks_per_event,
        #        ev_idx,
        #        reco_m1_idx,
        #        gen_trk_idx,
        #        mult_m1,
        #        file_data["rec_data"]["m1"]["cls"]["x"][reco_m1_idx],
        #        file_data["rec_data"]["m1"]["cls"]["y"][reco_m1_idx],
        #        file_data["rec_data"]["m1"]["cls"]["z"][reco_m1_idx],
        #        file_data["gen_data"]["cls"]["x"][ev_idx],
        #        file_data["gen_data"]["cls"]["y"][ev_idx],
        #        file_data["gen_data"]["cls"]["layer"][ev_idx],
        #        file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx],
        #        file_data["rec_data"]["m1"]["theta"][reco_m1_idx],
        #        file_data["rec_data"]["m1"]["phi"][reco_m1_idx],
        #        file_data["gen_data"]["theta"][ev_idx],
        #        file_data["gen_data"]["phi"][ev_idx],
        #        file_data["rec_data"]["m1"]["d_sum"][reco_m1_idx],
        #        eff_info,
        #        rec_info,
        #        eff_counters,
        #        theta_angle_threshold,
        #        phi_angle_threshold,
        #        return_good_flag=True,
        #        return_delta=False,
        #        method="m1",
        #        dump=None
        #    )
        #    #print(f"{eff_counters['m1']['eff']['tot_rec_gen']} / {eff_counters['mc']['tot_gen']} good tracks reconstructed by M1 in event {ev_idx} with {n_tracks_per_event} tracks")
        #    tot_fake_m1 += n_fake_m1
        
        # Process M2 (always)
        reco_m2_idx = np.where(ev_idx == file_data["rec_data"]["m2"]["event_idx"])[0]
        if reco_m2_idx.tolist():
        
            good_for_eff_m2, reco_to_gen_m2, n_fake_m2, fake_id_m2, diff_theta_m2, diff_phi_m2, gen_theta, gen_phi = process_efficiency_event(
                n_tracks_per_event,
                ev_idx,
                reco_m2_idx,
                gen_trk_idx,
                mult_m2,
                file_data["rec_data"]["m2"]["cls"]["x"][reco_m2_idx],
                file_data["rec_data"]["m2"]["cls"]["y"][reco_m2_idx],
                file_data["rec_data"]["m2"]["cls"]["z"][reco_m2_idx],
                file_data["gen_data"]["cls"]["x"][ev_idx],
                file_data["gen_data"]["cls"]["y"][ev_idx],
                file_data["gen_data"]["cls"]["layer"][ev_idx],
                file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx],
                file_data["rec_data"]["m2"]["theta"][reco_m2_idx],
                file_data["rec_data"]["m2"]["phi"][reco_m2_idx],
                file_data["gen_data"]["theta"][ev_idx],
                file_data["gen_data"]["phi"][ev_idx],
                file_data["rec_data"]["m2"]["d_sum"][reco_m2_idx],
                eff_info,
                rec_info,
                eff_counters,
                theta_angle_threshold,
                phi_angle_threshold,
                return_good_flag=True,
                return_delta=False,
                method="m2",
                dump=None
            )

            tot_fake_m2 += n_fake_m2

    #total_mc_tracks = n_events * n_tracks_per_event

    # Conta quante sono state ricostruite per ogni metodo
    #n_reco_m1 = sum(eff_info["m1"][ev_idx][trk_idx] 
    #                for ev_idx in range(n_events) 
    #                for trk_idx in range(n_tracks_per_event))
    #
    #n_reco_m2 = sum(eff_info["m2"][ev_idx][trk_idx] 
    #                for ev_idx in range(n_events) 
    #                for trk_idx in range(n_tracks_per_event))
            
    # Calculate efficiencies
    results["efficiency"]["tot"] = ( eff_counters["m2"]["eff"]["tot_rec_gen"] / eff_counters["mc"]["tot_gen"] if eff_counters["mc"]["tot_gen"] > 0 else 0)
    results["efficiency"]["3cl"] = ( eff_counters["m2"]["eff"]["3cl_rec_gen"] / eff_counters["mc"]["3cl_gen"] if eff_counters["mc"]["3cl_gen"] > 0 else 0)
    results["efficiency"]["2cl"] = ( eff_counters["m2"]["eff"]["2cl_rec_gen"] / eff_counters["mc"]["2cl_gen"] if eff_counters["mc"]["2cl_gen"] > 0 else 0)
    results["efficiency"]["tot_err"] = np.sqrt(results["efficiency"]["tot"] * (1 - results["efficiency"]["tot"]) / eff_counters["mc"]["tot_gen"]) if eff_counters["mc"]["tot_gen"] > 0 else 0
    results["efficiency"]["3cl_err"] = np.sqrt(results["efficiency"]["3cl"] * (1 - results["efficiency"]["3cl"]) / eff_counters["mc"]["3cl_gen"]) if eff_counters["mc"]["3cl_gen"] > 0 else 0
    results["efficiency"]["2cl_err"] = np.sqrt(results["efficiency"]["2cl"] * (1 - results["efficiency"]["2cl"]) / eff_counters["mc"]["2cl_gen"]) if eff_counters["mc"]["2cl_gen"] > 0 else 0
    
    results["fake"]["tot"] = ( eff_counters["m2"]["fake"]["tot_fake"] / eff_counters["m2"]["fake"]["tot_rec"] if eff_counters["m2"]["fake"]["tot_rec"] > 0 else 0)
    results["fake"]["3cl"] = ( eff_counters["m2"]["fake"]["3cl_fake"] / eff_counters["m2"]["fake"]["3cl_rec"] if eff_counters["m2"]["fake"]["3cl_rec"] > 0 else 0)
    results["fake"]["2cl"] = ( eff_counters["m2"]["fake"]["2cl_fake"] / eff_counters["m2"]["fake"]["2cl_rec"] if eff_counters["m2"]["fake"]["2cl_rec"] > 0 else 0)
    results["fake"]["tot_err"] = np.sqrt(results["fake"]["tot"] * (1 - results["fake"]["tot"]) / eff_counters["m2"]["fake"]["tot_rec"]) if eff_counters["m2"]["fake"]["tot_rec"] > 0 else 0
    results["fake"]["3cl_err"] = np.sqrt(results["fake"]["3cl"] * (1 - results["fake"]["3cl"]) / eff_counters["m2"]["fake"]["3cl_rec"]) if eff_counters["m2"]["fake"]["3cl_rec"] > 0 else 0
    results["fake"]["2cl_err"] = np.sqrt(results["fake"]["2cl"] * (1 - results["fake"]["2cl"]) / eff_counters["m2"]["fake"]["2cl_rec"]) if eff_counters["m2"]["fake"]["2cl_rec"] > 0 else 0
    
    return r_val, results, n_tracks_per_event


def write_detailed_report(dump, radii, eff_m1, eff_m2_list, counters_m1, counters_m2_list):
    """
    Write a detailed efficiency report.
    """
    dump.write("\n" + "="*80 + "\n")
    dump.write("EFFICIENCY VS RADIUS - DETAILED REPORT\n")
    dump.write("="*80 + "\n\n")
    
    # M1 report (constant across radii)
    dump.write("M1 EFFICIENCY:\n")
    dump.write("-"*40 + "\n")
    if counters_m1:
        write_counters(dump, counters_m1, "m1")
        dump.write(f"\nM1 Efficiency: {counters_m1['reconstructed_good']}/"
                  f"{counters_m1['total_gen_good']} = {100*eff_m1:.2f}%\n")
    dump.write("\n")
    
    # M2 report for each radius
    dump.write("M2 EFFICIENCY VS RADIUS:\n")
    dump.write("-"*40 + "\n")
    
    for r, eff_m2, counters in zip(radii, eff_m2_list, counters_m2_list):
        dump.write("\n" + "="*20 + "\n")
        dump.write(f"Radius = {r:.2f} mm:")
        dump.write("\n" + "="*20 + "\n")
        write_counters(dump, counters, "m2")
        dump.write(f"\nM2 Efficiency: {counters['reconstructed_good']}/"
                  f"{counters['total_gen_good']} = {100*eff_m2:.2f}%\n")
    
    dump.write("\n" + "="*80 + "\n")


def plot_efficiency_vs_ntracks(input_dir, output_dir):
    """
    Main function to calculate and plot efficiency vs number of tracks.
    """
    # Create dump file
    dump_path = os.path.join(output_dir, "efficiency_vs_ntracks_dump.txt")
    with open(dump_path, 'w') as dump:
        
        # Find all ROOT files inside input directory
        files = sorted(glob.glob(os.path.join(input_dir, "*.root")))
        if not files:
            raise RuntimeError(f"No ROOT files found in {input_dir}")
        
        print("\nFiles to be processed:")
        print("-"*60)
        for f in files:
            print(os.path.basename(f))
        print("-"*60 + "\n")
        
        # Calculate gen_mask from first file only (valid for all files)
        #print("Computing gen_mask from first file...")
        #first_file = uproot.open(files[0])
        #tree_gen = first_file["MCtruth"]
        #gen_mask = compute_gen_mask(tree_gen, [])
        #print(f"\nGenerated tracks selected: {np.sum(gen_mask)} out of {len(gen_mask)}")
        
        n_tracks = []
        eff_sh_all_list = []
        eff_not_sh_all_list = []
        err_sh_all_list = []
        err_not_sh_all_list = []
        eff_sh_3cl_list = []
        eff_sh_2cl_list = []
        eff_not_sh_3cl_list = []
        eff_not_sh_2cl_list = []
        err_sh_3cl_list = []
        err_sh_2cl_list = []
        err_not_sh_3cl_list = []
        err_not_sh_2cl_list = []
        fake_sh_all_list = []
        fake_not_sh_all_list = []
        err_fake_sh_all_list = []
        err_fake_not_sh_all_list = []
        fake_sh_3cl_list = []
        fake_sh_2cl_list = []
        fake_not_sh_3cl_list = []
        fake_not_sh_2cl_list = []
        err_fake_sh_3cl_list = []
        err_fake_sh_2cl_list = []
        err_fake_not_sh_3cl_list = []
        err_fake_not_sh_2cl_list = []
        
        # Process all files
        for i, fname in enumerate(files):
            # Calculate M1 only for the first file
            
            r_val, results, n_tracks_per_event = calculate_efficiency(fname)
            
            #radii.append(r_val)

            s_str = None
            if "not_shared" in fname:
                n_tracks.append(n_tracks_per_event)
                s_str = "not_shared"
                eff_not_sh_all_list.append(results["efficiency"]["tot"])
                err_not_sh_all_list.append(results["efficiency"]["tot_err"])
                eff_not_sh_3cl_list.append(results["efficiency"]["3cl"])
                eff_not_sh_2cl_list.append(results["efficiency"]["2cl"])
                err_not_sh_3cl_list.append(results["efficiency"]["3cl_err"])
                err_not_sh_2cl_list.append(results["efficiency"]["2cl_err"])
                fake_not_sh_all_list.append(results["fake"]["tot"])
                err_fake_not_sh_all_list.append(results["fake"]["tot_err"])
                fake_not_sh_3cl_list.append(results["fake"]["3cl"])
                fake_not_sh_2cl_list.append(results["fake"]["2cl"])
                err_fake_not_sh_3cl_list.append(results["fake"]["3cl_err"])
                err_fake_not_sh_2cl_list.append(results["fake"]["2cl_err"])
                
                print(f"M2 Efficiency (3-clusters / not_shared) - Tracks per event: {n_tracks_per_event} = {100*eff_not_sh_3cl_list[-1]:.2f}%")
                print(f"M2 Efficiency (2-clusters / not_shared) - Tracks per event: {n_tracks_per_event} = {100*eff_not_sh_2cl_list[-1]:.2f}%")
                print(f"\nM2 Fake Rate (3-clusters / not_shared) - Tracks per event: {n_tracks_per_event} = {100*fake_not_sh_3cl_list[-1]:.2f}%")
                print(f"M2 Fake Rate (2-clusters / not_shared) - Tracks per event: {n_tracks_per_event} = {100*fake_not_sh_2cl_list[-1]:.2f}%")
            else:
                s_str = "shared"
                eff_sh_all_list.append(results["efficiency"]["tot"])
                err_sh_all_list.append(results["efficiency"]["tot_err"])
                eff_sh_3cl_list.append(results["efficiency"]["3cl"])
                eff_sh_2cl_list.append(results["efficiency"]["2cl"])
                err_sh_3cl_list.append(results["efficiency"]["3cl_err"])
                err_sh_2cl_list.append(results["efficiency"]["2cl_err"])
                fake_sh_all_list.append(results["fake"]["tot"])
                err_fake_sh_all_list.append(results["fake"]["tot_err"])
                fake_sh_3cl_list.append(results["fake"]["3cl"])
                fake_sh_2cl_list.append(results["fake"]["2cl"])
                err_fake_sh_3cl_list.append(results["fake"]["3cl_err"])
                err_fake_sh_2cl_list.append(results["fake"]["2cl_err"])
                print(f"M2 Efficiency (3-clusters / shared) - Tracks per event: {n_tracks_per_event} = {100*eff_sh_3cl_list[-1]:.2f}%")
                print(f"M2 Efficiency (2-clusters / shared) - Tracks per event: {n_tracks_per_event} = {100*eff_sh_2cl_list[-1]:.2f}%")
                print(f"\nM2 Fake Rate (3-clusters / shared) - Tracks per event: {n_tracks_per_event} = {100*fake_sh_3cl_list[-1]:.2f}%")
                print(f"M2 Fake Rate (2-clusters / shared) - Tracks per event: {n_tracks_per_event} = {100*fake_sh_2cl_list[-1]:.2f}%")
            
                #counters_m1 = counters["m1"]
                
            # Print M1 results
            #print(f"\nM1 Efficiency: - Tracks per event: {n_tracks_per_event} = {100*eff_m1_list[-1]:.2f}%")
            #print(f"M1 Efficiency (3-clusters): - Tracks per event: {n_tracks_per_event} = {100*eff_m1_3cl_list[-1]:.2f}%")
            #print(f"M1 Efficiency (2-clusters): - Tracks per event: {n_tracks_per_event} = {100*eff_m1_2cl_list[-1]:.2f}%")
            #print(f"\nM1 Fake Rate: - Tracks per event: {n_tracks_per_event} = {100*fake_m1_list[-1]:.2f}%")
            #print(f"M1 Fake Rate (3-clusters): - Tracks per event: {n_tracks_per_event} = {100*fake_m1_3cl_list[-1]:.2f}%")
            #print(f"M1 Fake Rate (2-clusters): - Tracks per event: {n_tracks_per_event} = {100*fake_m1_2cl_list[-1]:.2f}%\n")

            # Print M2 results
            #print(f"M2 Efficiency - Tracks per event: {n_tracks_per_event} = {100*eff_m2_list[-1]:.2f}%")
            
            #print(f"\nM2 Fake Rate - Tracks per event: {n_tracks_per_event} = {100*fake_m2_list[-1]:.2f}%")
            #print(f"M2 Fake Rate (3-clusters) - Tracks per event: {n_tracks_per_event} = {100*fake_m2_3cl_list[-1]:.2f}%")
            #print(f"M2 Fake Rate (2-clusters) - Tracks per event: {n_tracks_per_event} = {100*fake_m2_2cl_list[-1]:.2f}%")
        
        # Write complete report
        #write_detailed_report(dump, radii, eff_m1, eff_m2_list, 
        #                     counters_m1, counters_m2_list)
    
    #print(f"\nDetailed report saved to: {dump_path}")
    
    # Create plots
    create_efficiency_plots(n_tracks, eff_not_sh_all_list, err_not_sh_all_list, eff_sh_all_list, err_sh_all_list, output_dir, r_val, miny=0.7, maxy=1.0, cl="all")
    create_efficiency_plots(n_tracks, eff_not_sh_3cl_list, err_not_sh_3cl_list, eff_sh_3cl_list, err_sh_3cl_list, output_dir, r_val, miny=0.7, maxy=1.0, cl=3)
    create_efficiency_plots(n_tracks, eff_not_sh_2cl_list, err_not_sh_2cl_list, eff_sh_2cl_list, err_sh_2cl_list, output_dir, r_val, miny=0.7, maxy=1.0, cl=2)
    create_fake_rate_plots(n_tracks, fake_not_sh_all_list, err_fake_not_sh_all_list, fake_sh_all_list, err_fake_sh_all_list, output_dir, r_val, miny=-0.01, maxy=0.2, cl="all")
    create_fake_rate_plots(n_tracks, fake_not_sh_3cl_list, err_fake_not_sh_3cl_list, fake_sh_3cl_list, err_fake_sh_3cl_list, output_dir, r_val, miny=-0.01, maxy=0.045, cl=3)
    create_fake_rate_plots(n_tracks, fake_not_sh_2cl_list, err_fake_not_sh_2cl_list, fake_sh_2cl_list, err_fake_sh_2cl_list, output_dir, r_val, miny=0.0, maxy=0.5, cl=2)



def create_efficiency_plots(n_tracks, eff_not_sh_list, err_not_sh_list, eff_sh_list, err_sh_list, output_dir, radius, miny, maxy, cl=None):
    """
    Create efficiency plots.
    """
    n_points = len(n_tracks)
    #print(f"\neff_not_sh: {eff_not_sh_list} - s_str: {s_str} - cl: {cl}")
    #print(f"eff_sh: {eff_sh_list} - s_str: {s_str} - cl: {cl}")
    #print(f"n_tracks: {n_tracks}\n")

    # Create canvas
    c = ROOT.TCanvas(f"c_eff_vs_ntracks_{cl if cl is not None else 'all'}_cl", f"Efficiency vs Number of Tracks - {cl if cl is not None else 'All'} Clusters", 800, 600)
    
    # M2 graph
    gr_ns = ROOT.TGraphErrors(n_points, np.array(n_tracks, dtype='float64'), 
                        np.array(eff_not_sh_list, dtype='float64'), 
                        np.zeros(n_points), np.array(err_not_sh_list, dtype='float64'))
    gr_ns.SetMarkerStyle(20)
    gr_ns.SetMarkerColor(ROOT.kBlue)
    gr_ns.SetLineColor(ROOT.kBlue)
    gr_ns.SetMaximum(maxy)
    gr_ns.SetMinimum(miny)
    gr_ns.SetTitle(f"Efficiency vs Number of Tracks - radius: {radius} mm - {cl if cl is not None else 'All'} Clusters;Number of Tracks;Efficiency")
    gr_ns.Draw("AP")
    gr_s = ROOT.TGraphErrors(n_points, np.array(n_tracks, dtype='float64'), 
                        np.array(eff_sh_list, dtype='float64'), 
                        np.zeros(n_points), np.array(err_sh_list, dtype='float64'))
    gr_s.SetMarkerStyle(20)
    gr_s.SetMarkerColor(ROOT.kRed)
    gr_s.SetLineColor(ROOT.kRed)
    gr_s.Draw("PSAME")
    
    # Horizontal line for M1 (constant)
    #if eff_m1 is not None:
        #xmin, xmax = min(n_tracks), max(n_tracks)
        #line_m1 = ROOT.TLine(xmin, eff_m1, xmax, eff_m1)
        #line_m1.SetLineColor(ROOT.kBlue)
        #line_m1.SetLineWidth(2)
        #line_m1.SetLineStyle(2)  # Dashed line
        #line_m1.Draw()
        
        # Legend
    leg = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
    leg.AddEntry(gr_ns, "Not shared", "p")
    leg.AddEntry(gr_s, "Shared", "p")
    leg.SetTextSize(0.04)
    leg.SetFillStyle(1001)
    leg.SetFillColor(ROOT.kWhite)
    leg.Draw()
    r_str = f"{radius:.2f}".replace('.', 'p')
    
    # Save plot
    plot_path = os.path.join(output_dir, f"efficiency_vs_ntracks_r_{r_str}_{cl if cl is not None else 'all'}_cl_share.pdf")
    c.SaveAs(plot_path)
    print(f"Plot saved to: {plot_path}")


def create_fake_rate_plots(n_tracks, fake_not_sh_list, err_fake_not_sh_list, fake_sh_list, err_fake_sh_list, output_dir, radius, miny, maxy, cl=None):
    """
    Create fake rate plots.
    """
    n_points = len(n_tracks)
    #print(f"\nfake_not_sh: {fake_not_sh_list} - s_str: {s_str} - cl: {cl}")
    #print(f"fake_sh: {fake_sh_list} - s_str: {s_str} - cl: {cl}")
    #print(f"n_tracks: {n_tracks}\n")
    
    # Create canvas
    c = ROOT.TCanvas(f"c_fake_vs_ntracks_{cl if cl is not None else 'all'}_cl", f"Fake Rate vs Number of Tracks - {cl if cl is not None else 'All'} Clusters", 800, 600)
    
    # M2 graph
    gr_ns_f = ROOT.TGraphErrors(n_points, np.array(n_tracks, dtype='float64'), 
                        np.array(fake_not_sh_list, dtype='float64'), 
                        np.zeros(n_points), np.array(err_fake_not_sh_list, dtype='float64'))
    gr_ns_f.SetMarkerStyle(20)
    gr_ns_f.SetMarkerColor(ROOT.kBlue)
    gr_ns_f.SetLineColor(ROOT.kBlue)
    gr_ns_f.SetMaximum(maxy)
    gr_ns_f.SetMinimum(miny)
    gr_ns_f.SetTitle(f"Fake Rate vs Number of Tracks - radius: {radius} mm - {cl if cl is not None else 'All'} Clusters;Number of Tracks;Fake Rate")
    gr_ns_f.Draw("AP")
    gr_s_f = ROOT.TGraphErrors(n_points, np.array(n_tracks, dtype='float64'), 
                        np.array(fake_sh_list, dtype='float64'), 
                        np.zeros(n_points), np.array(err_fake_sh_list, dtype='float64'))
    gr_s_f.SetMarkerStyle(20)
    gr_s_f.SetMarkerColor(ROOT.kRed)
    gr_s_f.SetLineColor(ROOT.kRed)
    gr_s_f.Draw("PSAME")
    
    # Horizontal line for M1 (constant)
    #if fake_m1 is not None:
    #    xmin, xmax = min(n_tracks), max(n_tracks)
    #    line_m1 = ROOT.TLine(xmin, fake_m1, xmax, fake_m1)
    #    line_m1.SetLineColor(ROOT.kBlue)
    #    line_m1.SetLineWidth(2)
    #    line_m1.SetLineStyle(2)  # Dashed line
    #    line_m1.Draw()
        
    # Legend
    leg = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
    leg.AddEntry(gr_ns_f, "Not shared", "p")
    leg.AddEntry(gr_s_f, "Shared", "p")
    leg.SetTextSize(0.04)
    leg.SetFillStyle(1001)
    leg.SetFillColor(ROOT.kWhite)
    leg.Draw()
    
    r_str = f"{radius:.2f}".replace('.', 'p')
    # Save plot
    plot_path = os.path.join(output_dir, f"fake_rate_vs_ntracks_r_{r_str}_{cl if cl is not None else 'all'}_cl_share.pdf")
    c.SaveAs(plot_path)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate efficiency / fake rate vs ntracks.")
    parser.add_argument("--input-dir", required=True, 
                       help="Directory containing .root files.")
    parser.add_argument("--output-dir", default=None, 
                       help="Define output directory.")
    args = parser.parse_args()
    
    config_file = "config/config_efficiency.yaml"
    with open(config_file, "r") as f_cfg:
        config = yaml.safe_load(f_cfg)
    
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = config["output_dir"]
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    plot_efficiency_vs_ntracks(args.input_dir, output_dir)