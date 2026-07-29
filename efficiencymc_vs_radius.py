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
import re
import json

def format_latex(val, err):
    return f"({100*val:.2f} $\\pm$ {100*err:.2f})"


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
    delta_counter_m1 = 0
    delta_counter_m2 = 0
    tot_fake_m1 = 0
    tot_fake_m2 = 0
    # Open file and read radius
    f = uproot.open(fname)
    p = f.get("radius")
    if not p:
        match = re.search(r"_r_0p(\d+)_", fname)

        if match:
            r_val = float("0." + match.group(1))
        else:
            raise RuntimeError("TParameter 'radius' not found in file")
    else:
        r_val = p.value
    print(f"\n{'='*100}\n")
    print(f"Processing file: {fname}, radius: {r_val:.2f} mm\n")
    
    # Read basic data using utility function
    file_data = read_data_from_file(f)

    n_events = len(file_data["mult_data"]["event_idx"])
    n_tracks_per_event = len(file_data["gen_data"]["theta"][0])
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
                "m1": {
                    "tot": 0,
                    "tot_err": 0,
                    "3cl": 0,
                    "3cl_err": 0,
                    "2cl": 0,
                    "2cl_err": 0,
                },
                "m2": {
                    "tot": 0,
                    "tot_err": 0,
                    "3cl": 0,
                    "3cl_err": 0,
                    "2cl": 0,
                    "2cl_err": 0,
                }
            },
            "fake" : {
                "m1": {
                    "tot": 0,
                    "tot_err": 0,
                    "3cl": 0,
                    "3cl_err": 0,
                    "2cl": 0,
                    "2cl_err": 0,
                },
                "m2": {
                    "tot": 0,
                    "tot_err": 0,
                    "3cl": 0,
                    "3cl_err": 0,
                    "2cl": 0,
                    "2cl_err": 0,
                }
            }
        }

        
        # Process M1 (only if requested - first file only)
        if calculate_m1:
            reco_m1_idx = np.where(ev_idx == file_data["rec_data"]["m1"]["event_idx"])[0]
            if reco_m1_idx.tolist():

                good_for_eff_m1, reco_to_gen_m1, n_fake_m1, fake_id_m1, diff_theta_m1, diff_phi_m1, gen_theta, gen_phi = process_efficiency_event(
                    n_tracks_per_event,
                    ev_idx,
                    reco_m1_idx,
                    gen_trk_idx,
                    mult_m1,
                    file_data["rec_data"]["m1"]["cls"]["x"][reco_m1_idx],
                    file_data["rec_data"]["m1"]["cls"]["y"][reco_m1_idx],
                    file_data["rec_data"]["m1"]["cls"]["z"][reco_m1_idx],
                    file_data["gen_data"]["cls"]["x"][ev_idx],
                    file_data["gen_data"]["cls"]["y"][ev_idx],
                    file_data["gen_data"]["cls"]["layer"][ev_idx],
                    file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx],
                    file_data["rec_data"]["m1"]["theta"][reco_m1_idx],
                    file_data["rec_data"]["m1"]["phi"][reco_m1_idx],
                    file_data["gen_data"]["theta"][ev_idx],
                    file_data["gen_data"]["phi"][ev_idx],
                    file_data["rec_data"]["m1"]["d_sum"][reco_m1_idx],
                    eff_info,
                    rec_info,
                    eff_counters,
                    theta_angle_threshold,
                    phi_angle_threshold,
                    return_good_flag=True,
                    return_delta=False,
                    method="m1",
                    dump=None
                )

                tot_fake_m1 += n_fake_m1

        
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

    if calculate_m1: 
        print(f"\nM1 Efficiency: {eff_counters["m1"]["eff"]["tot_rec_gen"]} / {eff_counters["mc"]["tot_gen"]} = {100*eff_counters["m1"]["eff"]["tot_rec_gen"]/eff_counters["mc"]["tot_gen"]:.2f}%")
        print(f"M1 3-cluster Efficiency: {eff_counters["m1"]["eff"]["3cl_rec_gen"]} / {eff_counters["mc"]["3cl_gen"]} = {100*eff_counters["m1"]["eff"]["3cl_rec_gen"]/eff_counters["mc"]["3cl_gen"]:.2f}%")
        print(f"M1 2-cluster Efficiency: {eff_counters["m1"]["eff"]["2cl_rec_gen"]} / {eff_counters["mc"]["2cl_gen"]} = {100*eff_counters["m1"]["eff"]["2cl_rec_gen"]/eff_counters["mc"]["2cl_gen"]:.2f}%")
        print(f"\nM1 Fake Rate: {eff_counters["m1"]["fake"]["tot_fake"]} / {eff_counters["m1"]["fake"]["tot_rec"]} = {100*eff_counters["m1"]["fake"]["tot_fake"]/eff_counters["m1"]["fake"]["tot_rec"]:.2f}%")
        print(f"M1 3-cluster Fake Rate: {eff_counters["m1"]["fake"]["3cl_fake"]} / {eff_counters["m1"]["fake"]["3cl_rec"]} = {100*eff_counters["m1"]["fake"]["3cl_fake"]/eff_counters["m1"]["fake"]["3cl_rec"]:.2f}%")
        print(f"M1 2-cluster Fake Rate: {eff_counters["m1"]["fake"]["2cl_fake"]} / {eff_counters["m1"]["fake"]["2cl_rec"]} = {100*eff_counters["m1"]["fake"]["2cl_fake"]/eff_counters["m1"]["fake"]["2cl_rec"]:.2f}%")
    
    print(f"\nM2 Efficiency - Radius: {r_val:.2f} mm: {eff_counters["m2"]["eff"]["tot_rec_gen"]} / {eff_counters["mc"]["tot_gen"]} = {100*eff_counters["m2"]["eff"]["tot_rec_gen"]/eff_counters["mc"]["tot_gen"]:.2f}%")
    print(f"M2 3-cluster Efficiency - Radius: {r_val:.2f} mm: {eff_counters["m2"]["eff"]["3cl_rec_gen"]} / {eff_counters["mc"]["3cl_gen"]} = {100*eff_counters["m2"]["eff"]["3cl_rec_gen"]/eff_counters["mc"]["3cl_gen"]:.2f}%")
    print(f"M2 2-cluster Efficiency - Radius: {r_val:.2f} mm: {eff_counters["m2"]["eff"]["2cl_rec_gen"]} / {eff_counters["mc"]["2cl_gen"]} = {100*eff_counters["m2"]["eff"]["2cl_rec_gen"]/eff_counters["mc"]["2cl_gen"]:.2f}%")
    print(f"\nM2 Fake Rate - Radius: {r_val:.2f} mm: {eff_counters["m2"]["fake"]["tot_fake"]} / {eff_counters["m2"]["fake"]["tot_rec"]} = {100*eff_counters["m2"]["fake"]["tot_fake"]/eff_counters["m2"]["fake"]["tot_rec"]:.2f}%")
    print(f"M2 3-cluster Fake Rate - Radius: {r_val:.2f} mm: {eff_counters["m2"]["fake"]["3cl_fake"]} / {eff_counters["m2"]["fake"]["3cl_rec"]} = {100*eff_counters["m2"]["fake"]["3cl_fake"]/eff_counters["m2"]["fake"]["3cl_rec"]:.2f}%")
    print(f"M2 2-cluster Fake Rate - Radius: {r_val:.2f} mm: {eff_counters["m2"]["fake"]["2cl_fake"]} / {eff_counters["m2"]["fake"]["2cl_rec"]} = {100*eff_counters["m2"]["fake"]["2cl_fake"]/eff_counters["m2"]["fake"]["2cl_rec"]:.2f}%")
            
    # Calculate efficiencies
    results["efficiency"]["m1"]["tot"] = ( eff_counters["m1"]["eff"]["tot_rec_gen"] / eff_counters["mc"]["tot_gen"] if eff_counters["mc"]["tot_gen"] > 0 else 0)
    results["efficiency"]["m1"]["3cl"] = ( eff_counters["m1"]["eff"]["3cl_rec_gen"] / eff_counters["mc"]["3cl_gen"] if eff_counters["mc"]["3cl_gen"] > 0 else 0)
    results["efficiency"]["m1"]["2cl"] = ( eff_counters["m1"]["eff"]["2cl_rec_gen"] / eff_counters["mc"]["2cl_gen"] if eff_counters["mc"]["2cl_gen"] > 0 else 0)
    results["efficiency"]["m2"]["tot"] = ( eff_counters["m2"]["eff"]["tot_rec_gen"] / eff_counters["mc"]["tot_gen"] if eff_counters["mc"]["tot_gen"] > 0 else 0)
    results["efficiency"]["m2"]["3cl"] = ( eff_counters["m2"]["eff"]["3cl_rec_gen"] / eff_counters["mc"]["3cl_gen"] if eff_counters["mc"]["3cl_gen"] > 0 else 0)
    results["efficiency"]["m2"]["2cl"] = ( eff_counters["m2"]["eff"]["2cl_rec_gen"] / eff_counters["mc"]["2cl_gen"] if eff_counters["mc"]["2cl_gen"] > 0 else 0)    
    results["fake"]["m1"]["tot"] = ( eff_counters["m1"]["fake"]["tot_fake"] / eff_counters["m1"]["fake"]["tot_rec"] if eff_counters["m1"]["fake"]["tot_rec"] > 0 else 0)
    results["fake"]["m1"]["3cl"] = ( eff_counters["m1"]["fake"]["3cl_fake"] / eff_counters["m1"]["fake"]["3cl_rec"] if eff_counters["m1"]["fake"]["3cl_rec"] > 0 else 0)
    results["fake"]["m1"]["2cl"] = ( eff_counters["m1"]["fake"]["2cl_fake"] / eff_counters["m1"]["fake"]["2cl_rec"] if eff_counters["m1"]["fake"]["2cl_rec"] > 0 else 0)
    results["fake"]["m2"]["tot"] = ( eff_counters["m2"]["fake"]["tot_fake"] / eff_counters["m2"]["fake"]["tot_rec"] if eff_counters["m2"]["fake"]["tot_rec"] > 0 else 0)
    results["fake"]["m2"]["3cl"] = ( eff_counters["m2"]["fake"]["3cl_fake"] / eff_counters["m2"]["fake"]["3cl_rec"] if eff_counters["m2"]["fake"]["3cl_rec"] > 0 else 0)
    results["fake"]["m2"]["2cl"] = ( eff_counters["m2"]["fake"]["2cl_fake"] / eff_counters["m2"]["fake"]["2cl_rec"] if eff_counters["m2"]["fake"]["2cl_rec"] > 0 else 0)
    results["efficiency"]["m1"]["tot_err"] = np.sqrt(results["efficiency"]["m1"]["tot"] * (1 - results["efficiency"]["m1"]["tot"]) / eff_counters["mc"]["tot_gen"]) if eff_counters["mc"]["tot_gen"] > 0 else 0
    results["efficiency"]["m1"]["3cl_err"] = np.sqrt(results["efficiency"]["m1"]["3cl"] * (1 - results["efficiency"]["m1"]["3cl"]) / eff_counters["mc"]["3cl_gen"]) if eff_counters["mc"]["3cl_gen"] > 0 else 0
    results["efficiency"]["m1"]["2cl_err"] = np.sqrt(results["efficiency"]["m1"]["2cl"] * (1 - results["efficiency"]["m1"]["2cl"]) / eff_counters["mc"]["2cl_gen"]) if eff_counters["mc"]["2cl_gen"] > 0 else 0
    results["efficiency"]["m2"]["tot_err"] = np.sqrt(results["efficiency"]["m2"]["tot"] * (1 - results["efficiency"]["m2"]["tot"]) / eff_counters["mc"]["tot_gen"]) if eff_counters["mc"]["tot_gen"] > 0 else 0
    results["efficiency"]["m2"]["3cl_err"] = np.sqrt(results["efficiency"]["m2"]["3cl"] * (1 - results["efficiency"]["m2"]["3cl"]) / eff_counters["mc"]["3cl_gen"]) if eff_counters["mc"]["3cl_gen"] > 0 else 0
    results["efficiency"]["m2"]["2cl_err"] = np.sqrt(results["efficiency"]["m2"]["2cl"] * (1 - results["efficiency"]["m2"]["2cl"]) / eff_counters["mc"]["2cl_gen"]) if eff_counters["mc"]["2cl_gen"] > 0 else 0
    results["fake"]["m1"]["tot_err"] = np.sqrt(results["fake"]["m1"]["tot"] * (1 - results["fake"]["m1"]["tot"]) / eff_counters["m1"]["fake"]["tot_rec"]) if eff_counters["m1"]["fake"]["tot_rec"] > 0 else 0
    results["fake"]["m1"]["3cl_err"] = np.sqrt(results["fake"]["m1"]["3cl"] * (1 - results["fake"]["m1"]["3cl"]) / eff_counters["m1"]["fake"]["3cl_rec"]) if eff_counters["m1"]["fake"]["3cl_rec"] > 0 else 0
    results["fake"]["m1"]["2cl_err"] = np.sqrt(results["fake"]["m1"]["2cl"] * (1 - results["fake"]["m1"]["2cl"]) / eff_counters["m1"]["fake"]["2cl_rec"]) if eff_counters["m1"]["fake"]["2cl_rec"] > 0 else 0
    results["fake"]["m2"]["tot_err"] = np.sqrt(results["fake"]["m2"]["tot"] * (1 - results["fake"]["m2"]["tot"]) / eff_counters["m2"]["fake"]["tot_rec"]) if eff_counters["m2"]["fake"]["tot_rec"] > 0 else 0
    results["fake"]["m2"]["3cl_err"] = np.sqrt(results["fake"]["m2"]["3cl"] * (1 - results["fake"]["m2"]["3cl"]) / eff_counters["m2"]["fake"]["3cl_rec"]) if eff_counters["m2"]["fake"]["3cl_rec"] > 0 else 0
    results["fake"]["m2"]["2cl_err"] = np.sqrt(results["fake"]["m2"]["2cl"] * (1 - results["fake"]["m2"]["2cl"]) / eff_counters["m2"]["fake"]["2cl_rec"]) if eff_counters["m2"]["fake"]["2cl_rec"] > 0 else 0

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


def plot_efficiency_vs_radius(input_dir, output_dir, part):
    """
    Main function to calculate and plot efficiency vs radius.
    """
    # Create dump file
    dump_path = os.path.join(output_dir, "efficiency_vs_radius_dump.txt")
    if "/not_shared/" in input_dir:
        s_str = "not_shared"
    elif "/shared/" in input_dir:
        s_str = "shared"
    else:
        s_str = "not_shared"

    if part == "e":
        particle = "electron"
    elif part == "p":
        particle = "proton"
    elif part == "mu":
        particle = "muon"
        
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
        
        radii = []
        eff_m1 = None
        eff_m1_3cl = None
        eff_m1_2cl = None
        err_m1 = None
        err_m1_3cl = None
        err_m1_2cl = None
        eff_m2_list = []
        eff_m2_3cl_list = []
        eff_m2_2cl_list = []
        err_m2_list = []
        err_m2_3cl_list = []
        err_m2_2cl_list = []
        fake_m1 = None
        fake_m1_3cl = None
        fake_m1_2cl = None
        err_fake_m1 = None
        err_fake_m1_3cl = None
        err_fake_m1_2cl = None
        fake_m2_list = []
        fake_m2_3cl_list = []
        fake_m2_2cl_list = []
        err_fake_m2_list = []
        err_fake_m2_3cl_list = []
        err_fake_m2_2cl_list = []
        counters_m1 = None
        counters_m2_list = []

        data_per_radius = []
        
        # Process all files
        for i, fname in enumerate(files):
            # Calculate M1 only for the first file
            calculate_m1 = (i == 0)
            
            r_val, results, n_tracks_per_event = calculate_efficiency(fname, calculate_m1)

            entry = {
                "radius": r_val,

                "m2": {
                    "efficiency": {
                        "tot": format_latex(
                            results["efficiency"]["m2"]["tot"],
                            results["efficiency"]["m2"]["tot_err"]
                        ),
                        "3cl": format_latex(
                            results["efficiency"]["m2"]["3cl"],
                            results["efficiency"]["m2"]["3cl_err"]
                        ),
                        "2cl": format_latex(
                            results["efficiency"]["m2"]["2cl"],
                            results["efficiency"]["m2"]["2cl_err"]
                        ),
                    },

                    "fake": {
                        "tot": format_latex(
                            results["fake"]["m2"]["tot"],
                            results["fake"]["m2"]["tot_err"]
                        ),
                        "3cl": format_latex(
                            results["fake"]["m2"]["3cl"],
                            results["fake"]["m2"]["3cl_err"]
                        ),
                        "2cl": format_latex(
                            results["fake"]["m2"]["2cl"],
                            results["fake"]["m2"]["2cl_err"]
                        ),
                    }
                }
            }

            data_per_radius.append(entry)
            
            radii.append(r_val)
            eff_m2_list.append(results["efficiency"]["m2"]["tot"])
            eff_m2_3cl_list.append(results["efficiency"]["m2"]["3cl"])
            eff_m2_2cl_list.append(results["efficiency"]["m2"]["2cl"])
            err_m2_list.append(results["efficiency"]["m2"]["tot_err"])
            err_m2_3cl_list.append(results["efficiency"]["m2"]["3cl_err"])
            err_m2_2cl_list.append(results["efficiency"]["m2"]["2cl_err"])
            fake_m2_list.append(results["fake"]["m2"]["tot"])
            fake_m2_3cl_list.append(results["fake"]["m2"]["3cl"])
            fake_m2_2cl_list.append(results["fake"]["m2"]["2cl"])
            err_fake_m2_list.append(results["fake"]["m2"]["tot_err"])
            err_fake_m2_3cl_list.append(results["fake"]["m2"]["3cl_err"])
            err_fake_m2_2cl_list.append(results["fake"]["m2"]["2cl_err"])
            #counters_m2_list.append(counters["m2"])
            
            if calculate_m1:
                eff_m1 = results["efficiency"]["m1"]["tot"]
                eff_m1_3cl = results["efficiency"]["m1"]["3cl"]
                eff_m1_2cl = results["efficiency"]["m1"]["2cl"]
                err_m1 = results["efficiency"]["m1"]["tot_err"]
                err_m1_3cl = results["efficiency"]["m1"]["3cl_err"]
                err_m1_2cl = results["efficiency"]["m1"]["2cl_err"]
                fake_m1 = results["fake"]["m1"]["tot"]
                fake_m1_3cl = results["fake"]["m1"]["3cl"]
                fake_m1_2cl = results["fake"]["m1"]["2cl"]
                err_fake_m1 = results["fake"]["m1"]["tot_err"]
                err_fake_m1_3cl = results["fake"]["m1"]["3cl_err"]
                err_fake_m1_2cl = results["fake"]["m1"]["2cl_err"]

                m1_entry = {
                    "efficiency": {
                        "tot": format_latex(eff_m1, err_m1),
                        "3cl": format_latex(eff_m1_3cl, err_m1_3cl),
                        "2cl": format_latex(eff_m1_2cl, err_m1_2cl),
                    },
                    "fake": {
                        "tot": format_latex(fake_m1, err_fake_m1),
                        "3cl": format_latex(fake_m1_3cl, err_fake_m1_3cl),
                        "2cl": format_latex(fake_m1_2cl, err_fake_m1_2cl),
                    }
                }
                #counters_m1 = counters["m1"]


                # Print M1 results
                '''
                print(f"\nM1 Efficiency: = {100*eff_m1:.2f}%")
                print(f"M1 3-cluster Efficiency: = {100*eff_m1_3cl:.2f}%")
                print(f"M1 2-cluster Efficiency: = {100*eff_m1_2cl:.2f}%")
                print(f"\nM1 Fake Rate: = {100*fake_m1:.2f}%")
                print(f"M1 3-cluster Fake Rate: = {100*fake_m1_3cl:.2f}%")
                print(f"M1 2-cluster Fake Rate: = {100*fake_m1_2cl:.2f}%\n")
            
                '''
            # Print M2 results
            '''
            print(f"M2 Efficiency - Radius: {r_val:.2f} mm = {100*eff_m2_list[-1]:.2f}%")
            print(f"M2 3-cluster Efficiency - Radius: {r_val:.2f} mm = {100*eff_m2_3cl_list[-1]:.2f}%")
            print(f"M2 2-cluster Efficiency - Radius: {r_val:.2f} mm = {100*eff_m2_2cl_list[-1]:.2f}%")
            print(f"\nM2 Fake Rate - Radius: {r_val:.2f} mm = {100*fake_m2_list[-1]:.2f}%")
            print(f"M2 3-cluster Fake Rate - Radius: {r_val:.2f} mm = {100*fake_m2_3cl_list[-1]:.2f}%")
            print(f"M2 2-cluster Fake Rate - Radius: {r_val:.2f} mm = {100*fake_m2_2cl_list[-1]:.2f}%")
        
            '''
        # Write complete report
        #write_detailed_report(dump, radii, eff_m1, eff_m2_list, 
        #                     counters_m1, counters_m2_list)
    
    #print(f"\nDetailed report saved to: {dump_path}")
    
    # Create plots
    create_efficiency_plots(radii, eff_m1, err_m1, eff_m2_list, err_m2_list, output_dir, n_tracks_per_event, miny=0.6, maxy=1.05, cl=None, s_str=s_str, particle=particle)
    create_efficiency_plots(radii, eff_m1_3cl, err_m1_3cl, eff_m2_3cl_list, err_m2_3cl_list, output_dir, n_tracks_per_event, miny=0.6, maxy=1.05, cl=3, s_str=s_str, particle=particle)
    create_efficiency_plots(radii, eff_m1_2cl, err_m1_2cl, eff_m2_2cl_list, err_m2_2cl_list, output_dir, n_tracks_per_event, miny=0.6, maxy=1.05, cl=2, s_str=s_str, particle=particle)
    create_fake_rate_plots(radii, fake_m1, err_fake_m1, fake_m2_list, err_fake_m2_list, output_dir, n_tracks_per_event, miny=0.0, maxy=0.02, cl=None, s_str=s_str, particle=particle)
    create_fake_rate_plots(radii, fake_m1_3cl, err_fake_m1_3cl, fake_m2_3cl_list, err_fake_m2_3cl_list, output_dir, n_tracks_per_event, miny=0.0, maxy=0.04, cl=3, s_str=s_str, particle=particle)
    create_fake_rate_plots(radii, fake_m1_2cl, err_fake_m1_2cl, fake_m2_2cl_list, err_fake_m2_2cl_list, output_dir, n_tracks_per_event, miny=0.0, maxy=0.04, cl=2, s_str=s_str, particle=particle)

    final_dict = {
        "input_dir": input_dir,
        "particle": particle,
        "mode": s_str,
        "data": data_per_radius,
        "m1": m1_entry
    }

    output_json = os.path.join(output_dir, f"efficiency_fake_r_{part}_{s_str}.json")
    json_path = os
    with open(output_json, "w") as f:
       json.dump(final_dict, f, indent=4)


def create_efficiency_plots(radii, eff_m1, err_m1, eff_m2_list, err_m2_list, output_dir, n_tracks_per_event, miny, maxy, cl=None, s_str=None, particle="electron"):
    """
    Create efficiency plots.
    """
    n_points = len(radii)
    
    # Create canvas
    c = ROOT.TCanvas(f"c_eff_vs_radius_{cl if cl is not None else 'all'}_cl", f"Efficiency vs Radius - {cl if cl is not None else 'All'} Clusters ({particle}s)", 800, 600)
    
    # M2 graph
    gr_m2 = ROOT.TGraphErrors(n_points, np.array(radii, dtype='float64'), 
                        np.array(eff_m2_list, dtype='float64'), 
                        np.zeros(n_points), np.array(err_m2_list, dtype='float64'))
    gr_m2.SetMarkerStyle(20)
    gr_m2.SetMarkerColor(ROOT.kRed)
    gr_m2.SetLineColor(ROOT.kRed)
    gr_m2.SetMaximum(maxy)
    gr_m2.SetMinimum(miny)
    gr_m2.SetTitle(f"Efficiency vs Radius - {n_tracks_per_event} tracks per event - {cl if cl is not None else 'All'} Clusters ({particle}s);Radius [mm];Efficiency")
    gr_m2.SetName("gr_m2")
    gr_m2.Draw("AP")
    
    # Horizontal line for M1 (constant)
    if eff_m1 is not None:
        xmin, xmax = min(radii), max(radii)
        line_m1 = ROOT.TLine(xmin, eff_m1, xmax, eff_m1)
        line_m1.SetLineColor(ROOT.kBlue)
        line_m1.SetLineWidth(2)
        line_m1.SetLineStyle(2)  # Dashed line
        line_m1.Draw()
        
        # Legend
        leg = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
        leg.AddEntry(line_m1, f"M1 = {100*eff_m1:.2f}%", "l")
        leg.AddEntry(gr_m2, "M2", "p")
        leg.SetTextSize(0.04)
        leg.SetFillStyle(1001)
        leg.SetFillColor(ROOT.kWhite)
        leg.Draw()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"efficiency_vs_radius_{n_tracks_per_event}_evs_{cl if cl is not None else 'all'}_cl_{s_str}.pdf")
    c.SaveAs(plot_path)
    print(f"Plot saved to: {plot_path}")


def create_fake_rate_plots(radii, fake_m1, err_fake_m1, fake_m2_list, err_fake_m2_list, output_dir, n_tracks_per_event, miny, maxy, cl=None, s_str=None, particle="electron"):
    """
    Create fake rate plots.
    """
    n_points = len(radii)
    
    # Create canvas
    c = ROOT.TCanvas(f"c_fake_vs_radius_{cl if cl is not None else 'all'}_cl", f"Fake Rate vs Radius - {cl if cl is not None else 'All'} Clusters ({particle}s)", 800, 600)
    
    # M2 graph
    gr_m2 = ROOT.TGraphErrors(n_points, np.array(radii, dtype='float64'), 
                        np.array(fake_m2_list, dtype='float64'), 
                        np.zeros(n_points), np.array(err_fake_m2_list, dtype='float64'))
    gr_m2.SetMarkerStyle(20)
    gr_m2.SetMarkerColor(ROOT.kRed)
    gr_m2.SetLineColor(ROOT.kRed)
    gr_m2.SetMaximum(maxy)
    gr_m2.SetMinimum(miny)
    gr_m2.SetTitle(f"Fake Rate vs Radius - {n_tracks_per_event} tracks per event - {cl if cl is not None else 'All'} Clusters ({particle}s);Radius [mm];Fake Rate")
    gr_m2.GetYaxis().SetTitleOffset(1.5);
    gr_m2.SetName("gr_m2")
    gr_m2.Draw("AP")
    
    # Horizontal line for M1 (constant)
    if fake_m1 is not None:
        xmin, xmax = min(radii), max(radii)
        line_m1 = ROOT.TLine(xmin, fake_m1, xmax, fake_m1)
        line_m1.SetLineColor(ROOT.kBlue)
        line_m1.SetLineWidth(2)
        line_m1.SetLineStyle(2)  # Dashed line
        line_m1.Draw()
        
        # Legend
        leg = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
        leg.AddEntry(line_m1, f"M1 = {100*fake_m1:.2f}%", "l")
        leg.AddEntry(gr_m2, "M2", "p")
        leg.SetTextSize(0.04)
        leg.SetFillStyle(1001)
        leg.SetFillColor(ROOT.kWhite)
        leg.Draw()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"fake_rate_vs_radius_{n_tracks_per_event}_evs_{cl if cl is not None else 'all'}_cl_{s_str}.pdf")
    c.SaveAs(plot_path)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate efficiency / fake rate vs radius.")
    parser.add_argument("--input-dir", required=True, 
                       help="Directory containing .root files.")
    parser.add_argument("--output-dir", default=None, 
                       help="Define output directory.")
    parser.add_argument("--particle", required=True, help="Select particle configuration.")

    args = parser.parse_args()

    if args.particle not in ["e", "p", "c", "mu"]:
        print(f"Error: Invalid particle type '{args.particle}'. Must be one of: e, p, c.")
        sys.exit(1)
    
    config_file = "config/config_efficiency.yaml"
    with open(config_file, "r") as f_cfg:
        config = yaml.safe_load(f_cfg)
    
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = config["output_dir"]
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    plot_efficiency_vs_radius(args.input_dir, output_dir, args.particle)