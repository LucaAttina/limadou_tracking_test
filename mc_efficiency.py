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
    make_purity_hist,
    process_efficiency_event,
    check_angle_mismatch,
    read_data_from_file,
    write_counters,
    print_counters,
    make_efficiency_hist,
    make_fake_hist,
    compute_gen_mask, 
    get_bins,
)


def compute_efficiency(input_file, output_dir, config, plot_flag, theta_angle_threshold=5.0, phi_angle_threshold=5.0):
    """
    Main function to calculate tracking efficiency and create plots.
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Open file and read basic data
    f = uproot.open(input_file)
    print(f"\nProcessing file: {input_file}")
    
    # Read basic data using utility function
    file_data = read_data_from_file(f)

    n_events = len(file_data["mult_data"]["event_idx"])
    n_tracks_per_event = len(file_data["gen_data"]["theta"][0])
    print(f"Number of events: {n_events}")
    print(f"Tracks per event: {n_tracks_per_event}")

    #print(f"gen_cl_x: {file_data["gen_data"]["cls"]["x"]}")
    #print(f"theta_gen: {file_data["gen_data"]["theta"]}")

    #print_selected_data(file_data)
    # Open dump file
    dump_path = os.path.join(log_dir, f"efficiency_dump_{n_tracks_per_event}_evs_0p05.txt")
    dump = open(dump_path, 'w')

    # Print track counts
    print("\nTotal tracks:")
    print(f" - M1 (Hough): {file_data['n_tracks']['m1']}")
    print(f" - M2 (Combinatorial): {file_data['n_tracks']['m2']}")
    print(f" - MC: {file_data['n_tracks']['mc']} (correctly generated ones)")

    eff_info, rec_info, eff_counters = initialize_counters(n_events, n_tracks_per_event, file_data)
    
    # Dump file header
    dump.write("="*100 + "\n")
    dump.write("TRACKING EFFICIENCY DEBUG DUMP\n")
    dump.write(f"File: {input_file}\n")
    dump.write("="*100 + "\n")
    
    # totale MC tracks

    tot_fake_m1 = 0
    tot_fake_m2 = 0
    count_mult = 0

    fake_theta_m1 = []
    fake_phi_m1 = []
    fake_theta_m2 = []
    fake_phi_m2 = []
    fake_theta_3cl_m1 = []
    fake_phi_3cl_m1 = []
    fake_theta_3cl_m2 = []
    fake_phi_3cl_m2 = []
    fake_theta_2cl_m1 = []
    fake_phi_2cl_m1 = []
    fake_theta_2cl_m2 = []
    fake_phi_2cl_m2 = []
    pur_theta_m1 = []
    pur_phi_m1 = []
    pur_theta_m2 = []
    pur_phi_m2 = []

    all_reco_theta_m1 = []
    all_reco_phi_m1 = []
    all_reco_theta_m2 = []
    all_reco_phi_m2 = []
    all_reco_theta_3cl_m1 = []
    all_reco_phi_3cl_m1 = []
    all_reco_theta_2cl_m1 = []
    all_reco_phi_2cl_m1 = []
    all_reco_theta_3cl_m2 = []
    all_reco_phi_3cl_m2 = []
    all_reco_theta_2cl_m2 = []
    all_reco_phi_2cl_m2 = []


    # Process all events
    for ev_idx in range(n_events):

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
        if mult_m2 >= 2:
            count_mult += 1
        
        # Check if event has data for verbose output
        has_data = (mult_m1 != 0) or (mult_m2 != 0)
        
        if has_data:
            dump.write(f"\n\n\n{'='*80}\n")
            dump.write(f"EVENT {ev_idx}\n")
            dump.write(f"{'='*80}\n")
            dump.write(f"Multiplicity: M1={mult_m1}, M2={mult_m2}")

        #print(f"rec_m1_cls_x ev {ev_idx}: {file_data['rec_data']['m1']['cls']['x'][ev_idx]}")
        #print(f"gen_theta ev {ev_idx}: {file_data['gen_data']['theta'][ev_idx]}")
        gen_trk_idx = file_data["gen_data"]["trk_id"][ev_idx]

        for clsn in range(n_tracks_per_event):
            # Conta quanti elementi ci sono per questa combinazione (ev_idx, clsn)
            count = np.sum(file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx] == clsn)
            if count >= 4:
                print(f"Violazione: evento {ev_idx}, classe {clsn} ha {count} elementi (>=4)")

        reco_m1_idx = np.where(ev_idx == file_data["rec_data"]["m1"]["event_idx"])[0]
        #print(f"pre {rec_info['m1'][ev_idx]}")

        if reco_m1_idx.tolist():

            # Process M1
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
                dump=dump
            )

            #print(f"post {rec_info['m1'][ev_idx]}")

            tot_fake_m1 += n_fake_m1

            for i, reco_idx in enumerate(reco_m1_idx):
                theta = file_data["rec_data"]["m1"]["theta"][reco_idx]
                phi = file_data["rec_data"]["m1"]["phi"][reco_idx]

                all_reco_theta_m1.append(theta)
                all_reco_phi_m1.append(phi)
                n_cls = len([c for c in file_data["rec_data"]["m1"]["cls"]["x"][reco_idx] if c != -999])

                if n_cls == 3:
                    all_reco_theta_3cl_m1.append(theta)
                    all_reco_phi_3cl_m1.append(phi)
                else:
                    all_reco_theta_2cl_m1.append(theta)
                    all_reco_phi_2cl_m1.append(phi)
                #print(f"ev: {ev_idx} - reco_idx: {reco_idx} - rec_info: {rec_info['m1'][ev_idx][i]} - theta: {theta} - phi: {phi}")

                if not rec_info["m1"][ev_idx][i]:
                    fake_theta_m1.append(theta)
                    fake_phi_m1.append(phi)

                    if n_cls == 3:
                        fake_theta_3cl_m1.append(theta)
                        fake_phi_3cl_m1.append(phi)

                    else:
                        fake_theta_2cl_m1.append(theta)
                        fake_phi_2cl_m1.append(phi)
                else:
                    pur_theta_m1.append(theta)
                    pur_phi_m1.append(phi)
                    #print(f"ev: {ev_idx} - reco_idx: {reco_idx} - FAKE track in M1 with theta: {theta} and phi: {phi}")

            #for trk_idx in range(n_tracks_per_event):
            #    if good_for_eff_m1[trk_idx]:
            #        rec_flags["m1"][ev_idx, trk_idx] = True

            # Dump M1 info
            '''
            if has_data:
                dump.write(f"\n--- Method M1 ---\n")

                if mult_m1 == 0:
                    dump.write(f"No reconstructed tracks (multiplicity=0)\n")
                else:
                    dump.write("\nMC truth clusters:\n")
                    for i, (x, y, l, idx) in enumerate(zip(
                        file_data["gen_data"]["cls"]["x"][ev_idx], 
                        file_data["gen_data"]["cls"]["y"][ev_idx], 
                        file_data["gen_data"]["cls"]["layer"][ev_idx],
                        file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx]
                    )):
                        dump.write(f"  MC {i}: x={x:.2f}, y={y:.2f}, layer={l}, cls_to_trk_idx={idx}\n")

                    # Per ogni traccia MC
                    for ntr in range(n_tracks_per_event):
                        dump.write(f"\nMC track {ntr}: gen_theta={file_data['gen_data']['theta'][ev_idx][ntr]:.3f}°, "
                                  f"gen_phi={file_data['gen_data']['phi'][ev_idx][ntr]:.3f}°\n")

                        # Stampa i cluster MC di questa traccia
                        for i, (x, y, l, idx) in enumerate(zip(
                            file_data["gen_data"]["cls"]["x"][ev_idx], 
                            file_data["gen_data"]["cls"]["y"][ev_idx], 
                            file_data["gen_data"]["cls"]["layer"][ev_idx],
                            file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx]
                        )):
                            if ntr == idx:
                                dump.write(f"  MC Cluster {i}: x={x:.2f}, y={y:.2f}, layer={l}, cls_to_trk_idx={idx}\n")

                        # Usa reco_to_gen_m1 per trovare la traccia ricostruita associata
                        # reco_to_gen_m1 è un dict: {gen_track_id: reco_track_id} o una lista dove l'indice è gen_track_id
                        if good_for_eff_m1[ntr] and reco_to_gen_m1[ntr] is not None:
                            reco_id = reco_to_gen_m1[ntr]  # Questo è l'ID della traccia ricostruita

                            # Trova la posizione nei dati ricostruiti
                            if reco_id < len(reco_m1_idx):
                                reco_pos = reco_m1_idx[reco_id]

                                dump.write(f"* Matching Reco track {reco_id}:\n")
                                dump.write(f"  reco_theta={file_data['rec_data']['m1']['theta'][reco_pos]:.3f}°, "
                                          f"reco_phi={file_data['rec_data']['m1']['phi'][reco_pos]:.3f}°\n")

                                # Stampa i cluster ricostruiti
                                for i, (x, y, z) in enumerate(zip(
                                    file_data["rec_data"]["m1"]["cls"]["x"][reco_pos], 
                                    file_data["rec_data"]["m1"]["cls"]["y"][reco_pos], 
                                    file_data["rec_data"]["m1"]["cls"]["z"][reco_pos]
                                )):
                                    if x != -999 and y != -999 and z != -999:
                                        if abs(z - 17.825) < 1e-3:
                                            lay = 0
                                        elif abs(z - 26.325) < 1e-3:
                                            lay = 1
                                        elif abs(z - 34.825) < 1e-3:
                                            lay = 2
                                        else:
                                            lay = -1
                                        dump.write(f"  Reco Cluster {i}: x={x:.2f}, y={y:.2f}, z={z:.2f} (layer {lay})\n")
                            else:
                                dump.write(f"* Warning: reco_id {reco_id} out of range\n")
                        else:
                            dump.write("* Not reconstructed\n")
#                            for i, (x, y, l) in enumerate(zip(
#                                file_data["rec_data"]["m1"]["cls"]["x"][reco_m1_idx], 
#                                file_data["rec_data"]["m1"]["cls"]["y"][reco_m1_idx], 
#                                file_data["rec_data"]["m1"]["cls"]["z"][reco_m1_idx],
#                                #file_data["rec_data"]["m1"]["cls"]["cls_to_trk"][reco_m1_idx]
#                            )):
#                                print(f"{file_data["rec_data"]["m1"]["cls"]["x"][reco_m1_idx]}")
#                                dump.write(f"  Reco {i}: x={x:.2f}, y={y:.2f}, layer={l}, cls_to_trk_idx={idx}\n")
#                good_values = list(good_for_eff_m1.values())
#                dump.write(f"\n -- Correctly reconstructed tracks: {sum(good_values)}/{n_tracks_per_event} --\n")
                '''
        
        reco_m2_idx = np.where(ev_idx == file_data["rec_data"]["m2"]["event_idx"])[0]
        if reco_m2_idx.tolist():
            #print(f"reco_m2_idx = {reco_m2_idx}")
            #print(f"rec_cls_x: {file_data["rec_data"]["m2"]["cls"]["x"][reco_m2_idx]}")
            #print(f"reco_theta_m2: {file_data["rec_data"]["m2"]["theta"][reco_m2_idx]}")
            #print(f"gen_theta_m2: {file_data["gen_data"]["theta"][ev_idx]}")
            #print(f"pid: {file_data["gen_data"]["trk_id"][ev_idx]}")
        
            #print(f"ev_idx: {ev_idx} - mult_m2: {mult_m2}: {file_data["rec_data"]["m2"]["d_sum"][reco_m2_idx]}")

            # Process M2
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
                dump=dump
            )

            tot_fake_m2 += n_fake_m2

            for i, reco_idx in enumerate(reco_m2_idx):
                theta = file_data["rec_data"]["m2"]["theta"][reco_idx]
                phi = file_data["rec_data"]["m2"]["phi"][reco_idx]

                all_reco_theta_m2.append(theta)
                all_reco_phi_m2.append(phi)
                n_cls = len([c for c in file_data["rec_data"]["m2"]["cls"]["x"][reco_idx] if c != -999])

                if n_cls == 3:
                    all_reco_theta_3cl_m2.append(theta)
                    all_reco_phi_3cl_m2.append(phi)
                else:
                    all_reco_theta_2cl_m2.append(theta)
                    all_reco_phi_2cl_m2.append(phi)
                #print(f"ev: {ev_idx} - reco_idx: {reco_idx} - rec_info: {rec_info['m2'][ev_idx][i]} - theta: {theta} - phi: {phi}")

                if not rec_info["m2"][ev_idx][i]:
                    fake_theta_m2.append(theta)
                    fake_phi_m2.append(phi)

                    if n_cls == 3:
                        fake_theta_3cl_m2.append(theta)
                        fake_phi_3cl_m2.append(phi)
                    else:
                        fake_theta_2cl_m2.append(theta)
                        fake_phi_2cl_m2.append(phi)
                else:
                    pur_theta_m2.append(theta)
                    pur_phi_m2.append(phi)
                    #print(f"ev: {ev_idx} - reco_idx: {reco_idx} - FAKE track in M2 with theta: {theta} and phi: {phi}")

            #print(f"{eff_counters}")

            #for trk_idx in range(n_tracks_per_event):
            #    if good_for_eff_m2[trk_idx]:
            #        rec_flags["m2"][ev_idx, trk_idx] = True
            
            # Dump M2 info
            '''
            if has_data:
                dump.write(f"\n--- Method M2 ---\n")

                if mult_m2 == 0:
                    dump.write(f"No reconstructed tracks (multiplicity=0)\n")
                else:
                    dump.write("\nMC truth clusters:\n")
                    for i, (x, y, l, idx) in enumerate(zip(
                        file_data["gen_data"]["cls"]["x"][ev_idx], 
                        file_data["gen_data"]["cls"]["y"][ev_idx], 
                        file_data["gen_data"]["cls"]["layer"][ev_idx],
                        file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx]
                    )):
                        dump.write(f"  MC {i}: x={x:.2f}, y={y:.2f}, layer={l}, cls_to_trk_idx={idx}\n")

                    # Per ogni traccia MC
                    for ntr in range(n_tracks_per_event):
                        dump.write(f"\nMC track {ntr}: gen_theta={file_data['gen_data']['theta'][ev_idx][ntr]:.3f}°, "
                                  f"gen_phi={file_data['gen_data']['phi'][ev_idx][ntr]:.3f}°\n")

                        # Stampa i cluster MC di questa traccia
                        for i, (x, y, l, idx) in enumerate(zip(
                            file_data["gen_data"]["cls"]["x"][ev_idx], 
                            file_data["gen_data"]["cls"]["y"][ev_idx], 
                            file_data["gen_data"]["cls"]["layer"][ev_idx],
                            file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx]
                        )):
                            if ntr == idx:
                                dump.write(f"  MC Cluster {i}: x={x:.2f}, y={y:.2f}, layer={l}, cls_to_trk_idx={idx}\n")

                        # Usa reco_to_gen_m2 per trovare la traccia ricostruita associata
                        # reco_to_gen_m2 è un dict: {gen_track_id: reco_track_id} o una lista dove l'indice è gen_track_id
                        if good_for_eff_m2[ntr] and reco_to_gen_m2[ntr] is not None:
                            reco_id = reco_to_gen_m2[ntr]  # Questo è l'ID della traccia ricostruita

                            # Trova la posizione nei dati ricostruiti
                            if reco_id < len(reco_m2_idx):
                                reco_pos = reco_m2_idx[reco_id]

                                dump.write(f"* Matching Reco track {reco_id}:\n")
                                dump.write(f"  reco_theta={file_data['rec_data']['m2']['theta'][reco_pos]:.3f}°, "
                                          f"reco_phi={file_data['rec_data']['m2']['phi'][reco_pos]:.3f}°\n")

                                # Stampa i cluster ricostruiti
                                for i, (x, y, z) in enumerate(zip(
                                    file_data["rec_data"]["m2"]["cls"]["x"][reco_pos], 
                                    file_data["rec_data"]["m2"]["cls"]["y"][reco_pos], 
                                    file_data["rec_data"]["m2"]["cls"]["z"][reco_pos]
                                )):
                                    if x != -999 and y != -999 and z != -999:
                                        if abs(z - 17.825) < 1e-3:
                                            lay = 0
                                        elif abs(z - 26.325) < 1e-3:
                                            lay = 1
                                        elif abs(z - 34.825) < 1e-3:
                                            lay = 2
                                        else:
                                            lay = -1
                                        dump.write(f"  Reco Cluster {i}: x={x:.2f}, y={y:.2f}, z={z:.2f} (layer {lay})\n")
                            else:
                                dump.write(f"* Warning: reco_id {reco_id} out of range\n")
                        else:
                            dump.write("* Not reconstructed\n")
                good_values = list(good_for_eff_m2.values())
                dump.write(f"\n -- Correctly reconstructed tracks: {sum(good_values)}/{n_tracks_per_event} --\n")
                '''

    
    # Write counters to dump file using utility function
    #for method in ["m1", "m2"]:
    #    write_counters(dump, counters[method], method)
    
    dump.close()
    print(f"\nDump saved to: {dump_path}")
    print(f"\n\n Events with mult > 1: {count_mult}")
    
    # Print counters to console
    #for method in ["m1", "m2"]:
    #    print_counters(counters[method], method)
    #    
    #    # Calculate and print efficiency
    #    if counters[method]['total_gen_good'] > 0:
    #        eff = counters[method]['reconstructed_good'] / counters[method]['total_gen_good']
    #        print(f"Efficiency: {counters[method]['reconstructed_good']}/{counters[method]['total_gen_good']} = {100*eff:.2f}%\n")
    
    # ========================================
    # CALCOLO EFFICIENZE DA eff_info
    # ========================================

    #print(f"{eff_info["m2"]}")
    
    # Calcola il numero totale di MC tracks (denominatore)
    total_mc_tracks = n_events * n_tracks_per_event

    # Conta quante sono state ricostruite per ogni metodo
    n_reco_m1 = sum(eff_info["m1"][ev_idx][trk_idx] 
                    for ev_idx in range(n_events) 
                    for trk_idx in range(n_tracks_per_event))
    
    n_reco_m2 = sum(eff_info["m2"][ev_idx][trk_idx] 
                    for ev_idx in range(n_events) 
                    for trk_idx in range(n_tracks_per_event))
    
    # Stampa efficienze
    #print("\n" + "="*50)
    #print("EFFICIENCY RESULTS")
    #print("="*50)
    #print(f"Total MC tracks: {total_mc_tracks}")
    #print(f"M1 reconstructed: {n_reco_m1} / {total_mc_tracks} = {100*n_reco_m1/total_mc_tracks:.2f}%")
    #print(f"M2 reconstructed: {n_reco_m2} / {total_mc_tracks} = {100*n_reco_m2/total_mc_tracks:.2f}%")
    #if file_data['n_tracks']['m1'] > 0:
    #    print(f"Fake tracks M1: {tot_fake_m1} / {file_data['n_tracks']['m1']} = {100*tot_fake_m1/file_data['n_tracks']['m1']:.2f}%")
    #print(f"Fake tracks M2: {tot_fake_m2} / {file_data['n_tracks']['m2']} = {100*tot_fake_m2/file_data['n_tracks']['m2']:.2f}%")


    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    for meth in ["m1", "m2"]:

        rec_gen_all = eff_counters[meth]["eff"]["tot_rec_gen"]
        gen_all = eff_counters["mc"]["tot_gen"]
        rec_gen_3cls = eff_counters[meth]["eff"]["3cl_rec_gen"]
        gen_3cls = eff_counters["mc"]["3cl_gen"]
        rec_gen_2cls = eff_counters[meth]["eff"]["2cl_rec_gen"]
        gen_2cls = eff_counters["mc"]["2cl_gen"]

        fp_all = eff_counters[meth]["fake"]["tot_fake"]
        rec_all = eff_counters[meth]["fake"]["tot_rec"] # tp + fp
        fp_3cls = eff_counters[meth]["fake"]["3cl_fake"]
        rec_3cls = eff_counters[meth]["fake"]["3cl_rec"]
        fp_2cls = eff_counters[meth]["fake"]["2cl_fake"]
        rec_2cls = eff_counters[meth]["fake"]["2cl_rec"]

        if rec_all != rec_gen_all + fp_all:
            print(f"Warning: Mismatch in total reconstructed tracks for {meth}: tp ({rec_gen_all}) + fp ({fp_all}) = {rec_gen_all + fp_all} vs rec_all = {rec_all}")
        if rec_3cls != rec_gen_3cls + fp_3cls:
            print(f"Warning: Mismatch in 3-cluster reconstructed tracks for {meth}: tp ({rec_gen_3cls}) + fp ({fp_3cls}) = {rec_gen_3cls + fp_3cls} vs rec_3cls = {rec_3cls}")
        if rec_2cls != rec_gen_2cls + fp_2cls:
            print(f"Warning: Mismatch in 2-cluster reconstructed tracks for {meth}: tp ({rec_gen_2cls}) + fp ({fp_2cls}) = {rec_gen_2cls + fp_2cls} vs rec_2cls = {rec_2cls}")

        print(f"---------- {meth} ----------")

        print(f"{meth} efficiency: {rec_gen_all} / {gen_all} = {100*rec_gen_all/gen_all:.2f}%")
        print(f"{meth} efficiency (3 cls): {rec_gen_3cls} / {gen_3cls} = {100*rec_gen_3cls/gen_3cls:.2f}%")
        print(f"{meth} efficiency (2 cls): {rec_gen_2cls} / {gen_2cls} = {100*rec_gen_2cls/gen_2cls:.2f}%\n")

        if rec_all > 0:
            print(f"{meth} fake: {fp_all} / {rec_all} = {100*fp_all/rec_all:.2f}%")
            print(f"{meth} purity: {rec_all-fp_all} / {rec_all} = {100*(rec_all-fp_all)/rec_all:.2f}%")
        else:
            print(f"{meth} fake: {fp_all} / {rec_all} = N/A (no reconstructed tracks)")
            print(f"{meth} purity: {rec_all-fp_all} / {rec_all} = N/A (no reconstructed tracks)")
        if rec_3cls > 0:
            print(f"{meth} fake (3 cls): {fp_3cls} / {rec_3cls} = {100*fp_3cls/rec_3cls:.2f}%")
            print(f"{meth} purity (3 cls): {rec_3cls-fp_3cls} / {rec_3cls} = {100*(rec_3cls-fp_3cls)/rec_3cls:.2f}%")
        else:
            print(f"{meth} fake (3 cls): {fp_3cls} / {rec_3cls} = N/A (no reconstructed tracks with 3 cls)")
            print(f"{meth} purity (3 cls): {rec_3cls-fp_3cls} / {rec_3cls} = N/A (no reconstructed tracks with 3 cls)")
        if rec_2cls > 0:
            print(f"{meth} fake (2 cls): {fp_2cls} / {rec_2cls} = {100*fp_2cls/rec_2cls:.2f}%")
            print(f"{meth} purity (2 cls): {rec_2cls-fp_2cls} / {rec_2cls} = {100*(rec_2cls-fp_2cls)/rec_2cls:.2f}%\n")
        else:
            print(f"{meth} fake (2 cls): {fp_2cls} / {rec_2cls} = N/A (no reconstructed tracks with 2 cls)")
            print(f"{meth} purity (2 cls): {rec_2cls-fp_2cls} / {rec_2cls} = N/A (no reconstructed tracks with 2 cls)\n")


        
        '''
        print(f"{meth} efficiency: {eff_counters[meth]["eff"]["tot_rec_gen"]} / {eff_counters["mc"]["tot_gen"]} = {100*eff_counters[meth]["eff"]["tot_rec_gen"]/eff_counters["mc"]["tot_gen"]:.2f}%")
        print(f"{meth} efficiency (3 cls): {eff_counters[meth]["eff"]["3cl_rec_gen"]} / {eff_counters["mc"]["3cl_gen"]} = {100*eff_counters[meth]["eff"]["3cl_rec_gen"]/eff_counters["mc"]["3cl_gen"]:.2f}%")
        print(f"{meth} efficiency (2 cls): {eff_counters[meth]["eff"]["2cl_rec_gen"]} / {eff_counters["mc"]["2cl_gen"]} = {100*eff_counters[meth]["eff"]["2cl_rec_gen"]/eff_counters["mc"]["2cl_gen"]:.2f}%\n")

        if eff_counters[meth]["fake"]["tot_rec"] > 0:
            print(f"{meth} fake: {eff_counters[meth]["fake"]["tot_fake"]} / {eff_counters[meth]["fake"]["tot_rec"]} = {100*eff_counters[meth]["fake"]["tot_fake"]/eff_counters[meth]["fake"]["tot_rec"]:.2f}%")

        else:
            print(f"{meth} fake: {eff_counters[meth]["fake"]["tot_fake"]} / {eff_counters[meth]["fake"]["tot_rec"]} = N/A (no reconstructed tracks)")
        
        if eff_counters[meth]["fake"]["3cl_rec"] > 0:
            print(f"{meth} fake (3 cls): {eff_counters[meth]["fake"]["3cl_fake"]} / {eff_counters[meth]["fake"]["3cl_rec"]} = {100*eff_counters[meth]["fake"]["3cl_fake"]/eff_counters[meth]["fake"]["3cl_rec"]:.2f}%")
        else:
            print(f"{meth} fake (3 cls): {eff_counters[meth]["fake"]["3cl_fake"]} / {eff_counters[meth]["fake"]["3cl_rec"]} = N/A (no reconstructed tracks with 3 cls)")
        if eff_counters[meth]["fake"]["2cl_rec"] > 0:
            print(f"{meth} fake (2 cls): {eff_counters[meth]["fake"]["2cl_fake"]} / {eff_counters[meth]["fake"]["2cl_rec"]} = {100*eff_counters[meth]["fake"]["2cl_fake"]/eff_counters[meth]["fake"]["2cl_rec"]:.2f}%\n")
        else:
            print(f"{meth} fake (2 cls): {eff_counters[meth]["fake"]["2cl_fake"]} / {eff_counters[meth]["fake"]["2cl_rec"]} = N/A (no reconstructed tracks with 2 cls)\n")

        '''
        #print(f"{len(fake_theta_m1)} fake tracks in M1")
        #print(f"{len(fake_theta_m2)} fake tracks in M2")
        #print(f"{len(all_reco_theta_m1)} total reco tracks in M1")
        #print(f"{len(all_reco_theta_m2)} total reco tracks in M2")


        #if file_data['n_tracks']['m1'] > 0:
        #    print(f"Fake tracks M1: {tot_fake_m1} / {file_data['n_tracks']['m1']} = {100*tot_fake_m1/file_data['n_tracks']['m1']:.2f}%")
        #print(f"Fake tracks M2: {tot_fake_m2} / {file_data['n_tracks']['m2']} = {100*tot_fake_m2/file_data['n_tracks']['m2']:.2f}%")


    # Histogram data setting
    gen_theta_flat = []
    rec_theta_m1_flat = []
    rec_theta_m2_flat = []
    gen_phi_flat = []
    rec_phi_m1_flat = []
    rec_phi_m2_flat = []
    gen_theta_flat_3cls = []
    rec_theta_m1_flat_3cls = []
    rec_theta_m2_flat_3cls = []
    gen_phi_flat_3cls = []
    rec_phi_m1_flat_3cls = []
    rec_phi_m2_flat_3cls = []
    gen_theta_flat_2cls = []
    rec_theta_m1_flat_2cls = []
    rec_theta_m2_flat_2cls = []
    gen_phi_flat_2cls = []
    rec_phi_m1_flat_2cls = []
    rec_phi_m2_flat_2cls = []


    # Itera su tutti gli eventi e tutte le MC tracks
    for ev_idx in range(n_events):
        for trk_idx in range(n_tracks_per_event):
            # Aggiungi al denominatore (tutte le MC tracks)
            gen_theta_flat.append(file_data["gen_data"]["theta"][ev_idx][trk_idx])
            gen_phi_flat.append(file_data["gen_data"]["phi"][ev_idx][trk_idx])
            mc_cls_n = 0
            mc_cls_n = sum(1 for x in file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx] if x == trk_idx)
            if mc_cls_n == 2:
                gen_theta_flat_2cls.append(file_data["gen_data"]["theta"][ev_idx][trk_idx])
                gen_phi_flat_2cls.append(file_data["gen_data"]["phi"][ev_idx][trk_idx])
            elif mc_cls_n == 3:
                gen_theta_flat_3cls.append(file_data["gen_data"]["theta"][ev_idx][trk_idx])
                gen_phi_flat_3cls.append(file_data["gen_data"]["phi"][ev_idx][trk_idx])

            rec_cls_n = 0

            # M1 - aggiungi solo se ricostruita
            if eff_info["m1"][ev_idx][trk_idx]:
                rec_theta_m1_flat.append(file_data["gen_data"]["theta"][ev_idx][trk_idx])
                rec_phi_m1_flat.append(file_data["gen_data"]["phi"][ev_idx][trk_idx])
                rec_cls_n = sum(1 for x in file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx] if x == trk_idx)
                if rec_cls_n == 2:
                    rec_theta_m1_flat_2cls.append(file_data["gen_data"]["theta"][ev_idx][trk_idx])
                    rec_phi_m1_flat_2cls.append(file_data["gen_data"]["phi"][ev_idx][trk_idx])
                elif rec_cls_n == 3:
                    rec_theta_m1_flat_3cls.append(file_data["gen_data"]["theta"][ev_idx][trk_idx])
                    rec_phi_m1_flat_3cls.append(file_data["gen_data"]["phi"][ev_idx][trk_idx])
            
            # M2 - aggiungi solo se ricostruita
            if eff_info["m2"][ev_idx][trk_idx]:
                rec_theta_m2_flat.append(file_data["gen_data"]["theta"][ev_idx][trk_idx])
                rec_phi_m2_flat.append(file_data["gen_data"]["phi"][ev_idx][trk_idx])
                rec_cls_n = sum(1 for x in file_data["gen_data"]["cls"]["cls_to_trk"][ev_idx] if x == trk_idx)
                if rec_cls_n != 2 and rec_cls_n != 3:
                    print(f"ev_idx: {ev_idx} - trk_idx: {trk_idx} - Warning: rec_cls_n = {rec_cls_n} (not 2 or 3)")
                if rec_cls_n == 2:
                    rec_theta_m2_flat_2cls.append(file_data["gen_data"]["theta"][ev_idx][trk_idx])
                    rec_phi_m2_flat_2cls.append(file_data["gen_data"]["phi"][ev_idx][trk_idx])
                elif rec_cls_n == 3:
                    rec_theta_m2_flat_3cls.append(file_data["gen_data"]["theta"][ev_idx][trk_idx])
                    rec_phi_m2_flat_3cls.append(file_data["gen_data"]["phi"][ev_idx][trk_idx])

    #print(f"Total gen tracks (denominator): {len(gen_theta_flat)}")
    #print(f"Total gen tracks with 3 cls: {len(gen_theta_flat_3cls)}")
    #print(f"Total gen tracks with 2 cls: {len(gen_theta_flat_2cls)}")
    #print(f"Total reco M1 tracks: {len(rec_theta_m1_flat)}")
    #print(f"Total reco M2 tracks: {len(rec_theta_m2_flat)}")
    #print(f"Total reco M1 tracks (2cls): {len(rec_theta_m1_flat_2cls)}")
    #print(f"Total reco M2 tracks (2cls): {len(rec_theta_m2_flat_2cls)}")
    #print(f"Total reco M1 tracks (3cls): {len(rec_theta_m1_flat_3cls)}")
    #print(f"Total reco M2 tracks (3cls): {len(rec_theta_m2_flat_3cls)}")
    
    # DEBUG: Confronta i contatori
    
    print("\n" + "="*50)
    print("DEBUG: CONFRONTO CONTATORI")
    print("="*50)

    #Conta manualmente dai vettori
    manual_reco_m1_2cls = len(rec_theta_m1_flat_2cls)
    manual_reco_m1_3cls = len(rec_theta_m1_flat_3cls)
    manual_reco_m2_2cls = len(rec_theta_m2_flat_2cls)
    manual_reco_m2_3cls = len(rec_theta_m2_flat_3cls)

    print(f"M1 - 2cls: eff_counters={eff_counters['m1']["eff"]["2cl_rec_gen"]} vs vettori={manual_reco_m1_2cls}")
    print(f"M1 - 3cls: eff_counters={eff_counters['m1']["eff"]["3cl_rec_gen"]} vs vettori={manual_reco_m1_3cls}")
    print(f"M2 - 2cls: eff_counters={eff_counters['m2']["eff"]["2cl_rec_gen"]} vs vettori={manual_reco_m2_2cls}")
    print(f"M2 - 3cls: eff_counters={eff_counters['m2']["eff"]["3cl_rec_gen"]} vs vettori={manual_reco_m2_3cls}")

    # Verifica anche i denominatori MC
    manual_mc_2cls = len(gen_theta_flat_2cls)
    manual_mc_3cls = len(gen_theta_flat_3cls)
    print(f"MC - 2cls: eff_counters={eff_counters['mc']['2cl_gen']} vs vettori={manual_mc_2cls}")
    print(f"MC - 3cls: eff_counters={eff_counters['mc']['3cl_gen']} vs vettori={manual_mc_3cls}")
    

    # Create output ROOT file
    #root_out_path = os.path.join(output_dir, f"efficiency_{n_tracks_per_event}_evs.root")
    #f_out = ROOT.TFile(root_out_path, "RECREATE")
    
    
    # PDF file names
    if plot_flag:
        pdf_theta = os.path.join(plot_dir, f"muon_theta_dist_{n_tracks_per_event}_evs.pdf")
        pdf_phi = os.path.join(plot_dir, f"muon_phi_dist_{n_tracks_per_event}_evs.pdf")
        pdf_eff_single = os.path.join(plot_dir, f"muon_efficiencies_vs_angles_single_{n_tracks_per_event}_evs.pdf")
        pdf_eff_comp = os.path.join(plot_dir, f"muon_efficiencies_vs_angles_compare_{n_tracks_per_event}_evs.pdf")
        pdf_eff_single_3cls = os.path.join(plot_dir, f"muon_efficiencies_vs_angles_single_{n_tracks_per_event}_evs_3cls.pdf")
        pdf_eff_comp_3cls = os.path.join(plot_dir, f"muon_efficiencies_vs_angles_compare_{n_tracks_per_event}_evs_3cls.pdf")
        pdf_eff_single_2cls = os.path.join(plot_dir, f"muon_efficiencies_vs_angles_single_{n_tracks_per_event}_evs_2cls.pdf")
        pdf_eff_comp_2cls = os.path.join(plot_dir, f"muon_efficiencies_vs_angles_compare_{n_tracks_per_event}_evs_2cls.pdf")
        pdf_fake_single = os.path.join(plot_dir, f"muon_fake_tracks_vs_angles_single_{n_tracks_per_event}_evs.pdf")
        pdf_fake_single_3cls = os.path.join(plot_dir, f"muon_fake_tracks_vs_angles_single_{n_tracks_per_event}_evs_3cls.pdf")
        pdf_fake_single_2cls = os.path.join(plot_dir, f"muon_fake_tracks_vs_angles_single_{n_tracks_per_event}_evs_2cls.pdf")
        pdf_fake_comp = os.path.join(plot_dir, f"muon_fake_tracks_vs_angles_compare_{n_tracks_per_event}_evs.pdf")
        pdf_fake_comp_3cls = os.path.join(plot_dir, f"muon_fake_tracks_vs_angles_compare_{n_tracks_per_event}_evs_3cls.pdf")
        pdf_fake_comp_2cls = os.path.join(plot_dir, f"muon_fake_tracks_vs_angles_compare_{n_tracks_per_event}_evs_2cls.pdf")
        pdf_ang_comp = os.path.join(plot_dir, f"muon_angular_distributions_comparison_{n_tracks_per_event}_evs.pdf")
        pdf_ang_comp_3cls = os.path.join(plot_dir, f"muon_angular_distributions_comparison_{n_tracks_per_event}_evs_3cls.pdf")
        pdf_ang_comp_2cls = os.path.join(plot_dir, f"muon_angular_distributions_comparison_{n_tracks_per_event}_evs_2cls.pdf")
        pdf_purity_single = os.path.join(plot_dir, f"muon_purity_vs_angles_single_{n_tracks_per_event}_evs.pdf")
        pdf_purity_comp = os.path.join(plot_dir, f"muon_purity_vs_angles_compare_{n_tracks_per_event}_evs.pdf")
        pdf_purity_single_3cls = os.path.join(plot_dir, f"muon_purity_vs_angles_single_{n_tracks_per_event}_evs_3cls.pdf")
        pdf_purity_comp_3cls = os.path.join(plot_dir, f"muon_purity_vs_angles_compare_{n_tracks_per_event}_evs_3cls.pdf")
        pdf_purity_single_2cls = os.path.join(plot_dir, f"muon_purity_vs_angles_single_{n_tracks_per_event}_evs_2cls.pdf")
        pdf_purity_comp_2cls = os.path.join(plot_dir, f"muon_purity_vs_angles_compare_{n_tracks_per_event}_evs_2cls.pdf")

        # Open PDF canvases
        ctmp = ROOT.TCanvas()
        ctmp.Print(pdf_theta + "[")
        ctmp.Print(pdf_phi + "[")
        ctmp.Print(pdf_eff_single + "[")
        ctmp.Print(pdf_eff_comp + "[")
        ctmp.Print(pdf_fake_single + "[")
        ctmp.Print(pdf_fake_comp + "[")
        ctmp.Print(pdf_eff_single_3cls + "[")
        ctmp.Print(pdf_eff_comp_3cls + "[")
        ctmp.Print(pdf_eff_single_2cls + "[")
        ctmp.Print(pdf_eff_comp_2cls + "[")
        ctmp.Print(pdf_fake_single_3cls + "[")
        ctmp.Print(pdf_fake_comp_3cls + "[")
        ctmp.Print(pdf_fake_single_2cls + "[")
        ctmp.Print(pdf_fake_comp_2cls + "[")
        ctmp.Print(pdf_ang_comp + "[")
        ctmp.Print(pdf_ang_comp_3cls + "[")
        ctmp.Print(pdf_ang_comp_2cls + "[")
        ctmp.Print(pdf_purity_single + "[")
        ctmp.Print(pdf_purity_comp + "[")
        ctmp.Print(pdf_purity_single_3cls + "[")
        ctmp.Print(pdf_purity_comp_3cls + "[")
        ctmp.Print(pdf_purity_single_2cls + "[")
        ctmp.Print(pdf_purity_comp_2cls + "[")
        del ctmp
        
        # Create and save histograms for each variable
        for var in ["theta", "phi"]:
            bins = get_bins(var, config)
            
            # Get flattened arrays
            if var == "theta":
                gen_flat = gen_theta_flat
                rec_m1_flat = rec_theta_m1_flat
                rec_m2_flat = rec_theta_m2_flat
                fake_m1_flat = fake_theta_m1
                fake_m2_flat = fake_theta_m2
                pur_m1_flat = pur_theta_m1
                pur_m2_flat = pur_theta_m2
                all_reco_m1_flat = all_reco_theta_m1
                all_reco_m2_flat = all_reco_theta_m2
                gen_flat_3cls = gen_theta_flat_3cls
                rec_m1_flat_3cls = rec_theta_m1_flat_3cls
                rec_m2_flat_3cls = rec_theta_m2_flat_3cls
                gen_flat_2cls = gen_theta_flat_2cls
                rec_m1_flat_2cls = rec_theta_m1_flat_2cls
                rec_m2_flat_2cls = rec_theta_m2_flat_2cls
                fake_m1_flat_3cls = fake_theta_3cl_m1
                fake_m2_flat_3cls = fake_theta_3cl_m2
                fake_m1_flat_2cls = fake_theta_2cl_m1
                fake_m2_flat_2cls = fake_theta_2cl_m2
            else:
                gen_flat = gen_phi_flat
                rec_m1_flat = rec_phi_m1_flat
                rec_m2_flat = rec_phi_m2_flat
                fake_m1_flat = fake_phi_m1
                fake_m2_flat = fake_phi_m2
                pur_m1_flat = pur_phi_m1
                pur_m2_flat = pur_phi_m2
                all_reco_m1_flat = all_reco_phi_m1
                all_reco_m2_flat = all_reco_phi_m2
                gen_flat_3cls = gen_phi_flat_3cls
                rec_m1_flat_3cls = rec_phi_m1_flat_3cls
                rec_m2_flat_3cls = rec_phi_m2_flat_3cls
                gen_flat_2cls = gen_phi_flat_2cls
                rec_m1_flat_2cls = rec_phi_m1_flat_2cls
                rec_m2_flat_2cls = rec_phi_m2_flat_2cls
                rec_m1_flat_2cls = rec_phi_m1_flat_2cls
                rec_m2_flat_2cls = rec_phi_m2_flat_2cls
                fake_m1_flat_3cls = fake_phi_3cl_m1
                fake_m2_flat_3cls = fake_phi_3cl_m2
                fake_m1_flat_2cls = fake_phi_2cl_m1
                fake_m2_flat_2cls = fake_phi_2cl_m2

            # Create histograms using utility function
            
            h_rec_m1, h_rec_m2, h_gen, h_eff_m1, h_eff_m2 = make_efficiency_hist(
                var, rec_m1_flat, rec_m2_flat, gen_flat, bins, n_tracks_per_event
            )

            h_rec_m1_3cls, h_rec_m2_3cls, h_gen_3cls, h_eff_m1_3cls, h_eff_m2_3cls = make_efficiency_hist(
                var, rec_m1_flat_3cls, rec_m2_flat_3cls, gen_flat_3cls, bins, n_tracks_per_event
            )

            h_rec_m1_2cls, h_rec_m2_2cls, h_gen_2cls, h_eff_m1_2cls, h_eff_m2_2cls = make_efficiency_hist(
                var, rec_m1_flat_2cls, rec_m2_flat_2cls, gen_flat_2cls, bins, n_tracks_per_event
            )


            h_fake_ratio_m1, h_fake_ratio_m2 = make_fake_hist(
                var, fake_m1_flat, fake_m2_flat, all_reco_m1_flat, all_reco_m2_flat, bins, n_tracks_per_event
            )

            h_fake_ratio_m1_3cls, h_fake_ratio_m2_3cls = make_fake_hist(
                var, fake_m1_flat_3cls, fake_m2_flat_3cls, rec_m1_flat_3cls, rec_m2_flat_3cls, bins, n_tracks_per_event
            )

            h_fake_ratio_m1_2cls, h_fake_ratio_m2_2cls = make_fake_hist(
                var, fake_m1_flat_2cls, fake_m2_flat_2cls, rec_m1_flat_2cls, rec_m2_flat_2cls, bins, n_tracks_per_event
            )
            
            
            h_purity_m1, h_purity_m2 = make_purity_hist(
                var, pur_m1_flat, pur_m2_flat, all_reco_m1_flat, all_reco_m2_flat, bins, n_tracks_per_event
            )

            h_purity_m1_3cls, h_purity_m2_3cls = make_purity_hist(
                var, fake_m1_flat, fake_m2_flat, rec_m1_flat_3cls, rec_m2_flat_3cls, bins, n_tracks_per_event
            )

            h_purity_m1_2cls, h_purity_m2_2cls = make_purity_hist(
                var, fake_m1_flat, fake_m2_flat, rec_m1_flat_2cls, rec_m2_flat_2cls, bins, n_tracks_per_event
            )

            # Save to ROOT file
            #h_rec_m1.Write()
            #h_rec_m2.Write()
            #h_gen.Write()
            #h_eff_m1.Write()
            #h_eff_m2.Write()
            #h_rec_m1_3cls.Write()
            #h_rec_m2_3cls.Write()
            #h_gen_3cls.Write()
            #h_eff_m1_3cls.Write()
            #h_eff_m2_3cls.Write()
            #h_rec_m1_2cls.Write()
            #h_rec_m2_2cls.Write()
            #h_gen_2cls.Write()
            #h_eff_m1_2cls.Write()
            #h_eff_m2_2cls.Write()
            #h_fake_ratio_m1.Write()
            #h_fake_ratio_m2.Write()
            
            # Determine PDF for this variable
            pdf_main = pdf_theta if var == "theta" else pdf_phi
            
            # Plot distributions
            plot_distributions(var, h_rec_m1, h_rec_m2, h_gen, pdf_main, n_tracks_per_event)
            plot_comparison_distributions(var, h_rec_m1, h_rec_m2, h_gen, pdf_ang_comp, ncls=None)

            plot_comparison_distributions(var, h_rec_m1_3cls, h_rec_m2_3cls, h_gen_3cls, pdf_ang_comp_3cls, ncls=3)
            plot_comparison_distributions(var, h_rec_m1_2cls, h_rec_m2_2cls, h_gen_2cls, pdf_ang_comp_2cls, ncls=2)

            # Plot efficiencies
            plot_efficiencies(var, h_eff_m1, h_eff_m2, pdf_eff_single, pdf_eff_comp, n_tracks_per_event, ncls=None)

            plot_efficiencies(var, h_eff_m1_3cls, h_eff_m2_3cls, pdf_eff_single_3cls, pdf_eff_comp_3cls, n_tracks_per_event, ncls=3)

            plot_efficiencies(var, h_eff_m1_2cls, h_eff_m2_2cls, pdf_eff_single_2cls, pdf_eff_comp_2cls, n_tracks_per_event, ncls=2)
            
            # Plot fake tracks
            plot_fake_tracks(var, h_fake_ratio_m1, h_fake_ratio_m2, pdf_fake_single, pdf_fake_comp, n_tracks_per_event, ncls=None)

            plot_fake_tracks(var, h_fake_ratio_m1_3cls, h_fake_ratio_m2_3cls, pdf_fake_single_3cls, pdf_fake_comp_3cls, n_tracks_per_event, ncls=3)

            plot_fake_tracks(var, h_fake_ratio_m1_2cls, h_fake_ratio_m2_2cls, pdf_fake_single_2cls, pdf_fake_comp_2cls, n_tracks_per_event, ncls=2)

            #Plot purity
            plot_purity(var, h_purity_m1, h_purity_m2, pdf_purity_single, pdf_purity_comp, n_tracks_per_event, ncls=None)

            plot_purity(var, h_purity_m1_3cls, h_purity_m2_3cls, pdf_purity_single_3cls, pdf_purity_comp_3cls, n_tracks_per_event, ncls=3)

            plot_purity(var, h_purity_m1_2cls, h_purity_m2_2cls, pdf_purity_single_2cls, pdf_purity_comp_2cls, n_tracks_per_event, ncls=2)
        
        # Close PDF canvases
        ctmp = ROOT.TCanvas()
        ctmp.Print(pdf_theta + "]")
        ctmp.Print(pdf_phi + "]")
        ctmp.Print(pdf_eff_single + "]")
        ctmp.Print(pdf_eff_comp + "]")
        ctmp.Print(pdf_fake_single + "]")
        ctmp.Print(pdf_fake_comp + "]")
        ctmp.Print(pdf_eff_single_3cls + "]")
        ctmp.Print(pdf_eff_comp_3cls + "]")
        ctmp.Print(pdf_eff_single_2cls + "]")
        ctmp.Print(pdf_eff_comp_2cls + "]")
        ctmp.Print(pdf_ang_comp + "]")
        ctmp.Print(pdf_fake_single_3cls + "]")
        ctmp.Print(pdf_fake_comp_3cls + "]")
        ctmp.Print(pdf_fake_single_2cls + "]")
        ctmp.Print(pdf_fake_comp_2cls + "]")
        ctmp.Print(pdf_ang_comp_3cls + "]")
        ctmp.Print(pdf_ang_comp_2cls + "]")
        ctmp.Print(pdf_purity_single + "]")
        ctmp.Print(pdf_purity_comp + "]")
        ctmp.Print(pdf_purity_single_3cls + "]")
        ctmp.Print(pdf_purity_comp_3cls + "]")
        ctmp.Print(pdf_purity_single_2cls + "]")
        ctmp.Print(pdf_purity_comp_2cls + "]")
        del ctmp
    
    #f_out.Close()
    #print(f"\nOutput ROOT file saved to: {root_out_path}")
    


def plot_distributions(var, h_rec_m1, h_rec_m2, h_gen, pdf_main, n_tracks_per_event):
    """
    Plot distribution histograms.
    """
    ROOT.gStyle.SetOptStat("e")
    
    for hist, hname, color in [
        (h_rec_m1, f"c_{var}_rec_m1", ROOT.kBlue),
        (h_rec_m2, f"c_{var}_rec_m2", ROOT.kRed),
        (h_gen, f"c_{var}_gen", ROOT.kGreen + 2)
    ]:
        c = ROOT.TCanvas(hname, "", 800, 600)
        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        hist.Draw("HIST")
        c.Update()
        
        # Adjust stats box
        st = hist.FindObject("stats")
        if st:
            st.SetTextSize(0.04)
            st.SetX1NDC(0.78)
            st.SetX2NDC(0.95)
            st.SetY1NDC(0.84)
            st.SetY2NDC(0.95)
        
        c.Print(pdf_main)
        c.Close()


def plot_comparison_distributions(var, h_rec_m1, h_rec_m2, h_gen, pdf_comp, ncls=None):
    """
    Plot comparison of generated and reconstructed angular distributions.
    """

    c = ROOT.TCanvas(f"c_comp_{var}", "", 800, 600)

    ROOT.gStyle.SetOptStat(0)

    h_gen.SetLineColor(ROOT.kGreen + 2)
    h_gen.SetLineWidth(2)

    h_rec_m1.SetLineColor(ROOT.kBlue)
    h_rec_m1.SetLineWidth(2)

    h_rec_m2.SetLineColor(ROOT.kRed)
    h_rec_m2.SetLineWidth(2)

    ymax = max(
        h_gen.GetMaximum(),
        h_rec_m1.GetMaximum(),
        h_rec_m2.GetMaximum()
    )

    if ncls == 2 and var == "theta":
        h_gen.SetMaximum(200)
    elif ncls == 2 and var == "phi":
        h_gen.SetMaximum(120)
    else:
        h_gen.SetMaximum(1.2 * ymax)

    if ncls is None:
        ncls_label = "all tracks"
    else:        
        ncls_label = f"{ncls} cls"

    h_gen.SetTitle(f"Angular distribution comparison - #{var} - {ncls_label}")

    h_gen.Draw("HIST")
    h_rec_m1.Draw("HIST SAME")
    h_rec_m2.Draw("HIST SAME")

    leg = ROOT.TLegend(0.53, 0.73, 0.90, 0.90)
    leg.SetTextSize(0.035)
    leg.AddEntry(
    h_gen,
        f"Generated (Entries: {int(h_gen.GetEntries())})",
        "l"
    )

    leg.AddEntry(
        h_rec_m1,
        f"Reco M1 (Entries: {int(h_rec_m1.GetEntries())})",
        "l"
    )

    leg.AddEntry(
        h_rec_m2,
        f"Reco M2 (Entries: {int(h_rec_m2.GetEntries())})",
        "l"
    )
    leg.Draw()

    c.Print(pdf_comp)
    c.Close()


def plot_efficiencies(var, h_eff_m1, h_eff_m2, pdf_eff_single, pdf_eff_comp, n_tracks_per_event, ncls=None):
    """
    Plot efficiency histograms.
    """
    # Single efficiency plots
    for hist, hname, color in [
        (h_eff_m1, f"c_eff_{var}_{ncls}_m1", ROOT.kBlue),
        (h_eff_m2, f"c_eff_{var}_{ncls}_m2", ROOT.kRed),
    ]:
        hist.SetStats(0)
        c = ROOT.TCanvas(hname, "", 800, 600)
        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        hist.SetMinimum(0)
        hist.SetMaximum(hist.GetMaximum() * 1.15)
        hist.Draw("E1")
        c.Print(pdf_eff_single)
        c.Close()
    
    # Comparison plot
    c = ROOT.TCanvas(f"c_eff_{var}_{ncls}_compare", "", 800, 600)
    if ncls is None:
        ncls_label = "all tracks"
    else:        
        ncls_label = f"{ncls} cls"
    h_eff_m1.SetTitle(f"Efficiency vs #{var} - {n_tracks_per_event} tracks per event - {ncls_label}")
    h_eff_m1.SetLineColor(ROOT.kBlue)
    h_eff_m2.SetLineColor(ROOT.kRed)
    
    ymax = max(h_eff_m1.GetMaximum(), h_eff_m2.GetMaximum())
    
    h_eff_m1.SetMinimum(0.0)
    h_eff_m1.SetMaximum(1.15 * ymax)
    h_eff_m1.Draw("E1")
    h_eff_m2.Draw("E1 SAME")
    
    leg = ROOT.TLegend(0.73, 0.79, 0.90, 0.90)
    leg.AddEntry(h_eff_m1, "Hough", "lep")
    leg.AddEntry(h_eff_m2, "Comb.", "lep")
    leg.SetTextSize(0.04)
    leg.SetFillStyle(1001)
    leg.SetFillColor(ROOT.kWhite)
    leg.Draw()
    
    c.Print(pdf_eff_comp)
    c.Close()


def plot_fake_tracks(var, h_fake_ratio_m1, h_fake_ratio_m2, pdf_fake_single, pdf_fake_comp, n_tracks_per_event, ncls=None):
    """
    Plot fake track ratio histograms.
    """
    # Single fake track ratio plots
    for hist, hname, color in [
        (h_fake_ratio_m1, f"c_fake_ratio_{var}_{ncls}_m1", ROOT.kBlue),
        (h_fake_ratio_m2, f"c_fake_ratio_{var}_{ncls}_m2", ROOT.kRed),
    ]:
        hist.SetStats(0)
        c = ROOT.TCanvas(hname, "", 800, 600)
        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        hist.SetMinimum(0)
        hist.SetMaximum(hist.GetMaximum() * 1.15)
        hist.Draw("E1")
        c.Print(pdf_fake_single)
        c.Close()
    
    # Comparison plot
    c = ROOT.TCanvas(f"c_fake_ratio_{var}_{ncls}_compare", "", 800, 600)
    if ncls is None:
        ncls_label = "all tracks"
    else:        
        ncls_label = f"{ncls} cls"
    h_fake_ratio_m1.SetTitle(f"Fake Tracks Ratio vs #{var} - {n_tracks_per_event} tracks per event - {ncls_label}")
    h_fake_ratio_m1.SetLineColor(ROOT.kBlue)
    h_fake_ratio_m2.SetLineColor(ROOT.kRed)
    
    ymax = max(h_fake_ratio_m1.GetMaximum(), h_fake_ratio_m2.GetMaximum())
    
    h_fake_ratio_m1.SetMinimum(0.0)
    h_fake_ratio_m1.SetMaximum(ymax * 1.5)
    h_fake_ratio_m1.Draw("E1")
    h_fake_ratio_m2.Draw("E1 SAME")
    
    leg = ROOT.TLegend(0.73, 0.79, 0.90, 0.90)
    leg.AddEntry(h_fake_ratio_m1, "Hough", "lep")
    leg.AddEntry(h_fake_ratio_m2, "Comb.", "lep")
    leg.SetTextSize(0.04)
    leg.SetFillStyle(1001)
    leg.SetFillColor(ROOT.kWhite)
    leg.Draw()
    
    c.Print(pdf_fake_comp)
    c.Close()

def plot_purity(var, h_purity_m1, h_purity_m2, pdf_purity_single, pdf_purity_comp, n_tracks_per_event, ncls=None):
    """
    Plot purity histograms.
    """
    # Single purity plots
    for hist, hname, color in [
        (h_purity_m1, f"c_purity_{var}_{ncls}_m1", ROOT.kBlue),
        (h_purity_m2, f"c_purity_{var}_{ncls}_m2", ROOT.kRed),
    ]:
        hist.SetStats(0)
        c = ROOT.TCanvas(hname, "", 800, 600)
        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        hist.SetMinimum(0)
        hist.SetMaximum(hist.GetMaximum() * 1.15)
        hist.Draw("E1")
        c.Print(pdf_purity_single)
        c.Close()
    
    # Comparison plot
    c = ROOT.TCanvas(f"c_purity_{var}_{ncls}_compare", "", 800, 600)
    if ncls is None:
        ncls_label = "all tracks"
    else:        
        ncls_label = f"{ncls} cls"
    h_purity_m1.SetTitle(f"Purity vs #{var} - {n_tracks_per_event} tracks per event - {ncls_label}")
    h_purity_m1.SetLineColor(ROOT.kBlue)
    h_purity_m2.SetLineColor(ROOT.kRed)
    
    ymax = max(h_purity_m1.GetMaximum(), h_purity_m2.GetMaximum())
    
    h_purity_m1.SetMinimum(0.0)
    h_purity_m1.SetMaximum(1.15 * ymax)
    h_purity_m1.Draw("E1")
    h_purity_m2.Draw("E1 SAME")
    
    leg = ROOT.TLegend(0.73, 0.79, 0.90, 0.90)
    leg.AddEntry(h_purity_m1, "Hough", "lep")
    leg.AddEntry(h_purity_m2, "Comb.", "lep")
    leg.SetTextSize(0.04)
    leg.SetFillStyle(1001)
    leg.SetFillColor(ROOT.kWhite)
    leg.Draw()
    
    c.Print(pdf_purity_comp)
    c.Close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get efficiency plots as function of theta and phi angles.")
    parser.add_argument("--input-file", required=True, help="Use this .root file.")
    parser.add_argument("--output-dir", default=None, help="Define output directory.")
    parser.add_argument("--plot", action="store_true", default=False, help="Whether to produce plots. Default is False.")
    #parser.add_argument("--cls", default=None, help="Define track selection based on number of clusters. Default is None (no selection). Possible values are 2 or 3.")
    args = parser.parse_args()

    #if not args.cls == 3 and not args.cls == 2:
    #    print(f"Invalid --cls value: {args.cls}. Must be 2 or 3.")
    #    exit(1)
    
    config_file = "config/config_efficiency.yaml"
    with open(config_file, "r") as f_cfg:
        config = yaml.safe_load(f_cfg)
    
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = config["output_dir"]
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    log_dir = os.path.join(output_dir, "logs")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    if args.plot:
        plot_flag = True
    else:
        plot_flag = False
    
    compute_efficiency(args.input_file, output_dir, config, plot_flag)