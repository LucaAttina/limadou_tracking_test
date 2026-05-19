# utils/efficiency_utils.py
import numpy as np
import awkward as ak
import ROOT


def compute_gen_mask(tree_gen, gen_mask):
    """
    Consider event if at least 2 cluster with size > 0 and the tracks hit all triggers.
    """
    tr1x = tree_gen["TR1_posX"].array()
    tr1y = tree_gen["TR1_posY"].array()
    tr2x = tree_gen["TR2_posX"].array()
    tr2y = tree_gen["TR2_posY"].array() 
    csize = tree_gen["csize"].array()  
    cl_layer = tree_gen["cl_layer"].array()
    cls_idx = tree_gen["cls_to_trk_idx"].array()

    for i, (x1, y1, x2, y2, c, lay, cid) in enumerate(zip(tr1x, tr1y, tr2x, tr2y, csize, cl_layer, cls_idx)):

        # clusters condition
        if ak.count_nonzero(c, axis=0) < 2:
            gen_mask.append(False)
            continue
            
        # TR conditions
        if not ak.any(x1 != -999, axis=0):
            gen_mask.append(False)
            continue

        if not ak.any(y1 != -999, axis=0):
            gen_mask.append(False)
            continue
            
        if not ak.any(x2 != -999, axis=0):
            gen_mask.append(False)
            continue
            
        if not ak.any(y2 != -999, axis=0):
            gen_mask.append(False)
            continue

        if len(set(tuple(lay))) < 2:
            print(f"{lay}")
            gen_mask.append(False)
            continue

        mc_cls_num = sum(1 for x in cid if x != -999)
        if mc_cls_num < 2:
            print(f"mc_cls_num: {mc_cls_num}")
            gen_mask.append(False)
            continue 
       
        
        # if all conditions are true
        gen_mask.append(True)   
    
    return gen_mask



def get_bins(var_name, config):
    
    # Case 1: explicit list of theta bins
    if f"{var_name}_bins" in config:
        bins = config[f"{var_name}_bins"]
        return bins

    # Case 2: define bins by min, max, and step
    elif all(key in config for key in [f"{var_name}_min", f"{var_name}_max", f"{var_name}_step"]):
        bins = []
        val = config[f"{var_name}_min"]
        while val <= config[f"{var_name}_max"]:
            bins.append(val)
            val += config[f"{var_name}_step"]
        return bins

    else:
        raise ValueError(f"Invalid configuration: either '{var_name}_bins' or '{var_name}_min', '{var_name}_max', and '{var_name}_step' must be defined.")



def initialize_counters(methods, counter_types="efficiency"):
    """
    Initialize counters dictionary for multiple methods.
    
    Args:
        methods: list of method names (e.g., ["m1", "m2"])
        counter_types: "efficiency" for efficiency calculations, "delta" for delta distributions
    
    Returns:
        Dictionary with counters for each method
    """
    counters = {}
    
    if counter_types == "efficiency":
        base_counters = {
            "total_gen_good": 0,          # Denominator: events with gen_mask true
            "reconstructed_good": 0,      # Numerator: gen good AND well reconstructed
            "mult_0": 0,                  # Tracks with multiplicity = 0 (good generated)
            "not_reconstructed": 0,       # Total good generated - reconstructed ones
            "bad_mult": 0,                # Multiplicity > 1
            "issues": 0,                  # Tracks with issues (Dsum, missing_in_acceptance...)
            "angle_mismatch": 0,          # Reconstructed but with wrong angles
            "clusters_not_matching": 0,   # Generated but not reconstructed (clusters don't match)
        }
    elif counter_types == "delta":
        base_counters = {
            "tracks": 0,
            "theta_great_out": 0,
            "theta_low_out": 0,
            "phi_great_out": 0,
            "phi_low_out": 0,
        }
    else:
        base_counters = {}
    
    # generate counters dictionaries for both methods
    for method in methods:
        counters[method] = base_counters.copy()
    
    return counters



def check_angle_mismatch(rec_theta, rec_phi, mc_theta, mc_phi, th_threshold=None, ph_threshold=None):
    """
    Check if there is an angle mismatch compared to MC truth.
    
    Args:
        rec_theta, rec_phi: reconstructed angles
        mc_theta, mc_phi: MC truth angles
        th_threshold: threshold for theta mismatch in degrees
        ph_threshold: threshold for phi mismatch in degrees
    
    Returns:
        bool: True if angle mismatch exceeds threshold
    """
    if mc_theta is None or mc_phi is None or mc_theta == -999 or mc_phi == -999:
        return False
    
    diff_theta = rec_theta - mc_theta
    diff_phi = (rec_phi - mc_phi + 180) % 360 - 180
    
    # comparison with thresholds
    return abs(diff_theta) > th_threshold, abs(diff_phi) > ph_threshold, diff_theta, diff_phi



def read_data_from_file(file):
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
    gen_theta = trees["gen"]["gen_theta"].array()
    gen_phi = trees["gen"]["gen_phi"].array()
    p_id = trees["gen"]["particle_id"].array()
    
    #print(f"  - gen_theta type: {type(gen_theta)}")
    #print(f"  - gen_theta length: {len(gen_theta)}")
    #if len(gen_theta) > 0:
    #    print(f"  - gen_theta[0] type: {type(gen_theta[0])}")
    #    print(f"  - gen_theta[0] length: {len(gen_theta[0]) if hasattr(gen_theta[0], '__len__') else 'scalar'}")
    #    print(f"  - gen_theta[0]: {gen_theta[0]}")
    
    # Cluster MC
    gen_cls_x = trees["gen"]["x_abspos"].array()
    gen_cls_y = trees["gen"]["y_abspos"].array()
    gen_cls_layer = trees["gen"]["cl_layer"].array()
    gen_cls_to_trk = trees["gen"]["cls_to_trk_idx"].array()
    
    #print(f"  - gen_cls_x length: {len(gen_cls_x)}")
    #if len(gen_cls_x) > 0:
    #    print(f"  - gen_cls_x[0] type: {type(gen_cls_x[0])}")
    #    print(f"  - gen_cls_x[0] length: {len(gen_cls_x[0]) if hasattr(gen_cls_x[0], '__len__') else 'scalar'}")
    #    print(f"  - gen_cls_x[0][:5]: {gen_cls_x[0][:5] if hasattr(gen_cls_x[0], '__getitem__') else gen_cls_x[0]}")
    
    # Determina n_tracks_per_event
    n_tracks_per_event = 0
    if len(gen_theta) > 0:
        if hasattr(gen_theta[0], '__len__'):
            n_tracks_per_event = len(gen_theta[0])
        else:
            n_tracks_per_event = 1  # scalare = 1 traccia
    #print(f"  - n_tracks_per_event: {n_tracks_per_event}")
    
    # ========================================
    # RECONSTRUCTED DATA - M1
    # ========================================
    
    #print(f"\n[DEBUG] M1 (SelectedEvents) tree:")
    
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
    
    # ========================================
    # RETURN
    # ========================================
    
    return {
        "rec_data": {
            "m1": {
                "theta": rec_m1_theta,
                "phi": rec_m1_phi,
                "event_idx": rec_m1_event_idx,
                "d_sum": d_sum_m1,
                "cls": {"x": rec_m1_cls_x, "y": rec_m1_cls_y, "z": rec_m1_cls_z}
            },
            "m2": {
                "theta": rec_m2_theta,
                "phi": rec_m2_phi,
                "event_idx": rec_m2_event_idx,
                "d_sum": d_sum_m2,
                "cls": {"x": rec_m2_cls_x, "y": rec_m2_cls_y, "z": rec_m2_cls_z}
            }
        },
        "gen_data": {
            "trk_id": p_id,
            "theta": gen_theta,
            "phi": gen_phi,
            "cls": {
                "x": gen_cls_x,
                "y": gen_cls_y,
                "layer": gen_cls_layer,
                "cls_to_trk": gen_cls_to_trk
            }
        },
        "mult_data": mult_data,
        "n_events": n_events,
        "n_tracks_per_event": n_tracks_per_event,
        "n_tracks": {             
            "mc": n_tracks_mc,
            "m1": n_tracks_m1,
            "m2": n_tracks_m2
        }
    }



def is_track_reconstructed(ev_idx, method, rec_cl_x, rec_cl_y, rec_cl_z, mc_cl_x, mc_cl_y, mc_cl_layer, 
                           cls_to_trk, n_tracks, reco_trk_idx, dump=None, tol=1.0, debug=False):
    """
    Versione con assegnazione esclusiva dei cluster ricostruiti.
    Ogni cluster ricostruito può essere usato al massimo una volta.
    """
    #if ev_idx == 0 or ev_idx == 100 or ev_idx == 158 or ev_idx == 562 or ev_idx == 573:
    #    debug = True
    if dump is not None:
        #print("\n" + "="*80)
        #print(f"EV IDX: {ev_idx} - Method: {method}")
        #print("="*80)
        dump.write("\n--- 1: Reconstructed clusters list ---")

    
    # Organizza cluster ricostruiti
    reco_clusters = []
    for n_tr in range(len(reco_trk_idx)):
        if dump is not None:
            dump.write(f"\n  Reconstructed track {n_tr}:")
        for i, (cx, cy, cz) in enumerate(zip(rec_cl_x[n_tr], rec_cl_y[n_tr], rec_cl_z[n_tr])):
            if cx == -999 or cy == -999 or cz == -999:
                if dump is not None:
                    dump.write(f"\n    Cluster {i}: ({cx}, {cy}, {cz}) -> SKIP (value: -999)")
                continue
            if abs(cz - 17.825) < 1e-3:
                lay = 0
            elif abs(cz - 26.325) < 1e-3:
                lay = 1 
            elif abs(cz - 34.825) < 1e-3:
                lay = 2
            else:
                if dump is not None:
                    dump.write(f"\n    Cluster {i}: ({cx:.3f}, {cy:.3f}, {cz:.3f}) -> SKIP (layer uknown)")
                continue
            if dump is not None:
                    dump.write(f"\n    Cluster {i}: ({cx:.3f}, {cy:.3f}, {cz:.3f}) -> layer {lay}")
            reco_clusters.append((int(n_tr), cx, cy, lay))
    if dump is not None:
        dump.write(f"\nTotal valid reconstructed clusters: {len(reco_clusters)}")
    
        # Organizza cluster MC per traccia
        dump.write("\n\n--- 2: MC clusters list ---")
    mc_by_track = {}
    for track_id in range(n_tracks):
        mc_indices = [i for i, trk in enumerate(cls_to_trk) if trk == track_id]
        mc_by_track[track_id] = mc_indices
        if mc_indices:
            if dump is not None:
                dump.write(f"\n  MC track {track_id}: {len(mc_indices)} cluster (indices: {mc_indices})")
            for mc_idx in mc_indices:
                if dump is not None:
                    dump.write(f"\n    MC cluster {mc_idx}: layer={mc_cl_layer[mc_idx]}, x={mc_cl_x[mc_idx]:.3f}, y={mc_cl_y[mc_idx]:.3f}")
        else:
            if dump is not None:
                dump.write(f"\n  MC track {track_id}: no clusters")
    
    results = [False] * n_tracks
    used_reco = set()
    used_mc_global = set()
    reco_to_gen = {track_id: None for track_id in range(n_tracks)} # dict in the form gen_id: reco_id
    if dump is not None:
        dump.write("\n\n--- 3: Matching clusters ---")
        #dump.write("-"*80)
    
    for track_id in range(n_tracks):
        if dump is not None:
            dump.write(f"\n>>> MC track {track_id} <<<")
        mc_indices = mc_by_track[track_id]
        
        if len(mc_indices) < 2:
            if dump is not None:
                dump.write(f"\n  -> SKIP: only {len(mc_indices)} cluster MC (>2 needed)")
            continue

        if dump is not None:
            dump.write(f"\n  MC cluster to be matched: {mc_indices}")
        temp_used_mc = set()
        temp_used_reco = set()
        n_matched = 0
        matched_reco_track_idx = None
        matched_pairs = []  # Per memorizzare i match trovati
        
        for reco_idx, (n_tr, cx, cy, lay) in enumerate(reco_clusters):
            if reco_idx in used_reco:
                #print(f"  Cluster ricostruito {reco_idx} (track {n_tr}) -> già usato in precedenza")
                continue
            
            for mc_idx in mc_indices:
                if mc_idx in used_mc_global or mc_idx in temp_used_mc:
                    continue
                
                # Verifica match
                layer_match = (lay == mc_cl_layer[mc_idx])
                dist = (mc_cl_x[mc_idx] - cx) * (mc_cl_x[mc_idx] - cx) + (mc_cl_y[mc_idx] - cy) * (mc_cl_y[mc_idx] - cy)
                r_match = dist <= tol 
                #x_match = abs(mc_cl_x[mc_idx] - cx) <= tol
                #y_match = abs(mc_cl_y[mc_idx] - cy) <= tol
                
                if layer_match and r_match:
                    if dump is not None:
                        dump.write(f"\n  ✓ MATCH FOUND!")
                        dump.write(f"\n    Reconstructed: idx={reco_idx}, track={n_tr}, layer={lay}, x={cx:.3f}, y={cy:.3f}")
                        dump.write(f"\n    MC: idx={mc_idx}, layer={mc_cl_layer[mc_idx]}, x={mc_cl_x[mc_idx]:.3f}, y={mc_cl_y[mc_idx]:.3f}")
                        #print(f"    Distanze: dx={abs(mc_cl_x[mc_idx]-cx):.3f}, dy={abs(mc_cl_y[mc_idx]-cy):.3f}")
                        dump.write(f"\n    Distance: r={abs(dist):.3f}")

                    temp_used_mc.add(mc_idx)
                    temp_used_reco.add(reco_idx)
                    matched_reco_track_idx = n_tr
                    matched_pairs.append((reco_idx, mc_idx, n_tr, lay))
                    n_matched += 1
                    break
        
        if dump is not None:
            dump.write(f"\n\n  Results for MC track {track_id}: {n_matched} match / {len(mc_indices)} MC clusters")
        #if matched_pairs:
        #    if dump is not None:
        #        dump.write(f"  Match specifici: {matched_pairs}")
        
        if n_matched >= 2:
            results[track_id] = True
            if matched_reco_track_idx is not None:
                reco_to_gen[track_id] = matched_reco_track_idx
            
            used_reco.update(temp_used_reco)
            used_mc_global.update(temp_used_mc)
            if dump is not None:
                dump.write(f"\n  -> MC track {track_id} RECONSTRUCTED (associated to reconstructed track {matched_reco_track_idx})")
                dump.write(f"\n     MC clusters used: {used_mc_global}")
                dump.write(f"\n     Reconstructed clusters used: {used_reco}")
        else:
            if dump is not None:
                dump.write(f"\n  -> MC track {track_id} NOT RECONSTRUCTED (only {n_matched} match, at least 2 needed)")
    if dump is not None:
        dump.write("\n\n" + "="*10)
        dump.write(" FINAL RESULTS ")
        dump.write("="*10)

        dump.write("\n\n--- MC tracks reconstructed ---")
    for track_id in range(n_tracks):
        status = "✓ RECONSTRUCTED" if results[track_id] else "✗ NOT RECONSTRUCTED"
        reco_track = reco_to_gen[track_id] if reco_to_gen[track_id] is not None else "N/A"
        if dump is not None:
            dump.write(f"\n  MC track {track_id}: {status} -> associated to reconstructed track {reco_track}")
    
    if dump is not None:
        dump.write("\n\n--- Match per cluster summary ---")
        dump.write("\nMC clusters used globally: " + (str(sorted(used_mc_global)) if used_mc_global else "none"))
        dump.write("\nReconstructed clusters used globally: " + (str(sorted(used_reco)) if used_reco else "none"))
    
    #print("\n" + "="*80)
    #print("FINE ANALISI")
    #print("="*80 + "\n")
    
    return results, reco_to_gen


def process_efficiency_event(n_tracks_per_event, ev_idx, reco_trk_idx, gen_trk_idx, mult_value, rec_cl_x, rec_cl_y, rec_cl_z,
                            mc_cl_x, mc_cl_y, mc_cl_layer, cls_to_trk, rec_theta, rec_phi,
                            gen_theta, gen_phi, d_sum, 
                            eff_info, rec_info, eff_counters,
                            theta_angle_threshold=5.0, phi_angle_threshold=5.0, return_good_flag=False, return_delta=False, method="m1", dump=None):
    """
    Process a single event for tracking efficiency calculation.
    
    Args:
        ev_idx: event index (and position for generated tracks)
        mult_value: multiplicity for this method
        event_idx_list: arrays containing indeces of reconstructed tracks
        cl_x_arr, cl_y_arr, cl_z_arr: reconstructed clusters coordinates
        mc_x, mc_y, mc_layer: MC truth clusters coordinates
        rec_theta, rec_phi: reconstructed angles arrays
        gen_theta, gen_phi: generated angles arrays
        counter: counters dictionary for this method
        theta_angle_threshold, phi_angle_treshold: threshold for angle mismatch in degrees
        return_good_flag: if True, return boolean for good reconstruction
    
    Returns:
        If return_good_flag: bool indicating if track is good for efficiency
        Otherwise: None
    """

    theta_angle_mismatch = np.zeros(n_tracks_per_event, dtype=bool)
    phi_angle_mismatch = np.zeros(n_tracks_per_event, dtype=bool)
    diff_theta = np.zeros(n_tracks_per_event, dtype=float)
    diff_phi = np.zeros(n_tracks_per_event, dtype=float)
    angle_ok = np.zeros(n_tracks_per_event, dtype=bool)
    
    n_fake = 0
    fake_id = []

    if mult_value == 0:
        if dump is not None:
            dump.write("\n\n" + "="*20)
            dump.write(f" Method {method} ")
            dump.write("="*20)
            dump.write(f"No reconstructed tracks (multiplicity=0)\n\n")
 
        mc_reconstructed = [False] * n_tracks_per_event
        #print(f"Event {ev_idx} - method {method}: SKIP")
        #for trk_idx in range(n_tracks_per_event):
            #eff_info[method][ev_idx][trk_idx]
        #return None
        diff_theta_list = np.array([])
        diff_phi_list = np.array([])
        gen_theta_list = np.array([])
        gen_phi_list = np.array([])
        
        # Inizializza diff_theta e diff_phi con NaN
        diff_theta[:] = np.nan
        diff_phi[:] = np.nan
        
    else:
        #found_good_track = False
        #for n_trk in mult_value:
        #    if found_good_track:
        #        break
        #print(f"\nProcessing event {ev_idx} - method {method} - multiplicity: {mult_value}")
        debug = False
        #if mult_value > 1:
        #    debug = True
        #print(f"{reco_trk_idx}")
        if dump is not None:
            dump.write("\n\n" + "="*20)
            dump.write(f" Method {method} ")
            dump.write("="*20 + "\n")

        mc_reconstructed, reco_to_gen = is_track_reconstructed(
            ev_idx, method, rec_cl_x, rec_cl_y, rec_cl_z, 
            mc_cl_x, mc_cl_y, mc_cl_layer, cls_to_trk, 
            n_tracks_per_event, reco_trk_idx, dump, debug=debug
        )
            
        #print(method)
        gen_id = []
        rec_id = []
        #eff_counters[method]["tot_reco"] += mult_value
        
        #mc_cls = sum(1 for x in cls_to_trk if x != -999)
        #eff_counters["mc"]["tot"] += 1
        #if mc_cls == 2:
        #    eff_counters["mc"]["2cl"] += 1
        #elif mc_cls == 3:
        #    eff_counters["mc"]["3cl"] += 1
        #else:
        #    print(f"ev_ {ev_idx}: WARNING: {mc_cls}")
        #print(f"reco_trk_idx: {reco_trk_idx}")
        #print(f"mult: {mult_value}")


        '''
        for gen, rec in reco_to_gen.items():
            gen_id.append(gen)
            rec_id.append(rec)
        
        #print(f"gen-rec: {gen_id} - {rec_id}")
        for ntr in range(mult_value):
            #print(f"rec cl x: {rec_cl_x[ntr]}")

            if len(rec_cl_x[ntr]) == 3:
                eff_counters[method]["reco_3cl"] += 1
                if ntr in rec_id:
                    eff_counters[method]["3cl"]["reco"] += 1
                    eff_counters[method]["tot"]["reco"] += 1
                    rec_info[method][ev_idx][ntr] = True

                else: 
                    eff_counters[method]["3cl"]["fake"] += 1
                    eff_counters[method]["tot"]["fake"] += 1
                    #print(f"ev: {ev_idx} angle 3cl fake: {rec_theta[ntr]} - {rec_phi[ntr]}")
            else:
                eff_counters[method]["reco_2cl"] += 1
                if ntr in rec_id:
                    eff_counters[method]["2cl"]["reco"] += 1
                    eff_counters[method]["tot"]["reco"] += 1
                    rec_info[method][ev_idx][ntr] = True
                else: 
                    eff_counters[method]["2cl"]["fake"] += 1
                    eff_counters[method]["tot"]["fake"] += 1
                    #print(f"ev: {ev_idx} angle 2cl fake: {rec_theta[ntr]} - {rec_phi[ntr]}")

                '''
        
        matched_reco_id = set(reco_to_gen.values())

        for trk_idx in range(n_tracks_per_event):
            if mc_reconstructed[trk_idx]:
                eff_counters[method]["eff"]["tot_rec_gen"] += 1
                mc_cls_m = sum(1 for x in cls_to_trk if x == trk_idx)
                if mc_cls_m == 2:
                    eff_counters[method]["eff"]["2cl_rec_gen"] += 1
                elif mc_cls_m == 3:
                    eff_counters[method]["eff"]["3cl_rec_gen"] += 1


        for reco_id in range(mult_value):
            n_cls = len([c for c in rec_cl_x[reco_id] if c != -999])
            eff_counters[method]["fake"]["tot_rec"] += 1
            if n_cls == 3:
                eff_counters[method]["fake"]["3cl_rec"] += 1
            else:
                eff_counters[method]["fake"]["2cl_rec"] += 1

            if reco_id in matched_reco_id:
                rec_info[method][ev_idx][reco_id] = True
            else:
                eff_counters[method]["fake"]["tot_fake"] += 1
                if n_cls == 3:
                    eff_counters[method]["fake"]["3cl_fake"] += 1
                else:
                    eff_counters[method]["fake"]["2cl_fake"] += 1
                
                rec_info[method][ev_idx][reco_id] = False
            
        #print(f"RECO INFO ev {ev_idx} - {method}: {rec_info[method][ev_idx]}")



        #print(f"reco_3cl - {method}: {eff_counters[method]["3cl"]["reco"]}")
        #print(f"fake_3cl - {method}: {eff_counters[method]["3cl"]["fake"]}")
        #print(f"reco_2cl - {method}: {eff_counters[method]["2cl"]["reco"]}")
        #print(f"fake_2cl - {method}: {eff_counters[method]["2cl"]["fake"]}")  
        #print(f"mc_3cl - {method}: {eff_counters["mc"]["3cl"]}")      
        #print(f"mc_2cl - {method}: {eff_counters["mc"]["2cl"]}")   
        #print(f"mc_tot - {method}: {eff_counters["mc"]["tot"]}")   



        total_reco_tracks = len(reco_trk_idx)
        #print(f"reco_m2_idx = {reco_trk_idx}")
        #match_reco_id = set()
        #for gen_id, reco_id in reco_to_gen.items():
        #    if reco_id is not None:
        #        match_reco_id.add(reco_id)
#
        #all_reco_id = set(range(total_reco_tracks))
        #fake_id = list(all_reco_id - match_reco_id)

        diff_theta_list = []
        diff_phi_list = []
        gen_theta_list = []
        gen_phi_list = []

        for trk_idx in range(n_tracks_per_event):
            if mc_reconstructed[trk_idx]:
                r_id = reco_to_gen[trk_idx]

                track_theta = rec_theta[r_id]
                track_phi = rec_phi[r_id]
                mc_theta_val = gen_theta[trk_idx]
                mc_phi_val = gen_phi[trk_idx]

                #if ev_idx == 4088:
#
                #    print(f"rec theta: {track_theta}")
                #    print(f"gen theta: {mc_theta_val}")
                #    print(f"rec phi: {track_phi}")
                #    print(f"gen phi: {mc_phi_val}")

                    
                theta_angle_mismatch[trk_idx], phi_angle_mismatch[trk_idx], diff_theta[trk_idx], diff_phi[trk_idx] = check_angle_mismatch(
                    track_theta, track_phi, mc_theta_val, mc_phi_val, 
                    theta_angle_threshold, phi_angle_threshold
                )


                #if ev_idx == 4088:
#
                #    print(f"diff theta: {diff_theta[trk_idx]}")
                #    print(f"diff phi: {diff_phi[trk_idx]}\n")
                

                diff_theta_list.append(diff_theta[trk_idx])
                diff_phi_list.append(diff_phi[trk_idx])
                gen_theta_list.append(mc_theta_val)
                gen_phi_list.append(mc_phi_val)

                #print(f"1: diff theta: {diff_theta_list}")
                #print(f"2: gen_theta: {gen_theta_list}")
                #print(f"3: gen_phi: {gen_phi_list}")
                #print(f"4: diff_phi: {diff_phi_list}\n")

                #if theta_angle_mismatch[trk_idx] or phi_angle_mismatch[trk_idx]:
                #    print(f"evt {ev_idx}: diff_theta = {diff_theta[trk_idx]}, diff_phi = {diff_phi[trk_idx]} - method {method}")

                #print(f"theta_mismatch: {diff_theta[trk_idx]}, phi_mismatch: {diff_phi[trk_idx]}")
                
                if not theta_angle_mismatch[trk_idx] and not phi_angle_mismatch[trk_idx]:       
                    angle_ok[trk_idx] = False
                
                    #counter["angle_mismatch"] += 1
                #else:
                    #counter["reconstructed_good"] += 1
                
            #else:
             #   angle_ok[trk_idx] = False
                #counter["not_reconstructed"] += 1

        #n_not_reconstructed = n_tracks_per_event - np.sum(mc_reconstructed)
                    
                        # Track correctly reconstructed
                        #found_good_track = True
                        #is_correctly_reconstructed = True
    for trk_idx in range(n_tracks_per_event):
        eff_info[method][ev_idx][trk_idx] = mc_reconstructed[trk_idx] # and angle_ok[trk_idx] 

    #print(f"ev: {ev_idx} - {method}     {eff_info[method][ev_idx]}")  
    #print(f"{mc_reconstructed}")
    #print(f"1: {angle_ok}")
    #print(f"2: diff theta: {diff_theta_list}")
    #print(f"3: gen_theta: {gen_theta_list}")
    #print(f"4: gen_phi: {gen_phi_list}")
    #print(f"5: diff_phi: {diff_phi_list}\n")
    
    #n_fake = len(reco_trk_idx) - np.sum(eff_info[method][ev_idx])
    n_matched = len([reco_id for reco_id in reco_to_gen.values() if reco_id is not None])
    #print(f"n_matched: {n_matched}")
    #if n_matched == 0:
    #    print(f"EV: {ev_idx} - mult: {mult_value} - 0 matched")
    #    print(f"rec cl x: {rec_cl_x}")
    #    print(f"mc cl x: {mc_cl_x}")
    #    print(f"rec cl y: {rec_cl_y}")
    #    print(f"mc cl y: {mc_cl_y}")
    n_fake = len(reco_trk_idx) - n_matched
    #print(f"      N FAKE: {len(reco_trk_idx)} - {n_matched} = {n_fake}")
    if n_fake < 0:
        print(f"AIUTOOOOOO")
    #print(f"len(reco_trk_idx): {len(reco_trk_idx)} - reco_to_gen: {reco_to_gen} - n_matched: {n_matched}")
    #print(f"fake: {n_fake}")
    #print(f"eff_info[method][ev_idx]: {eff_info[method][ev_idx]}\n")



    #if return_good_flag:
        # Ritorna lista di booleani per ogni MC track
    return eff_info[method][ev_idx], reco_to_gen, n_fake, fake_id, np.array(diff_theta_list), np.array(diff_phi_list), np.array(gen_theta_list), np.array(gen_phi_list)

    

def write_counters(dump, counters, method):
    """
    Write counters for a method to dump file (standard format).
    
    Args:
        dump: file object for writing
        counters: counters dictionary for the method
        method: method name (e.g., "m1", "m2")
    """

    dump.write(f"\n--- {method.upper()} ---\n")
    dump.write(f"Total well generated events (denominator): {counters['total_gen_good']}\n")
    dump.write(f"Well reconstructed events (numerator): {counters['reconstructed_good']}\n")
    dump.write(f"Not reconstructed events: {counters["not_reconstructed"]}\n\n")
    
    dump.write(f"Multiplicity = 0: {counters['mult_0']}\n")
    dump.write(f"Bad multiplicity (mult > 1): {counters['bad_mult']}\n")
    dump.write(f"Tracks with issues: {counters['issues']}\n")
    dump.write(f"Angle mismatches: {counters['angle_mismatch']}\n")
    dump.write(f"Clusters not matching: {counters['clusters_not_matching']}\n")



def print_counters(counters, method):
    """
    Print counters for a method to console (standard format).

    Args:
        counters: counters dictionary for the method
        method: method name (e.g., "m1", "m2")
    """
    print(f"\n-------- Summary {method} --------\n")
    print(f"Total well generated events: {counters['total_gen_good']}")
    print(f"Well reconstructed events: {counters['reconstructed_good']}")
    print(f"Not reconstructed events: {counters['not_reconstructed']}")
    print(f"Multiplicity = 0: {counters['mult_0']}")
    print(f"Bad multiplicity (mult > 1): {counters['bad_mult']}")
    print(f"Tracks with issues: {counters['issues']}")
    print(f"Angle mismatches: {counters['angle_mismatch']}")
    print(f"Clusters not matching: {counters['clusters_not_matching']}")



def make_efficiency_hist(var_name, rec_m1_flat, rec_m2_flat, gen_flat, bins, n_tracks_per_event):
    """
    Create efficiency histograms for theta or phi.
    
    Args:
        var_name: variable name ("theta" or "phi")
        rec_m1_flat: flattened array of M1 reconstructed values
        rec_m2_flat: flattened array of M2 reconstructed values
        gen_flat: flattened array of generated values
        bins: array of bin edges
    
    Returns:
        tuple: (h_rec_m1, h_rec_m2, h_gen, h_eff_m1, h_eff_m2)
    """
    # Convert bins to numpy array if needed
    bins_array = np.array(bins)
    n_bins = len(bins_array) - 1
    
    # Create histograms
    h_rec_m1 = ROOT.TH1F(f"h_rec_{var_name}_m1", 
                         f"Reconstructed tracks VS #{var_name} - M1 - {n_tracks_per_event} tracks per event;#{var_name} (deg);Entries", 
                         n_bins, bins_array)
    
    h_rec_m2 = ROOT.TH1F(f"h_rec_{var_name}_m2", 
                         f"Reconstructed tracks VS #{var_name} - M2 - {n_tracks_per_event} tracks per event;#{var_name} (deg);Entries", 
                         n_bins, bins_array)
    
    h_gen = ROOT.TH1F(f"h_gen_{var_name}", 
                      f"Generated tracks VS #{var_name};#{var_name} (deg);Entries", 
                      n_bins, bins_array)
    
    # Fill histograms
    for val in rec_m1_flat:
        h_rec_m1.Fill(val)
    for val in rec_m2_flat:
        h_rec_m2.Fill(val)
    for val in gen_flat:
        h_gen.Fill(val)
    
    # Create efficiency histograms
    h_eff_m1 = h_rec_m1.Clone(f"h_eff_{var_name}_m1")
    h_eff_m1.Divide(h_rec_m1, h_gen, 1.0, 1.0, "B")  # binomial errors
    h_eff_m1.SetTitle(f"Tracking Efficiency VS #{var_name} - M1 - {n_tracks_per_event} tracks per event;#{var_name} (deg);Efficiency")
    
    h_eff_m2 = h_rec_m2.Clone(f"h_eff_{var_name}_m2")
    h_eff_m2.Divide(h_rec_m2, h_gen, 1.0, 1.0, "B")  # binomial errors
    h_eff_m2.SetTitle(f"Tracking Efficiency VS #{var_name} - M2 - {n_tracks_per_event} tracks per event;#{var_name} (deg);Efficiency")
    
    return h_rec_m1, h_rec_m2, h_gen, h_eff_m1, h_eff_m2



def make_fake_hist(var_name, fake_m1_flat, fake_m2_flat, all_reco_m1_flat, all_reco_m2_flat, bins, n_tracks_per_event):
    """
    Create fake track histograms for theta or phi.
    
    Args:
        var_name: variable name ("theta" or "phi")
        rec_m1_flat: flattened array of M1 reconstructed values
        rec_m2_flat: flattened array of M2 reconstructed values
        gen_flat: flattened array of generated values
        bins: array of bin edges
    
    Returns:
        tuple: (h_rec_m1, h_rec_m2, h_gen, h_eff_m1, h_eff_m2)
    """
    # Convert bins to numpy array if needed
    bins_array = np.array(bins)
    n_bins = len(bins_array) - 1
    
    # Create histograms
    h_all_rec_m1 = ROOT.TH1F(f"h_all_rec_{var_name}_m1", 
                         f"All Reconstructed tracks VS #{var_name} - M1 - {n_tracks_per_event} tracks per event;#{var_name} (deg);Entries", 
                         n_bins, bins_array)
    
    h_all_rec_m2 = ROOT.TH1F(f"h_all_rec_{var_name}_m2", 
                         f"All Reconstructed tracks VS #{var_name} - M2 - {n_tracks_per_event} tracks per event;#{var_name} (deg);Entries", 
                         n_bins, bins_array)
    
    h_fake_m1 = ROOT.TH1F(f"h_fake_{var_name}_m1", 
                      f"Fake tracks VS #{var_name} - M1 - {n_tracks_per_event} tracks per event;#{var_name} (deg);Entries", 
                      n_bins, bins_array)
    
    h_fake_m2 = ROOT.TH1F(f"h_fake_{var_name}_m2", 
                      f"Fake tracks VS #{var_name} - M2 - {n_tracks_per_event} tracks per event;#{var_name} (deg);Entries", 
                      n_bins, bins_array)
    
    # Fill histograms
    for val in all_reco_m1_flat:
        h_all_rec_m1.Fill(val)
    for val in all_reco_m2_flat:
        h_all_rec_m2.Fill(val)
    for val in fake_m1_flat:
        h_fake_m1.Fill(val)
    for val in fake_m2_flat:
        h_fake_m2.Fill(val)
    
    # Create efficiency histograms
    h_fake_ratio_m1 = h_all_rec_m1.Clone(f"h_fake_ratio_{var_name}_m1")
    h_fake_ratio_m1.Divide(h_fake_m1, h_all_rec_m1, 1.0, 1.0, "B")  # binomial errors
    h_fake_ratio_m1.SetTitle(f"Fake Track Ratio VS #{var_name} - M1 - {n_tracks_per_event} tracks per event;#{var_name} (deg);Fake Ratio")
    
    h_fake_ratio_m2 = h_all_rec_m2.Clone(f"h_fake_ratio_{var_name}_m2")
    h_fake_ratio_m2.Divide(h_fake_m2, h_all_rec_m2, 1.0, 1.0, "B")  # binomial errors
    h_fake_ratio_m2.SetTitle(f"Fake Track Ratio VS #{var_name} - M2 - {n_tracks_per_event} tracks per event;#{var_name} (deg);Fake Ratio")
    
    return h_fake_ratio_m1, h_fake_ratio_m2


def initialize_counters(n_events, n_tracks_per_event, file_data):
    """
    Initialize counters and info dictionaries for efficiency analysis.
    
    Args:
        n_events: total number of events
        n_tracks_per_event: number of tracks per event
    """

    eff_info = {
        "m1": {},
        "m2": {}
    }

    rec_info = {
        "m1": {},
        "m2": {}
    }

    for m in ["m1", "m2"]:
        for ev_idx in range(n_events):
            eff_info[m][ev_idx] = {}
            rec_info[m][ev_idx] = {}
            for trk_idx in range(n_tracks_per_event):
                eff_info[m][ev_idx][trk_idx] = False
            for reco_trk_idx in range(file_data['mult_data'][m][ev_idx]):
                rec_info[m][ev_idx][reco_trk_idx] = False

    eff_counters = {
        "mc": {
            "tot_gen": 0,      # totale MC tracks (denominatore)
            "3cl_gen": 0,   # totale MC tracks con 3 cluster
            "2cl_gen": 0,   # totale MC tracks con 2 cluster
        },
        "m1": {
            "eff": {
                "tot_rec_gen": 0,  # totale MC tracks ricostruite (numeratore)
                "3cl_rec_gen": 0,   # totale MC tracks con 3 cluster ricostruite
                "2cl_rec_gen": 0,   # totale MC tracks con 2 cluster ricostruite
            },
            "fake": {
                "tot_rec": 0,      # totale reco tracks (denominatore)
                "tot_fake": 0,     # totale fake (numeratore)
                "3cl_rec": 0,      # totale reco con 3 cluster (denominatore)
                "3cl_fake": 0,     # totale fake con 3 cluster (numeratore)
                "2cl_rec": 0,      # totale reco con 2 cluster (denominatore)
                "2cl_fake": 0,     # totale fake con 2 cluster (numeratore)
            },
        },
        "m2": {
            "eff": {
                "tot_rec_gen": 0,  # totale MC tracks ricostruite (numeratore)
                "3cl_rec_gen": 0,   # totale MC tracks con 3 cluster ricostruite
                "2cl_rec_gen": 0,   # totale MC tracks con 2 cluster ricostruite
            },
            "fake": {
                "tot_rec": 0,      # totale reco tracks (denominatore)
                "tot_fake": 0,     # totale fake (numeratore)
                "3cl_rec": 0,      # totale reco con 3 cluster (denominatore)
                "3cl_fake": 0,     # totale fake con 3 cluster (numeratore)
                "2cl_rec": 0,      # totale reco con 2 cluster (denominatore)
                "2cl_fake": 0,     # totale fake con 2 cluster (numeratore)
            }
        }
    }

    return eff_info, rec_info, eff_counters


'''
def process_delta_event(ev_idx, mult_value, event_idx_arr, cl_x_arr, cl_y_arr, cl_z_arr,
                       mc_x, mc_y, mc_layer, rec_theta, rec_phi,
                       gen_theta, gen_phi, gen_mask, counter,
                       theta_outlier_threshold=0.5, phi_outlier_threshold=1.5):
    """
    Process a single event for delta (residual) distributions.
    
    Args:
        ev_idx: event index
        mult_value: multiplicity for this method
        event_idx_arr: array of event indices in reconstructed tree
        cl_x_arr, cl_y_arr, cl_z_arr: reconstructed clusters coordinates
        mc_x, mc_y, mc_layer: MC truth clusters coordinates
        rec_theta, rec_phi: reconstructed angles arrays
        gen_theta, gen_phi: generated angles coordinates arrays
        gen_mask: boolean mask for generated tracks
        counter: counters dictionary for this method
        theta_outlier_threshold: threshold for theta outlier in degrees
        phi_outlier_threshold: threshold for phi outlier in degrees
    
    Returns:
        tuple: (delta_theta, delta_phi, gen_theta_val, gen_phi_val) if track is valid, else None
    """

    # case 1: generated track is good
    is_correctly_reconstructed = False
    if is_gen_good:
        counter["total_gen_good"] += 1
        #is_correctly_reconstructed = False

        # No track reconstructed (mult = 0)
        if mult_value == 0:
            counter["mult_0"] += 1
            counter["not_reconstructed"] += 1

        # mult > 1 - check all tracks until one is found
        else:
            if mult_value > 1:
                counter["bad_mult"] += 1
       
            # find tracks positions matching indeces
            trk_pos = np.where(event_idx_list == ev_idx)[0]

            # track with issues (not found within SelectedEvents tree)
            if len(trk_pos) == 0:
                counter["issues"] += 1

            # tracks found, clusters check
            else:
                found_good_track = False   # for mult > 1 cases
                for i, idx in enumerate(trk_pos):

                    if found_good_track:
                        break

                    is_reconstructed = is_track_reconstructed(
                        cl_x_arr[idx], cl_y_arr[idx], cl_z_arr[idx], 
                        mc_x[ev_idx], mc_y[ev_idx], mc_layer[ev_idx]
                    )

                    # clusters not matching
                    if not is_reconstructed:
                        if not found_good_track:
                            counter["clusters_not_matching"] += 1
                        continue
                        #counter["not_reconstructed"] += 1
                    
                    # check angles differences
                    else:
                        track_theta = rec_theta[idx]
                        track_phi = rec_phi[idx]
                        mc_theta_val = gen_theta[ev_idx]
                        mc_phi_val = gen_phi[ev_idx] 

                        theta_angle_mismatch, phi_angle_mismatch = check_angle_mismatch(
                            track_theta, track_phi, mc_theta_val, mc_phi_val, 
                            theta_angle_threshold, phi_angle_threshold
                        )
                
                        if theta_angle_mismatch or phi_angle_mismatch:
                            if not found_good_track:
                                counter["angle_mismatch"] += 1
                            #counter["not_reconstructed"] += 1
                        else:
                            # Track correctly reconstructed
                            found_good_track = True
                            is_correctly_reconstructed = True
                            counter["reconstructed_good"] += 1

                if not found_good_track:
                    counter["not_reconstructed"] += 1 # if no valid track found

    if return_good_flag:
        return is_correctly_reconstructed
'''


def write_delta_counters(dump, counters, method):
    """
    Write delta analysis counters to dump file.
    
    Args:
        dump: file object for writing
        counters: counters dictionary for the method
        method: method name (e.g., "m1", "m2")
    """
    dump.write(f"\n--- {method.upper()} Delta Analysis ---\n")
    dump.write(f"Total tracks: {counters.get('tracks', 0)}\n")
    dump.write(f"Theta outliers (>0.5 deg): {counters.get('theta_great_out', 0)}\n")
    dump.write(f"Theta outliers (<-0.5 deg): {counters.get('theta_low_out', 0)}\n")
    dump.write(f"Phi outliers (>1.5 deg): {counters.get('phi_great_out', 0)}\n")
    dump.write(f"Phi outliers (<-1.5 deg): {counters.get('phi_low_out', 0)}\n")


def print_delta_counters(counters, method):
    """
    Print delta analysis counters to console.
    
    Args:
        counters: counters dictionary for the method
        method: method name (e.g., "m1", "m2")
    """
    print(f"\n--- {method.upper()} Delta Summary ---")
    print(f"Total tracks: {counters.get('tracks', 0)}")
    print(f"Theta outliers: >0.5 deg: {counters.get('theta_great_out', 0)}, "
          f"<-0.5 deg: {counters.get('theta_low_out', 0)}")
    print(f"Phi outliers: >1.5 deg: {counters.get('phi_great_out', 0)}, "
          f"<-1.5 deg: {counters.get('phi_low_out', 0)}")
