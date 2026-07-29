import uproot
import awkward as ak
import numpy as np
import math
import os
from geometry_utils import handle_two_cluster_track, track_hit_TR


def safe_first(arr):
    """Safely return the first element of an array, or NaN if empty."""
    return float(arr[0]) if len(arr) > 0 else float("nan")


def compute_trig_mask(tree):
    """
    Compute trigger mask from trig_conf_flag[6][:, 1].

    Returns:
      (trig_mask, count)
    """
    trig_conf_flag = tree["L2Event/trig_conf_flag[6]"].array(library="np")
    if getattr(trig_conf_flag, "ndim", None) == 2:
        trig_mask = trig_conf_flag[:, 1] == 1
        print("Using trigger mask (ndim=2)")
    else:
        trig_mask = np.array([evt[1] for evt in trig_conf_flag]) == 1
        print("Using trigger mask (object-array)")
    count = np.count_nonzero(trig_mask)
    return trig_mask, count


def compute_trig_count_mask(tree):
    """
    Compute trigger-count mask based on scintillator sums (TR1 & TR2).
    Condition: at least one pair sum > 50 in both TR1 and TR2.

    Returns:
      (trig_count_mask, count)
    """
    try:
        tr1_hg = tree["L2Event/scint_raw_counts/scint_raw_counts.TR1_HG[5][2]"].array(
            library="ak"
        )
        tr2_hg = tree["L2Event/scint_raw_counts/scint_raw_counts.TR2_HG[4][2]"].array(
            library="ak"
        )
    except Exception as e:
        tr1_key = "L2Event/scint_raw_counts/scint_raw_counts.TR1_HG[5][2]"
        tr2_key = "L2Event/scint_raw_counts/scint_raw_counts.TR2_HG[4][2]"
        if tr1_key in tree.keys() and tr2_key in tree.keys():
            tr1_hg = tree[tr1_key].array(library="ak")
            tr2_hg = tree[tr2_key].array(library="ak")
        else:
            raise RuntimeError(
                "Could not find TR1/TR2 scintillator branches in tree."
            ) from e

    tr1_sums = ak.sum(tr1_hg, axis=2)
    tr2_sums = ak.sum(tr2_hg, axis=2)

    tr1_counts_good = ak.any(tr1_sums > 50, axis=1)
    tr2_counts_good = ak.any(tr2_sums > 50, axis=1)

    trig_count_mask = tr1_counts_good & tr2_counts_good
    count = np.count_nonzero(trig_count_mask)
    return trig_count_mask, count


def compute_x0_multiplicity_mask(tree, x0_count):
    """
    Compute mask based on x0 multiplicity condition: ak.num(x0) == x0_count.

    Parameters:
      - tree: uproot tree object
      - x0_count: desired number of x0 elements

    Returns:
      (mask, count)
    """
    x0_full = tree["L2Event/x0"].array(library="ak")
    x0_multiplicity_mask = ak.num(x0_full) == x0_count
    count = np.count_nonzero(x0_multiplicity_mask)
    return x0_multiplicity_mask, count


def compute_x0_m2_multiplicity_mask(tree, x0_m2_count):
    """
    Compute mask based on x0_m2 multiplicity condition: ak.num(x0_m2) == x0_m2_count.

    Parameters:
      - tree: uproot tree object
      - x0_m2_count: desired number of x0_m2 elements

    Returns:
      (mask, count)
    """
    x0_m2_full = tree["L2Event/x0_m2"].array(library="ak")
    x0_m2_multiplicity_mask = ak.num(x0_m2_full) == x0_m2_count
    count = np.count_nonzero(x0_m2_multiplicity_mask)
    return x0_m2_multiplicity_mask, count


def compute_combined_multiplicity_mask(tree, x0_count, x0_m2_count):
    """
    Compute combined mask: ak.num(x0) == x0_count AND ak.num(x0_m2) == x0_m2_count.

    Parameters:
      - tree: uproot tree object
      - x0_count: desired number of x0 elements
      - x0_m2_count: desired number of x0_m2 elements

    Returns:If you would like to see the comparison plot to illustrate the improvement in statistics between Run2 and 3, we can prepare it. Do you need it for a presentation? Or is it just for curiosity?
      (combined_mask, count)
    """
    x0_multiplicity_mask, _ = compute_x0_multiplicity_mask(tree, x0_count)
    x0_m2_multiplicity_mask, _ = compute_x0_m2_multiplicity_mask(tree, x0_m2_count)
    combined_mask = x0_multiplicity_mask & x0_m2_multiplicity_mask
    count = np.count_nonzero(combined_mask)
    return combined_mask, count


def load_and_select_events(input_file, masks_to_apply=None, multiplicity_config=None):
    """
    Open ROOT file and apply selection masks.

    Parameters:
      - input_file: path to ROOT file
      - masks_to_apply: optional dict with boolean flags for which masks to apply.
                        Keys: 'trig', 'trig_count', 'x0_multiplicity', 'x0_m2_multiplicity' (default: trig=False, trig_count=False, others=False)
                        Example: {'trig': True, 'trig_count': True, 'x0_multiplicity': True, 'x0_m2_multiplicity': False}
      - multiplicity_config: optional dict specifying multiplicity values.
                        Keys: 'x0_count', 'x0_m2_count' (used only if x0_multiplicity or x0_m2_multiplicity is True)
                        Example: {'x0_count': 1, 'x0_m2_count': 0}

    Returns:
      (arrays, counters_dict)
      where counters_dict = {'total': N, 'trig': n_trig or None, 'trig_count': n_tc or None, 'x0_multiplicity': n_x0 or None, 'x0_m2_multiplicity': n_x0m2 or None, 'x0_and_x0_m2_multiplicity': n_and or None, 'final': n_final}
    """
    if masks_to_apply is None:
        masks_to_apply = {
            "trig": False,
            "trig_count": False,
            "x0_multiplicity": False,
            "x0_m2_multiplicity": False,
        }
    if multiplicity_config is None:
        multiplicity_config = {"x0_count": 1, "x0_m2_count": 0}

    print(f"🔍 Opening ROOT file: {input_file}")
    with uproot.open(input_file) as f:
        #print("📂 Available keys in ROOT file:")
        #for key in f.keys():
        #    print(f"  - {key}")
        tree = f["L2"]

        total_events = tree.num_entries
        print(f"Total events in file: {total_events}")

        mask = np.ones(total_events, dtype=bool)
        counters = {
            "total": total_events,
            "trig": None,
            "trig_count": None,
            "x0_multiplicity": None,
            "x0_m2_multiplicity": None,
            "x0_and_x0_m2_multiplicity": None,
        }

        # --- Trigger mask ---
        if masks_to_apply.get("trig", False):
            trig_mask, n_trig = compute_trig_mask(tree)
            counters["trig"] = n_trig
            print(
                f"Events after trig_mask: {n_trig} ({n_trig / total_events * 100:.2f}%)"
            )
            mask = mask & trig_mask

        # --- Trigger-count mask ---
        if masks_to_apply.get("trig_count", False):
            trig_count_mask, n_trig_count = compute_trig_count_mask(tree)
            counters["trig_count"] = n_trig_count
            print(
                f"Events after trig_count_mask: {n_trig_count} ({n_trig_count / total_events * 100:.2f}%)"
            )
            mask = mask & trig_count_mask

        # --- x0 and x0_m2 multiplicity masks ---
        x0_multiplicity_mask = None
        x0_m2_multiplicity_mask = None

        if masks_to_apply.get("x0_multiplicity", False):
            x0_count = multiplicity_config.get("x0_count", 1)
            x0_multiplicity_mask, n_x0 = compute_x0_multiplicity_mask(tree, x0_count)
            counters["x0_multiplicity"] = n_x0
            print(
                f"Events after x0_multiplicity_mask (x0 == {x0_count}): {n_x0} ({n_x0 / total_events * 100:.2f}%)"
            )
            mask = mask & x0_multiplicity_mask

        if masks_to_apply.get("x0_m2_multiplicity", False):
            x0_m2_count = multiplicity_config.get("x0_m2_count", 0)
            x0_m2_multiplicity_mask, n_x0_m2 = compute_x0_m2_multiplicity_mask(
                tree, x0_m2_count
            )
            counters["x0_m2_multiplicity"] = n_x0_m2
            print(
                f"Events after x0_m2_multiplicity_mask (x0_m2 == {x0_m2_count}): {n_x0_m2} ({n_x0_m2 / total_events * 100:.2f}%)"
            )
            mask = mask & x0_m2_multiplicity_mask

        # --- AND count if both multiplicities are enabled ---
        if masks_to_apply.get("x0_multiplicity", False) and masks_to_apply.get(
            "x0_m2_multiplicity", False
        ):
            x0_count = multiplicity_config.get("x0_count", 1)
            x0_m2_count = multiplicity_config.get("x0_m2_count", 0)
            n_and = np.count_nonzero(x0_multiplicity_mask & x0_m2_multiplicity_mask)
            counters["x0_and_x0_m2_multiplicity"] = n_and
            print(
                f"Events after AND of both multiplicities (x0 == {x0_count} AND x0_m2 == {x0_m2_count}): {n_and} ({n_and / total_events * 100:.2f}%)"
            )

        # --- Branch loading (only for selected events) ---
        branches = [
            "tracks_hough.x0",
            "tracks_comb.x0",
            "tracks_hough.y0",
            "tracks_comb.y0",
            "tracks_hough.theta",
            "tracks_comb.theta",
            "tracks_hough.phi",
            "tracks_comb.phi",
            "clusters.posX",
            "clusters.posY",
            "clusters.posZ",
            "clusters.cls_size",
            "clusters.cls_idx",
            "tracks_hough.clusterResX",
            "tracks_comb.clusterResX",
            "tracks_hough.clusterResY",
            "tracks_comb.clusterResY",
            "tracks_hough.nClusters",
            "tracks_comb.nClusters",
            "tracks_hough.trkIdx",
            "tracks_comb.trkIdx",   
            "tracks_hough.clusterIndices",
            "tracks_comb.clusterIndices",   
        ]
        arrays = tree.arrays(branches, library="ak")[mask]
        n_final = len(arrays)
        counters["final"] = n_final

    # --- Summary printout ---
    print("\n✅ Event selection summary:")
    print("=" * 60)

    summary_items = []
    summary_items.append(("Total events", total_events))

    if counters["trig"] is not None:
        summary_items.append(("After trig_mask", counters["trig"]))
    if counters["trig_count"] is not None:
        summary_items.append(("After trig_count_mask", counters["trig_count"]))
    if counters["x0_multiplicity"] is not None:
        x0_count = multiplicity_config.get("x0_count", 1)
        summary_items.append(
            (
                f"After x0_multiplicity_mask (x0 == {x0_count})",
                counters["x0_multiplicity"],
            )
        )
    if counters["x0_m2_multiplicity"] is not None:
        x0_m2_count = multiplicity_config.get("x0_m2_count", 0)
        summary_items.append(
            (
                f"After x0_m2_multiplicity_mask (x0_m2 == {x0_m2_count})",
                counters["x0_m2_multiplicity"],
            )
        )
    if counters["x0_and_x0_m2_multiplicity"] is not None:
        x0_count = multiplicity_config.get("x0_count", 1)
        x0_m2_count = multiplicity_config.get("x0_m2_count", 0)
        summary_items.append(
            (
                f"AND of both (x0 == {x0_count} AND x0_m2 == {x0_m2_count})",
                counters["x0_and_x0_m2_multiplicity"],
            )
        )

    summary_items.append(("After all masks combined", n_final))

    for label, value in summary_items:
        perc = (value / total_events * 100) if total_events > 0 else 0
        print(f"  {label:<50}: {value:6d} ({perc:5.2f}%)")

    print("=" * 60)

    return arrays, counters


class Cluster:
    def __init__(self, mean_x, mean_y, mean_z, size, res_x, res_y, track_idx, cluster_idx):
        self.mean_x = float(mean_x)
        self.mean_y = float(mean_y)
        self.mean_z = float(mean_z)
        self.size = int(size)
        self.res_x = float(res_x)
        self.res_y = float(res_y)
        self.track_idx = int(track_idx)
        self.cluster_idx = int(cluster_idx)

    def __repr__(self):
        return (
            f"Cluster(x={self.mean_x:.2f}, y={self.mean_y:.2f}, z={self.mean_z:.2f}, "
            f"size={self.size}, res=({self.res_x:.2f},{self.res_y:.2f}), "
            f"trk={self.track_idx})"
        )


class Track:
    def __init__(self, track_idx, x0, y0, theta, phi, clusters):
        self.track_idx = int(track_idx)
        self.x0 = x0
        self.y0 = y0
        self.theta = theta
        self.phi = phi
        self.clusters = clusters  # list[Cluster]
        self.n_cls = len(clusters)

        # Compute D_sum
        self.D_sum = sum(c.res_x**2 + c.res_y**2 for c in clusters)

        # To be filled by analyze_event
        self.hit_tr = False
        self.missing_in_acc = False
        self.issues = {}

    def __repr__(self):
        return (
            f"Track(idx={self.track_idx}, n_cls={self.n_cls}, "
            f"D_sum={self.D_sum:.3f}, hit_tr={self.hit_tr}, "
            f"missing={self.missing_in_acc})"
        )


def get_all_clusters(evt):
    """
    Return all clusters list for event 'evt' 
    """
    cls_mean_x = list(evt["clusters.posX"])
    cls_mean_y = list(evt["clusters.posY"])
    cls_mean_z = list(evt["clusters.posZ"])
    cls_size = list(evt["clusters.cls_size"])
    cls_idx_list = list(evt["clusters.cls_idx"])
    
    all_cls = []
    for i in range(len(cls_mean_x)):
        if cls_idx_list[i] == -999:
            continue
        cluster = Cluster(
            mean_x=cls_mean_x[i],
            mean_y=cls_mean_y[i],
            mean_z=cls_mean_z[i],
            size=cls_size[i],
            res_x=0.0,
            res_y=0.0,
            track_idx=-999,
            cluster_idx=int(cls_idx_list[i]),
        )
        all_cls.append(cluster)
    
    return all_cls

'''
def analyze_event(evt, event_idx, method=""):
    """
    Restituisce una lista di Track (uno per ogni track_idx) e la lista di cluster per il dato evento.

    Ogni Track contiene:
      - track_idx
      - x0, y0, theta, phi
      - clusters: list[Cluster]
      - n_cls
      - hit_tr
      - missing_in_acc
    """
    check_idx = 140
    #print(f"\n🔍 Analyzing event {evt['event_idx']} with method '{method}'")
    #print(f"{evt} - method: {method}\n")
    # Selection criteria to be applied
    apply_cut_tr = True # if True, selects hit_tr = True
    apply_cut_acc = True # if True, missing_in_acceptance = True
    apply_cut_same_z = True # if True, skips tracks with clusters on same z

    # Track indices
    track_idx_list = list(evt[f"tracks_{method}.trkIdx"])

    # --- Build Cluster and Track objects lists ---
    clusters = []
    track_list = []

    # --- Build all clusters list ---
    all_cls = get_all_clusters(evt)

    # if no tracks are reconstructed, just return clusters (if any)
    if len(track_idx_list) == 0:
        return [], all_cls

    clusters_per_track = {}


    # --- Event-level quantities ---
    x0_arr = list(evt[f"tracks_{method}.x0"])
    y0_arr = list(evt[f"tracks_{method}.y0"])
    theta_arr = list(evt[f"tracks_{method}.theta"])
    phi_arr = list(evt[f"tracks_{method}.phi"])
    cls_res_x = list(evt[f"tracks_{method}.clusterResX"])
    cls_res_y = list(evt[f"tracks_{method}.clusterResY"])
    cls_track_idx = list(evt[f"tracks_{method}.clusterIndices"]) # idxs of clusters assigned to tracks 
    
    cls_mean_x = list(evt[f"clusters.posX"])
    cls_mean_y = list(evt[f"clusters.posY"])
    cls_mean_z = list(evt[f"clusters.posZ"])
    cls_size = list(evt[f"clusters.cls_size"])
    cls_idx_list = list(evt[f"clusters.cls_idx"])
    #if event_idx == check_idx:
    #    print(f"event {event_idx} - method '{method}': ")
    #    print(f"  - Found {len(track_idx_list)} tracks and {len(cls_idx_list)} clusters - for method '{method}'")
    #    print(f"  - x0_arr: {x0_arr}")
    #    print(f"  - y0_arr: {y0_arr}")
    #    print(f"  - theta_arr: {theta_arr}")
    #    print(f"  - phi_arr: {phi_arr}")
    #    print(f"    - track_idx_list: {track_idx_list}")
    #    print(f"    - cls_idx_list: {cls_idx_list}")
    #    print(f"    - cls_track_idx: {cls_track_idx}")

    cluster_map = {
        cls_id: pos_id
        for pos_id, cls_id in enumerate(cls_idx_list)
    }

    #if event_idx == check_idx:
    #    print(f"  - Cluster map (cluster_idx -> position in arrays): {cluster_map}")

    if len(track_idx_list) != len(x0_arr):
        print(f"⚠️ WARNING {method}: Dimensions mismatch! "
              f"trk_idx={len(track_idx_list)}, x0_arr={len(x0_arr)}")
        return [], all_cls
    
    for i, ti in enumerate(track_idx_list):
        #if event_idx == check_idx:
        #    print(f"  - Processing track {method} {ti} with track_idx_list index {i}")

        cls_for_track = []

        assoc_cls_idxs = cls_track_idx[i] 
        res_x_track = cls_res_x[i]
        res_y_track = cls_res_y[i]

        #if event_idx == check_idx:
        #    print(f"  - Processing track {method} {ti} with associated cluster IDs {assoc_cls_idxs} and residuals (res_x: {res_x_track}, res_y: {res_y_track})")

        for ci, rx, ry in zip(assoc_cls_idxs, res_x_track, res_y_track):
            
            if ci in (-1, -999):
                continue
            
            cls_pos = cluster_map.get(ci)

            if cls_pos is None:
                print(f"  ⚠️ Track {method} {ti}: cluster with ID {ci} not found in cluster_map, skip")
                continue

            cluster = Cluster(
                mean_x=cls_mean_x[cls_pos],
                mean_y=cls_mean_y[cls_pos],
                mean_z=cls_mean_z[cls_pos],
                size=cls_size[cls_pos],
                res_x=rx,
                res_y=ry,
                track_idx=ti,
                cluster_idx=ci,
            )

            cls_for_track.append(cluster)

            #if event_idx == check_idx:
            #    print(f"    - Assigned Cluster {cluster} to track {method} {ti}")

        clusters_per_track[ti] = cls_for_track
        #if event_idx == check_idx:
        #    print(f"  - Track {method} {ti} has {len(clusters_per_track[ti])} assigned clusters")

    track_list = []          

    for i, ti in enumerate(track_idx_list):
        #if event_idx == check_idx:
        #    print(f"  - Processing track {method} {ti} with associated cluster IDs {[c.cluster_idx for c in clusters_per_track.get(ti, [])]}")

        cls_this_track = clusters_per_track.get(ti, [])
        #if event_idx == check_idx:
            #print(f"  - Track {method} {ti}: found {len(cls_this_track)} clusters assigned - {cls_this_track}")
        if not cls_this_track:
            #print(f"  ⚠️ Track {method} {ti}: no valid clusters assigned, skip")
            continue
        # clusters assigned to this track
        x0_val = x0_arr[i] if i < len(x0_arr) else float("nan")
        y0_val = y0_arr[i] if i < len(y0_arr) else float("nan")
        theta_val = theta_arr[i] if i < len(theta_arr) else float("nan")
        phi_val = phi_arr[i] if i < len(phi_arr) else float("nan")

        theta_rad = math.radians(theta_val)
        phi_rad = math.radians(phi_val)

        track_obj = Track(
            track_idx=ti,
            x0=x0_val,
            y0=y0_val,
            theta=theta_val,
            phi=phi_val,
            clusters=cls_this_track,
        )

        #if event_idx == check_idx:
        #    print(f"  - Created Track object for track {method} {ti}: {track_obj}")

        # skip tracks if clusters on same z (only if apply_cut_same_z == true)
        if(apply_cut_same_z):
            same_z_count = len(track_obj.clusters) - len(set(c.mean_z for c in track_obj.clusters))
            if same_z_count > 0:
                #print(f"    ⚠️ Track M1 {ti}: clusters on same z, skip")
                continue

         # --- Logic depending on n_cls ---
        if track_obj.n_cls == 2: # only for 2-cluster tracks
            # Convert Cluster -> dict for compatibility with old handle_two_cluster_track
            cls_dicts = [
                {
                    "mean_x": c.mean_x,
                    "mean_y": c.mean_y,
                    "mean_z": c.mean_z,
                    "size": c.size,
                    "res_x": c.res_x,
                    "res_y": c.res_y,
                    "track_idx": c.track_idx,
                }
                for c in cls_this_track
            ]

            result = handle_two_cluster_track(
                cls_dicts,  # now a list of dictionaries, as expected
                theta_rad,
                phi_rad,
                dist_z=8.5, # old 3.5
            )

            if result:
                track_obj.missing_in_acc = result["missing_in_acceptance"]
                #if track_obj.missing_in_acc:
                #    print("MISSING IN ACC")
                track_obj.hit_tr = result["hit_TR"]

        else:
            # All other cases → use track_hit_TR
            track_obj.hit_tr = track_hit_TR(
                track_obj.x0, track_obj.y0, theta_rad, phi_rad
            )
            track_obj.missing_in_acc = False

        # event selection 
        if track_obj.D_sum < 100:
            if apply_cut_tr and apply_cut_acc:
                # trigger and acceptance cuts
                if track_obj.hit_tr and not track_obj.missing_in_acc:
                    track_list.append(track_obj)
            elif apply_cut_tr and not apply_cut_acc:
                # trigger cut
                if track_obj.hit_tr:
                    track_list.append(track_obj)
            elif not apply_cut_tr and apply_cut_acc:
                # acceptance cut
                if not track_obj.missing_in_acc:
                    track_list.append(track_obj)
            else:
                # No cuts
                track_list.append(track_obj)


    #if event_idx == check_idx:
    #    print(f"  - Total tracks after selection for method '{method}': {len(track_list)}\n")    

    return track_list, all_cls
    '''


def analyze_event(evt, event_idx, method=""):
    """
    Restituisce:
      - track_list: lista di Track
      - all_cls: lista di Cluster
      - stats: dizionario con statistiche per questo evento
    """
    check_idx = 140
    
    apply_cut_tr = True
    apply_cut_acc = True
    apply_cut_same_z = True

    # Statistiche per questo evento
    stats = {
        'total_tracks': 0,                    # Tracks totali prima di qualsiasi taglio
        'n_cls_2': 0,                         # Tracks con 2 cluster
        'n_cls_3': 0,                         # Tracks con 3 cluster
        'n_cls_gt3': 0,                       # Tracks con >3 cluster
        'hit_tr_true': 0,                     # Tracks con hit_TR = True
        'hit_tr_false': 0,                    # Tracks con hit_TR = False
        'missing_in_acc_true': 0,             # Tracks con missing_in_acceptance = True
        'missing_in_acc_false': 0,            # Tracks con missing_in_acceptance = False
        'dsum_lt100': 0,                      # Tracks con D_sum < 100
        'dsum_ge100': 0,                      # Tracks con D_sum >= 100
        'passed_all': 0,                      # Tracks che passano tutti i tagli
        'rejected_by_tr': 0,                  # Tracks rifiutate per hit_TR = False
        'rejected_by_acc': 0,                 # Tracks rifiutate per missing_in_acceptance = True
        'rejected_by_dsum': 0,                # Tracks rifiutate per D_sum >= 100
        'rejected_by_same_z': 0,              # Tracks rifiutate per clusters same z
        'rejected_by_multiple': 0,            # Tracks rifiutate per multiple ragioni
        # Per tracks a 2 cluster
        'n_cls_2_hit_tr_true': 0,             # 2 cluster E hit_TR = True
        'n_cls_2_hit_tr_false': 0,            # 2 cluster E hit_TR = False
        'n_cls_2_missing_acc_false': 0,       # 2 cluster E missing_in_acceptance = False
        'n_cls_2_missing_acc_true': 0,        # 2 cluster E missing_in_acceptance = True
        'n_cls_2_good': 0,                    # 2 cluster E hit_TR = True E missing_in_acc = False
        # Per tracks a 3 cluster
        'n_cls_3_hit_tr_true': 0,             # 3 cluster E hit_TR = True
        'n_cls_3_hit_tr_false': 0,            # 3 cluster E hit_TR = False
        'n_cls_3_dsum_lt100': 0,              # 3 cluster E D_sum < 100
        'n_cls_3_dsum_ge100': 0,              # 3 cluster E D_sum >= 100
        'n_cls_3_good': 0,                    # 3 cluster E hit_TR = True E D_sum < 100
    }

    track_idx_list = list(evt[f"tracks_{method}.trkIdx"])
    clusters = []
    track_list = []
    all_cls = get_all_clusters(evt)

    if len(track_idx_list) == 0:
        return [], all_cls, stats

    clusters_per_track = {}

    x0_arr = list(evt[f"tracks_{method}.x0"])
    y0_arr = list(evt[f"tracks_{method}.y0"])
    theta_arr = list(evt[f"tracks_{method}.theta"])
    phi_arr = list(evt[f"tracks_{method}.phi"])
    cls_res_x = list(evt[f"tracks_{method}.clusterResX"])
    cls_res_y = list(evt[f"tracks_{method}.clusterResY"])
    cls_track_idx = list(evt[f"tracks_{method}.clusterIndices"])
    
    cls_mean_x = list(evt[f"clusters.posX"])
    cls_mean_y = list(evt[f"clusters.posY"])
    cls_mean_z = list(evt[f"clusters.posZ"])
    cls_size = list(evt[f"clusters.cls_size"])
    cls_idx_list = list(evt[f"clusters.cls_idx"])

    cluster_map = {
        cls_id: pos_id
        for pos_id, cls_id in enumerate(cls_idx_list)
    }

    if len(track_idx_list) != len(x0_arr):
        print(f"⚠️ WARNING {method}: Dimensions mismatch! "
              f"trk_idx={len(track_idx_list)}, x0_arr={len(x0_arr)}")
        return [], all_cls, stats
    
    for i, ti in enumerate(track_idx_list):
        cls_for_track = []
        assoc_cls_idxs = cls_track_idx[i] 
        res_x_track = cls_res_x[i]
        res_y_track = cls_res_y[i]

        for ci, rx, ry in zip(assoc_cls_idxs, res_x_track, res_y_track):
            if ci in (-1, -999):
                continue
            cls_pos = cluster_map.get(ci)
            if cls_pos is None:
                continue
            cluster = Cluster(
                mean_x=cls_mean_x[cls_pos],
                mean_y=cls_mean_y[cls_pos],
                mean_z=cls_mean_z[cls_pos],
                size=cls_size[cls_pos],
                res_x=rx,
                res_y=ry,
                track_idx=ti,
                cluster_idx=ci,
            )
            cls_for_track.append(cluster)

        clusters_per_track[ti] = cls_for_track

    for i, ti in enumerate(track_idx_list):
        cls_this_track = clusters_per_track.get(ti, [])
        if not cls_this_track:
            continue

        x0_val = x0_arr[i] if i < len(x0_arr) else float("nan")
        y0_val = y0_arr[i] if i < len(y0_arr) else float("nan")
        theta_val = theta_arr[i] if i < len(theta_arr) else float("nan")
        phi_val = phi_arr[i] if i < len(phi_arr) else float("nan")

        theta_rad = math.radians(theta_val)
        phi_rad = math.radians(phi_val)

        track_obj = Track(
            track_idx=ti,
            x0=x0_val,
            y0=y0_val,
            theta=theta_val,
            phi=phi_val,
            clusters=cls_this_track,
        )

        # Statistiche base
        stats['total_tracks'] += 1
        
        n_cls = track_obj.n_cls
        if n_cls == 2:
            stats['n_cls_2'] += 1
        elif n_cls == 3:
            stats['n_cls_3'] += 1
        else:
            stats['n_cls_gt3'] += 1

        # same z cut
        same_z_count = len(track_obj.clusters) - len(set(c.mean_z for c in track_obj.clusters))
        if apply_cut_same_z and same_z_count > 0:
            stats['rejected_by_same_z'] += 1
            continue

        # Calcola hit_TR e missing_in_acc in base a n_cls
        if n_cls == 2:
            cls_dicts = [
                {
                    "mean_x": c.mean_x,
                    "mean_y": c.mean_y,
                    "mean_z": c.mean_z,
                    "size": c.size,
                    "res_x": c.res_x,
                    "res_y": c.res_y,
                    "track_idx": c.track_idx,
                }
                for c in cls_this_track
            ]
            result = handle_two_cluster_track(
                cls_dicts,
                theta_rad,
                phi_rad,
                dist_z=8.5,
            )
            if result:
                track_obj.missing_in_acc = result["missing_in_acceptance"]
                track_obj.hit_tr = result["hit_TR"]
            else:
                track_obj.missing_in_acc = False
                track_obj.hit_tr = False
        else:
            track_obj.hit_tr = track_hit_TR(
                track_obj.x0, track_obj.y0, theta_rad, phi_rad
            )
            track_obj.missing_in_acc = False

        # Statistiche per n_cls == 2
        if n_cls == 2:
            if track_obj.hit_tr:
                stats['n_cls_2_hit_tr_true'] += 1
            else:
                stats['n_cls_2_hit_tr_false'] += 1
            
            if not track_obj.missing_in_acc:
                stats['n_cls_2_missing_acc_false'] += 1
            else:
                stats['n_cls_2_missing_acc_true'] += 1

        # Statistiche per n_cls == 3
        if n_cls == 3:
            if track_obj.hit_tr:
                stats['n_cls_3_hit_tr_true'] += 1
            else:
                stats['n_cls_3_hit_tr_false'] += 1

        # Statistiche hit_TR
        if track_obj.hit_tr:
            stats['hit_tr_true'] += 1
        else:
            stats['hit_tr_false'] += 1

        # Statistiche missing_in_acc
        if track_obj.missing_in_acc:
            stats['missing_in_acc_true'] += 1
        else:
            stats['missing_in_acc_false'] += 1

        # Statistiche D_sum
        Dsum = track_obj.D_sum
        if Dsum < 100:
            stats['dsum_lt100'] += 1
            if n_cls == 3:
                stats['n_cls_3_dsum_lt100'] += 1
        else:
            stats['dsum_ge100'] += 1
            if n_cls == 3:
                stats['n_cls_3_dsum_ge100'] += 1

        # ---- APPLICAZIONE TAGLI ----
        rejected = False
        reasons = []

        # Taglio D_sum
        if Dsum >= 100:
            rejected = True
            stats['rejected_by_dsum'] += 1
            reasons.append('dsum')

        # Taglio hit_TR
        if apply_cut_tr and not track_obj.hit_tr:
            rejected = True
            stats['rejected_by_tr'] += 1
            reasons.append('tr')

        # Taglio missing_in_acc
        if apply_cut_acc and track_obj.missing_in_acc:
            rejected = True
            stats['rejected_by_acc'] += 1
            reasons.append('acc')

        if not rejected:
            track_list.append(track_obj)
            stats['passed_all'] += 1
            
            # Statistiche per track che passano tutti i tagli
            if n_cls == 2:
                stats['n_cls_2_good'] += 1
            elif n_cls == 3:
                stats['n_cls_3_good'] += 1
        else:
            if len(reasons) > 1:
                stats['rejected_by_multiple'] += 1

    return track_list, all_cls, stats

    

def check_event_tree(
    input_file, output_dir, multiplicity_config=None
):
    """
    Search event info from EventSummary TTree.
    Usefult to apply multiplicity selection to events.
    """
    
    print("📥 Input file: ", input_file)
    print("📤 Output directory: ", output_dir)

    fin = uproot.open(input_file)
    tree1 = fin["SelectedEvents"]
    tree2 = fin["SelectedEvents_m2"]
    tree_mult = fin["EventSummary"]
    

    if multiplicity_config is None:
        multiplicity_config = {"x0_count": None, "x0_m2_count": None}

    x0_sel = multiplicity_config["x0_count"]
    x0_m2_sel = multiplicity_config["x0_m2_count"]

    # Read from EventSummary
    ev_idx_all = tree_mult["event"].array(library="np")
    x0_mult_all = tree_mult["mult_m1"].array(library="np")
    x0_m2_mult_all = tree_mult["mult_m2"].array(library="np") 

    print(f"\nTotal events in EventSummary: {len(ev_idx_all)}")
    
    # Good event array (true by default)
    good_events = np.ones(len(ev_idx_all), dtype=bool)

    if x0_sel is not None: # comparison with multiplicity_config
        good_events &= (x0_mult_all == x0_sel)
    if x0_m2_sel is not None:
        good_events &= (x0_m2_mult_all == x0_m2_sel)
    
    selected_events = set(ev_idx_all[good_events])
    #print(f"\nSelected events with multiplicity conditions (x0 == {x0_sel}, x0_m2 == {x0_m2_sel}): {selected_events}")

    # Read idxs from SelectedEvents trees
    ev1 = tree1["event_idx"].array(library="np")
    ev2 = tree2["event_idx"].array(library="np")

    print(f"\nTotal events in SelectedEvents: {len(ev1)}")
    print(f"Total events in SelectedEvents_m2: {len(ev2)}")

    mask1 = np.array([e in selected_events for e in ev1])
    mask2 = np.array([e in selected_events for e in ev2])

    print(f"\nGood events matching multiplicities: m1 = {mask1.sum()}, m2 = {mask2.sum()}")

    return x0_sel, x0_m2_sel, mask1, mask2, tree1, tree2


def remove_duplicate_tracks(tracks, event_idx = None):
    """
    Check for duplicate tracks.
    Two tracks are considered duplicates if they are built from the same set of clusters (trk.clusters)
    Only the first occurrence is kept.
    """
    seen = {} # track position (not track index)
    duplicate_indices = []

    for idx, tr in enumerate(tracks):
        clusters_key = tuple(sorted(c.cluster_idx for c in tr.clusters)) # sort clusters
        if clusters_key in seen:
            # duplicate track found
            duplicate_indices.append(idx)
            orig_idx = tracks[seen[clusters_key]].track_idx
            print(
                f"⚠️ Event {event_idx}: Duplicate track trk_idx={tr.track_idx} "
                f"(duplicates trk_idx={orig_idx})"
            )
        else:
            seen[clusters_key] = idx
    # remove clones and return list w/o duplicates
    for idx in sorted(duplicate_indices, reverse=True):
        removed_trk_idx = tracks[idx].track_idx
        del tracks[idx]
        print(f"🗑️ Rimosso trk_idx={removed_trk_idx} duplicato")

    return tracks


def is_good_track(trk):
    """
    Check conditions for good track.
    """
    
    issue_meanx = any(c.mean_x == -999 for c in trk.clusters)
    same_z_count = len(trk.clusters) - len(
        set(c.mean_z for c in trk.clusters)
    )

    is_good = (
        trk.n_cls < 4
        and not issue_meanx
        and trk.hit_tr
        and (trk.n_cls != 2 or not trk.missing_in_acc)
        and trk.D_sum < 100
        and same_z_count == 0
    )
    return is_good, issue_meanx, same_z_count

