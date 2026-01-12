import uproot
import awkward as ak
import numpy as np
import math
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
                        Keys: 'trig', 'trig_count', 'x0_multiplicity', 'x0_m2_multiplicity' (default: trig=True, trig_count=True, others=False)
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
            "trig": True,
            "trig_count": True,
            "x0_multiplicity": False,
            "x0_m2_multiplicity": False,
        }
    if multiplicity_config is None:
        multiplicity_config = {"x0_count": 1, "x0_m2_count": 0}

    print(f"🔍 Opening ROOT file: {input_file}")
    with uproot.open(input_file) as f:
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
        if masks_to_apply.get("trig", True):
            trig_mask, n_trig = compute_trig_mask(tree)
            counters["trig"] = n_trig
            print(
                f"Events after trig_mask: {n_trig} ({n_trig / total_events * 100:.2f}%)"
            )
            mask = mask & trig_mask

        # --- Trigger-count mask ---
        if masks_to_apply.get("trig_count", True):
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
            "L2Event/x0",
            "L2Event/x0_m2",
            "L2Event/y0",
            "L2Event/theta_m2",
            "L2Event/theta",
            "L2Event/phi_m2",
            "L2Event/phi",
            "L2Event/cls_mean_x",
            "L2Event/cls_mean_y",
            "L2Event/cls_mean_z",
            "L2Event/cls_size",
            "L2Event/cls_res_x",
            "L2Event/cls_res_y",
            "L2Event/trk_cls_res_x_m2",
            "L2Event/trk_cls_res_y_m2",
            "L2Event/cls_track_idx",
            "L2Event/trk_idx",
            "L2Event/trk_idx_m2",
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
    def __init__(self, mean_x, mean_y, mean_z, size, res_x, res_y, track_idx):
        self.mean_x = float(mean_x)
        self.mean_y = float(mean_y)
        self.mean_z = float(mean_z)
        self.size = int(size)
        self.res_x = float(res_x)
        self.res_y = float(res_y)
        self.track_idx = int(track_idx)

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



def analyze_event(evt):
    """
    Restituisce una lista di Track (uno per ogni track_idx).

    Ogni Track contiene:
      - track_idx
      - x0, y0, theta, phi
      - clusters: list[Cluster]
      - n_cls
      - hit_tr
      - missing_in_acc
    """

    # --- Event-level quantities ---
    x0_val = safe_first(evt["L2Event/x0"])
    y0_val = safe_first(evt["L2Event/y0"])
    theta_val = safe_first(evt["L2Event/theta"])
    phi_val = safe_first(evt["L2Event/phi"])

    theta_rad = math.radians(theta_val)
    phi_rad = math.radians(phi_val)

    # Track indices
    track_idx_list = list(evt["L2Event/trk_idx"])

    # --- Build all clusters as Cluster objects ---
    clusters = []
    for mx, my, mz, sz, rx, ry, ti in zip(
        evt["L2Event/cls_mean_x"],
        evt["L2Event/cls_mean_y"],
        evt["L2Event/cls_mean_z"],
        evt["L2Event/cls_size"],
        evt["L2Event/cls_res_x"],
        evt["L2Event/cls_res_y"],
        evt["L2Event/cls_track_idx"],
    ):
        if ti in (-1, -999):
            continue

        clusters.append(
            Cluster(
                mean_x=mx,
                mean_y=my,
                mean_z=mz,
                size=sz,
                res_x=rx,
                res_y=ry,
                track_idx=ti,
            )
        )

    # --- Build Track objects ---
    track_list = []

    for ti in track_idx_list:

        # clusters assigned to this track
        cls_for_track = [c for c in clusters if c.track_idx == ti]

        track_obj = Track(
            track_idx=ti,
            x0=x0_val,
            y0=y0_val,
            theta=theta_val,
            phi=phi_val,
            clusters=cls_for_track,
        )


        # --- Logic depending on n_cls ---
        if track_obj.n_cls == 2:
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
                for c in cls_for_track
            ]

            result = handle_two_cluster_track(
                cls_dicts,  # now a list of dictionaries, as expected
                theta_rad,
                phi_rad,
                dist_z=3.5,
            )

            if result:
                track_obj.missing_in_acc = result["missing_in_acceptance"]
                track_obj.hit_tr = result["hit_TR"]

        else:
            # All other cases → use track_hit_TR
            track_obj.hit_tr = track_hit_TR(
                track_obj.x0, track_obj.y0, theta_rad, phi_rad
            )
            track_obj.missing_in_acc = False

        track_list.append(track_obj)

    return track_list

