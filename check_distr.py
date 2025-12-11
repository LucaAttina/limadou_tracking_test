import argparse
import os
import numpy as np
import awkward as ak
import ROOT
import yaml

from utils import load_and_select_events, analyze_event, summarize_tracks
from geometry_utils import load_geometry


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


def write_txt_dump(
    arrays, output_dir, base_name="cluster_details", x0_count=1, x0_m2_count=0
):
    """Write event details to a text file for events matching given multiplicities.

    Only outputs tracks that satisfy:
    - hit_tr == True
    - n_cls < 4
    - if n_cls == 2, then missing_in_acc == False

    Dumps all available info for all selected tracks and their clusters.
    Also prints a global summary of tracks before selection (in console and in the TXT file).
    """

    import awkward as ak
    import os
    from utils import analyze_event

    # --------------------------
    # Select events matching multiplicities
    # --------------------------
    x0_mult = ak.num(arrays["L2Event/x0"])
    x0_m2_mult = ak.num(arrays["L2Event/x0_m2"])

    sel_mask = (x0_mult == x0_count) & (x0_m2_mult == x0_m2_count)
    nevt = int(ak.sum(sel_mask))
    print(f"Events for dump (x0=={x0_count}, x0_m2=={x0_m2_count}): {nevt}")

    sel = arrays[sel_mask]
    if len(sel) == 0:
        print(
            f"No events matching x0=={x0_count} && x0_m2=={x0_m2_count}. Skipping dump."
        )
        return

    # ===============================
    # Global summary counters (before filtering)
    # ===============================
    total_counts = {"n_cls_2": 0, "n_cls_3": 0, "n_cls_gt3": 0}
    hit_tr_counts = {"n_cls_2": 0, "n_cls_3": 0, "n_cls_gt3": 0}
    missing_in_acc_counts = {"n_cls_2": 0}  # only for n_cls == 2
    hit_and_not_missing_counts = {"n_cls_2": 0}  # hit_tr and missing_in_acc==False

    for evt in sel:
        track_list = analyze_event(evt)
        for trk in track_list:
            if trk.n_cls == 2:
                total_counts["n_cls_2"] += 1
                if trk.hit_tr:
                    hit_tr_counts["n_cls_2"] += 1
                if not trk.missing_in_acc:
                    missing_in_acc_counts["n_cls_2"] += 1
                if trk.hit_tr and not trk.missing_in_acc:
                    hit_and_not_missing_counts["n_cls_2"] += 1
            elif trk.n_cls == 3:
                total_counts["n_cls_3"] += 1
                if trk.hit_tr:
                    hit_tr_counts["n_cls_3"] += 1
            elif trk.n_cls > 3:
                total_counts["n_cls_gt3"] += 1
                if trk.hit_tr:
                    hit_tr_counts["n_cls_gt3"] += 1

    # ===============================
    # Prepare output text file
    # ===============================
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(txt_path, "w") as f:
        f.write("# Event dump for selected events\n")
        f.write(f"# x0 multiplicity: {x0_count}, x0_m2 multiplicity: {x0_m2_count}\n")
        f.write(
            "# Filters: hit_tr==True, n_cls<4, if n_cls==2 then missing_in_acc==False\n\n"
        )

        for evt_idx, evt in enumerate(sel):

            # analyze_event returns a list of Track objects
            track_list = analyze_event(evt)

            # --------------------------
            # Apply selection for dump
            # --------------------------
            selected_tracks = [
                trk
                for trk in track_list
                if trk.hit_tr
                and (trk.n_cls == 3 or (trk.n_cls == 2 and not trk.missing_in_acc))
            ]

            if len(selected_tracks) == 0:
                continue

            # Event header
            f.write(f"Event {evt_idx}\n")
            f.write(f"  n_tracks={len(selected_tracks)}\n")

            # Event-level Dsum = sum of all selected tracks’ D_sum
            event_Dsum = sum(trk.D_sum for trk in selected_tracks)
            f.write(f"  Event Dsum={event_Dsum:.5f}\n")

            # Dump each track
            for trk in selected_tracks:
                f.write(
                    f"  Track {trk.track_idx}: "
                    f"x0={trk.x0:.5f}, y0={trk.y0:.5f}, theta={trk.theta:.5f}, phi={trk.phi:.5f}, "
                    f"n_cls={trk.n_cls}, hit_tr={int(trk.hit_tr)}, missing_in_acc={int(trk.missing_in_acc)}, "
                    f"D_sum={trk.D_sum:.5f}\n"
                )

                # Dump all clusters inside this track
                for i, c in enumerate(trk.clusters):
                    attrs = ", ".join(f"{k}={getattr(c, k)}" for k in vars(c))
                    f.write(f"    Cluster {i}: {attrs}\n")

            f.write("\n")

        # ===============================
        # Write global summary at the end
        # ===============================
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
            else:
                f.write(f"{key}: total={tot}, hit_tr={hits} ({frac_hits:.1f}%)\n")

    # ===============================
    # Print same summary in console
    # ===============================
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
                f"{key}: total={tot}, hit_tr={hits} ({frac_hits:.1f}%), "
                f"missing_in_acc=False={missing} ({frac_missing:.1f}%), "
                f"hit_tr and missing_in_acc=False={both} ({frac_both:.1f}%)"
            )
        else:
            print(f"{key}: total={tot}, hit_tr={hits} ({frac_hits:.1f}%)")

    print(f"Saved event dump to {txt_path}")


def compare_angle_distributions(
    arrays, output_dir, base_name="angle_compare", x0_count=1, x0_m2_count=1
):
    """Compare theta/theta_m2 and phi/phi_m2 for events matching given multiplicities.

    Parameters:
      - arrays: awkward arrays returned by load_and_select_events (may already be filtered)
      - x0_count, x0_m2_count: multiplicity requirements used for the selection inside this function
    """
    # select events where x0 multiplicity == x0_count and x0_m2 multiplicity == x0_m2_count
    x0_mult = ak.num(arrays["L2Event/x0"])
    x0_m2_mult = ak.num(arrays["L2Event/x0_m2"])

    sel_mask = (x0_mult == x0_count) & (x0_m2_mult == x0_m2_count)
    nevt = int(ak.sum(sel_mask))
    print(f"Events with x0=={x0_count} and x0_m2=={x0_m2_count}: {nevt}")

    sel = arrays[sel_mask]
    if len(sel) == 0:
        print(
            f"No events matching x0=={x0_count} && x0_m2=={x0_m2_count}. Skipping angle comparisons."
        )
        return {}

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

    import math

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create multiplicity TH2 and compare angles for selected events"
    )
    parser.add_argument(
        "--config",
        help="YAML config file with options (keys: input, output_dir, mask_trig, mask_trig_count, apply_multiplicity, x0_count, x0_m2_count or nested 'masks')",
        default=None,
    )
    parser.add_argument(
        "--input", help="Input ROOT file (overrides config)", default=None
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (overrides config)"
    )
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

    # Determine final values (CLI overrides config)
    input_file = args.input if args.input is not None else cfg_get("input")
    if input_file is None:
        parser.error("Input file must be specified via --input or in the config file")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else cfg_get("output_dir", "../test/output/comp")
    )

    # masks: allow top-level keys or nested 'masks' mapping in the config
    masks_cfg = cfg_get("masks", {}) or {}

    mask_trig = (
        args.mask_trig
        if args.mask_trig is not None
        else cfg_get("mask_trig", masks_cfg.get("trig", True))
    )
    mask_trig_count = (
        args.mask_trig_count
        if args.mask_trig_count is not None
        else cfg_get("mask_trig_count", masks_cfg.get("trig_count", True))
    )

    apply_multiplicity = (
        args.apply_multiplicity
        if args.apply_multiplicity is not None
        else cfg_get("apply_multiplicity", masks_cfg.get("apply_multiplicity", False))
    )

    x0_count = args.x0_count if args.x0_count is not None else cfg_get("x0_count", 1)
    x0_m2_count = (
        args.x0_m2_count if args.x0_m2_count is not None else cfg_get("x0_m2_count", 0)
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

    multiplicity_config = {"x0_count": int(x0_count), "x0_m2_count": int(x0_m2_count)}

    load_geometry()

    arrays, counters = load_and_select_events(
        input_file, masks_to_apply=masks, multiplicity_config=multiplicity_config
    )

    # suffix outputs with multiplicity combo
    suffix = f"_x0_{x0_count}_x0m2_{x0_m2_count}"

    # Build multiplicity TH2 (for the selected events)
    h2 = make_multiplicity_th2(
        arrays, output_dir, base_name="multiplicity_selected" + suffix
    )

    # Write cluster details for events matching the multiplicities
    write_txt_dump(
        arrays,
        output_dir,
        base_name="cluster_details" + suffix,
        x0_count=x0_count,
        x0_m2_count=x0_m2_count,
    )
