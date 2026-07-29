import ROOT
import numpy as np
import os
import argparse
import sys
import re
import json

def fit_peak_parabola(hist, name, particle):

    max_bin = hist.GetMaximumBin()

    x_max = hist.GetBinCenter(max_bin)

    bin_width = hist.GetBinWidth(max_bin)


    # finestra attorno al massimo
    if particle == "e":
        xmin = x_max - 5*bin_width
        xmax = x_max + 5*bin_width
    elif particle == "p":
        xmin = x_max - 5*bin_width
        xmax = x_max + 5*bin_width
    if particle == "c":
        xmin = x_max - 5*bin_width
        xmax = x_max + 5*bin_width


    parabola = ROOT.TF1(
        name,
        "pol2",
        xmin,
        xmax
    )


    # inizializzazione
    parabola.SetParameters(
        hist.GetMaximum(),
        0,
        -1
    )


    result = hist.Fit(
        parabola,
        "SQ",
        "",
        xmin,
        xmax
    )


    if result.Status() != 0:
        print("⚠️ parabola fit failed")
        return None, None, None


    # pol2 = a*x^2+b*x+c
    a = parabola.GetParameter(2)
    b = parabola.GetParameter(1)


    if a == 0:
        return None, parabola, result


    # vertice
    mpv = -b/(2*a)


    return mpv, parabola, result


def get_fwhm_from_fit(parabola, mpv):

    ymax = parabola.Eval(mpv)

    level = 0.5 * ymax

    a = parabola.GetParameter(2)

    if a >= 0:
        return None

    dx = np.sqrt((level - ymax) / a)

    return dx * 2

def create_theta_hist(values, name, energy, particle):

    values = np.array(values)

    mean = np.mean(values)
    sigma = np.std(values)


    xmin = np.min(values)
    if particle == "e":

        if energy < 7:
            xmax = 70
            bins = 45
        else:
            xmax = 40
            bins = 35
        if energy > 25:
            xmax = 30
        if energy > 55:
            xmax = 15
        if energy > 89:
            xmax = 10
    elif particle == "p":
        xmax = 10
        bins = 40
        if energy > 51:
            xmax = 10
        if energy > 68:
            xmax = 8
        if energy > 99:
            xmax = 4
        if energy > 130:
            xmax = 3

    elif particle == "c":
        xmax = 3
        bins = 40

        if energy > 180:
            xmax = 2


    hist = ROOT.TH1F(
        name,
        "Theta distribution",
        bins,
        xmin,
        xmax
    )

    for v in values:
        hist.Fill(v)

    props = {
        "mean": mean,
        "sigma": sigma,
        "bins": bins,
        "xmin": xmin,
        "xmax": xmax,
        "entries": len(values)
    }

    return hist, props


def draw_theta_fits(hist1,fit1,mpv1,fwhm1,fitres1,hist2,fit2,mpv2,fwhm2,fitres2,filename,title):

    canvas=ROOT.TCanvas(
        "c",
        "theta fits",
        1200,
        500
    )

    canvas.Divide(2,1)

    canvas.cd(1)

    hist1.SetTitle(title+" M1")
    hist1.GetXaxis().SetTitle("#theta [deg]")
    hist1.GetYaxis().SetTitle("Entries")

    hist1.Draw("E")
    fit1.SetLineColor(ROOT.kRed)
    fit1.Draw("same")

    legend1 = ROOT.TLegend(0.6,0.7,0.9,0.9)

    legend1.AddEntry(
        ROOT.nullptr,
        f"MPV = {mpv1:.3f} deg",
        ""
    )

    legend1.AddEntry(
        ROOT.nullptr,
        f"FWHM = {fwhm1:.3f} deg",
        ""
    )

    legend1.AddEntry(
        ROOT.nullptr,
        f"RMS = {hist1.GetRMS():.3f} deg",
        ""
    )

    legend1.AddEntry(
        ROOT.nullptr,
        f"#chi^{{2}}/NDF = {fitres1.Chi2():.1f}/{fitres1.Ndf()}",
        ""
    )

    legend1.Draw()

    canvas.cd(2)

    hist2.SetTitle(title+" M2")
    hist2.GetXaxis().SetTitle("#theta [deg]")
    hist2.GetYaxis().SetTitle("Entries")

    hist2.Draw("E")
    fit2.SetLineColor(ROOT.kRed)
    fit2.Draw("same")


    legend2 = ROOT.TLegend(0.6,0.7,0.9,0.9)
    
    legend2.AddEntry(
        ROOT.nullptr,
        f"MPV = {mpv2:.3f} deg",
        ""
    )

    legend2.AddEntry(
        ROOT.nullptr,
        f"FWHM = {fwhm2:.3f} deg",
        ""
    )

    legend2.AddEntry(
        ROOT.nullptr,
        f"RMS = {hist2.GetRMS():.3f} deg",
        ""
    )

    legend2.AddEntry(
        ROOT.nullptr,
        f"#chi^{{2}}/NDF = {fitres2.Chi2():.1f}/{fitres2.Ndf()}",
        ""
    )

    legend2.Draw()



    canvas.SaveAs(filename)


def initialize_theta_results():

    return {

        "betap": [],

        "m1": {
            "val": [],
            "err": [],
            "rms": []
        },

        "m2": {
            "val": [],
            "err": [],
            "rms": []
        }
    }


def add_rms_results(std_results, betap, method, layer, coord, hist, values):
    """Add RMS results to the results structure."""
    if hist is None or hist.GetEntries() == 0:
        return
    
    try:
        
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        
        # Filtra entro 3 sigma
        mask = np.abs(values - mean) < 3 * std
        filtered_values = values[mask]
        
        # Calcola RMS sui dati filtrati
        sigma = np.std(filtered_values)
        sigma_err = sigma / np.sqrt(len(filtered_values))

        sigma_tot = hist.GetRMS()
        #sigma_err = hist.GetRMSError()
        
        layer_str = f"l{layer}"
        method_str = f"m{method}"
        coord_key_dev = f"{coord}_rms"
        coord_key_err = f"{coord}_rms_err"
        
        std_results[method_str][layer_str][coord_key_dev].append(sigma)
        std_results[method_str][layer_str][coord_key_err].append(sigma_err)
        print(f"   📊 {method_str} L{layer} {coord}: RMS completo={sigma_tot:.4f}, RMS 3σ={sigma:.4f} ({len(filtered_values)}/{len(values)} entries)")
    except Exception as e:
        print(f"⚠️ Error adding RMS results: {e}")


def load_all_trees_data(root_file_path):
    """Read ALL trees from the ROOT file."""
    f = ROOT.TFile(root_file_path, "READ")
    
    trees = []
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TTree"):
            tree_name = key.GetName()
            trees.append((tree_name, obj))
            print(f"📂 Found tree: {tree_name}")
    
    if not trees:
        raise RuntimeError(f"No TTree found in file {root_file_path}")
    
    all_data = {}
    
    for tree_name, tree in trees:
        print(f"\n📖 Processing tree: {tree_name}")
        
        tree_data = {
            'energy': None,
            'data': {}
        }
        
        total_entries = 0
        
        for entry in tree:
            energy = entry.energy
            betap = entry.betap
            method = entry.method
            theta = entry.theta

            
            if tree_data['energy'] is None:
                tree_data['energy'] = energy
            
            tree_data['data'].setdefault(betap, {}).setdefault(method, [])
            tree_data['data'][betap][method].append(theta)
            total_entries += 1
        
        all_data[tree_name] = tree_data
        
        print(f"   Energy: {tree_data['energy']:.0f} MeV")
        print(f"   Total entries: {total_entries}")
        
        #for betap in tree_data['data'].keys():
        #    n_events = sum(len(tree_data['data'][betap][method][layer]['x']) 
        #                   for method in tree_data['data'][betap] 
        #                   for layer in tree_data['data'][betap][method])
        #    print(f"   beta p={betap:.4f}: {n_events} entries")

    f.Close()
    return all_data



def process_single_tree(tree_name,tree_data,outdir,mc,particle):

    energy=tree_data["energy"]
    data=tree_data["data"]


    results=initialize_theta_results()
    outdir = os.path.join(outdir,"plots")


    if mc:
        pdf=os.path.join(
            outdir,
            f"MC_{tree_name}.pdf"
        )
    else:
        pdf=os.path.join(
            outdir,
            f"{tree_name}.pdf"
        )


    for betap in sorted(data.keys()):


        print(
            f"Processing beta p={betap:.4f}"
        )


        results["betap"].append(betap)



        if 1 not in data[betap] or 2 not in data[betap]:
            continue



        theta_m1=data[betap][1]

        theta_m2=data[betap][2]



        if len(theta_m1)<20 or len(theta_m2)<20:
            continue

        h1,props1=create_theta_hist(
            theta_m1,
            f"h_m1_{betap}",
            energy,
            particle
        )

        h2,props2=create_theta_hist(
            theta_m2,
            f"h_m2_{betap}",
            energy,
            particle
        )

        mpv1, parabola1, fitres1 = fit_peak_parabola(h1,"parabola_m1", particle)
        mpv2, parabola2, fitres2 = fit_peak_parabola(h2,"parabola_m2", particle)

        fwhm1 = get_fwhm_from_fit(parabola1, mpv1)
        fwhm2 = get_fwhm_from_fit(parabola2, mpv2)
        rms1 = h1.GetRMS()
        rms2 = h2.GetRMS()
        

        results["m1"]["val"].append(mpv1)
        results["m1"]["err"].append(fwhm1)
        results["m1"]["rms"].append(rms1)

        results["m2"]["val"].append(mpv2)
        results["m2"]["err"].append(fwhm2)
        results["m2"]["rms"].append(rms2)


        if mc:
            plotname=os.path.join(
                outdir,
                f"MC_{tree_name}_betap_{betap:.3f}.pdf"
            )
        else:
            plotname=os.path.join(
                outdir,
                f"{tree_name}_betap_{betap:.3f}.pdf"
            )


        draw_theta_fits(
            h1, parabola1, mpv1, fwhm1, fitres1,
            h2, parabola2, mpv2, fwhm2, fitres2,
            plotname,
            f"{energy:.0f} MeV beta p={betap:.3f}"
        )



    return results


def process_all_trees(input_file, output_dir, particle, mc):

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)


    print(f"\n📖 Reading file: {input_file}")


    all_data = load_all_trees_data(input_file)


    os.makedirs(output_dir,exist_ok=True)



    final_results = {

        "particle": particle,

        "MC": mc,

        "trees": {}

    }


    for tree_name,tree_data in all_data.items():


        print("\n"+"="*60)
        print(f"Processing {tree_name}")
        print("="*60)



        results = process_single_tree(
            tree_name,
            tree_data,
            output_dir,
            mc,
            particle
        )


        final_results["trees"][tree_name]=results

    # nome output
    prefix=os.path.basename(input_file).replace(".root","")


    if mc:
        json_name=f"{prefix}_fit_results_peak.json"
    else:
        json_name=f"{prefix}_fit_results_peak.json"

    json_path=os.path.join(
        output_dir,
        json_name
    )

    with open(json_path,"w") as f:

        json.dump(
            final_results,
            f,
            indent=4
        )



    print("\n✅ JSON saved:")
    print(json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit residuals distributions from ROOT file with multiple trees")
    parser.add_argument("--input-file", required=True, 
                        help="Input ROOT file containing multiple residuals trees")
    parser.add_argument("--output-dir", required=True, 
                        help="Output directory for fit results and PDFs")
    parser.add_argument("--particle", required=True, help="Select particle configuration.")
    parser.add_argument("--mc", action="store_true", help="Define name.")


    args = parser.parse_args()
    ROOT.EnableImplicitMT(20) 
    
    if args.particle not in ["e", "p", "c"]:
        print(f"Error: Invalid particle type '{args.particle}'. Must be one of: e, p, c.")
        sys.exit(1)
    
    if not os.path.exists(args.input_file):
        print(f"❌ Error: File {args.input_file} does not exist!")
        sys.exit(1)
    
    print("="*60)
    print("🔧 RESIDUALS FITTER - MULTIPLE TREES")
    print("="*60)
    print(f"📂 Input file:  {args.input_file}")
    print(f"📁 Output dir:  {args.output_dir}")
    print("="*60)
    
    process_all_trees(args.input_file, args.output_dir, args.particle, args.mc)
    
    print("\n" + "="*60)
    print("✅ COMPLETED!")
    print(f"📁 Results saved in: {args.output_dir}/")
    print("="*60)