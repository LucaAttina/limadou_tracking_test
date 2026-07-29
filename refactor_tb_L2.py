import uproot
import awkward as ak
import numpy as np
import argparse
import os
import ROOT
import array
from efficiency_utils_new import compute_gen_mask
            
def mctruth_gen_mask(input_file, output_file):  # Passa entrambi esplicitamente
    
    print("\n📂 Processing MCtruth tree...")
    
    # Leggi dal file di input
    with uproot.open(input_file) as f:
        mc_tree_uproot = f["MCtruth"]
        gen_mask = compute_gen_mask(mc_tree_uproot, [])
    
    total_events = len(gen_mask)
    selected_events = np.sum(gen_mask)
    
    print(f"Total events: {total_events}")
    print(f"Selected events: {selected_events}")
    
    if selected_events == 0:
        print("⚠️ Warning: No events selected!")
        return gen_mask
    
    # APRI IN LETTURA il file di input
    f_in = ROOT.TFile(input_file, "READ")
    original_tree = f_in.Get("MCtruth")
    
    # APRI IN SCRITTURA il file di output
    f_out = ROOT.TFile(output_file, "UPDATE")
    new_tree = original_tree.CloneTree(0)

    l2_tree = f_in.Get("L2")
    l2_new_tree = l2_tree.CloneTree(0)
    
    event_index = np.zeros(1, dtype=np.int32)
    new_tree.Branch("event_index", event_index, "event_index/I")
    
    print("\n📝 Copying selected events...")
    for i in range(total_events):
        if gen_mask[i]:
            original_tree.GetEntry(i)
            l2_tree.GetEntry(i)
            event_index[0] = i
            new_tree.Fill()      # MCtruth
            l2_new_tree.Fill()
    
    f_out.cd()
    new_tree.Write("MCtruth", ROOT.TObject.kOverwrite)
    l2_new_tree.Write("L2", ROOT.TObject.kOverwrite)
    print(f"✅ MCtruth tree written with {new_tree.GetEntries()} events")
    print(f"✅ L2 tree written with {l2_new_tree.GetEntries()} events")

    f_out.Close()
    f_in.Close()
    
    return gen_mask


def match_hit_to_cls(input_file, output_dir="../data/mc_sim/L1/", num=1):
    
    filename = os.path.basename(input_file) 
    filename = os.path.splitext(filename)[0]
    output_file = os.path.join(output_dir, f"{filename}_{num}_evs_ALL.root")
    
    # Prima crea il file di output con i tree copiati
    f_in = ROOT.TFile(input_file, "READ")
    f_out = ROOT.TFile(output_file, "RECREATE")
    
    tree_ls = ["MCdig", "MCcls", "L2", "TCalib"]
    print("\n📂 Copying TTrees from input file...")
    for tree_name in tree_ls:
        tree = f_in.Get(tree_name)
        if tree and tree.InheritsFrom("TTree"):
            cloned_tree = tree.CloneTree(-1, "fast")
            cloned_tree.Write()
            print(f"✅ {tree_name} tree copied to output ROOT file.")
        else:
            print(f"⚠️ Warning: {tree_name} not found or not a TTree")
    
    f_out.Close()
    
    # Ora processa MCtruth sul file di output
    gen_mask = mctruth_gen_mask(input_file, output_file)  # NOTA: input_file è il file sorgente
    
    # Opzionale: puoi anche copiare MCtruth modificato
    f_out = ROOT.TFile(output_file, "UPDATE")
    f_out.Close()
    
    f_in.Close()


# Esempio di utilizzo
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge n MC tracks (useful for efficiency studies).")
    parser.add_argument("--input-file", required=True, help="Extract data from MCtruth from this L1.root file.")
    parser.add_argument("--output-dir", default="./output", help="Define output directory.")
    parser.add_argument("--num", type=int, default=1, help="Number of MC tracks to merge.")
    args = parser.parse_args()
       
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    match_hit_to_cls(args.input_file, output_dir, args.num)
    #mctruth_gen_mask(args.input_file, output_dir)

  