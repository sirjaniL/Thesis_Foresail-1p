import os
import re

#Handles floats like 1, -1.0, .5, 1e-6, -3.2E+5, etc.
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

#param name and value is fetched ":" is a separator and () is a group
PARAM = re.compile(rf"^\s*([A-Za-z0-9_]+):\s*({NUMBER})\s*$", re.M)

#Mean attitude error seeking from the file
MEAN = re.compile(rf"^\s*Mean error:\s*({NUMBER})\s*$", re.M)

#Mean angular error seeking from the file
MEAN_AN = re.compile(rf"^\s*Mean angular error :\s*({NUMBER})\s*$", re.M)

#Parser, which forms the the key value pairs for mean and paramaters
def parse_result_file(path, search_value):
    """Return {'mean': float, 'params': dict} or None if Mean error is missing."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read() #Entire file read as text

    #Search for mean error value
    m = search_value.search(text)
    if not m:
        return None  #Mean error not found, faulthy file...

    mean = float(m.group(1)) #...mean error value found and extracted
    params = {}

    #Every param name and value is browsed through and aggregated as list of tuples 
    for name, val in PARAM.findall(text):
        
        #all stored as key value pairs, where value is float
        try:
            params[name] = float(val)
        except ValueError:
            pass 
    #key value pair returned
    return {"mean": mean, "params": params}
#Function compares best error mean while scanning every file
def find_best(directory, search_value):
    best = None
    #Loops entire directory and runs parser in it, which compares means between files
    for entry in os.scandir(directory): 
        #Loop looks only for files
        if not entry.is_file():
            continue 
        parsed = parse_result_file(entry.path, search_value)
        if not parsed:
            #error if mean is not found
            print(f"Skipping malformed file (no Mean error): {entry.name}", flush=True)
            continue
        #Comparison between new value and current best
        if best is None or parsed["mean"] < best["mean"]:
            best = {
                "file": entry.name,
                "mean": parsed["mean"],
                "params": parsed["params"]
            }

    return best

if __name__ == "__main__":
    # directory = r"/u/53/user1/unix/adcs_simulation_env/fs1p_adcs/simulations/results/output_files"
    directory = r"/u/53/lonnqvj1/unix/Thesis/Triton/outputfilesjson/Only_for_testing"  # example
    #directory = r"/scratch/work/lonnqvj1/Foresail_1p/adcs_simulation_env/fs1p_adcs/simulations/results/output_files_default_10368_nobias"  # example
    
    #operation for best attitude error mean
    best = find_best(directory, MEAN)
    if not best:
        print("No valid files found (no 'Mean error' present).")
    else:
        print("==== Best result attitude ====")
        print(f"File: {best['file']}")
        print(f"Mean error: {best['mean']}")
        print("Parameters:")
        for k in sorted(best["params"]):
            print(f"  {k}: {best['params'][k]}")
    
    #operation for best angular velocity error mean
    best = find_best(directory, MEAN_AN)
    if not best:
        print("No valid files found (no 'Mean error' present).")
    else:
        print("==== Best result angular velocity ====")
        print(f"File: {best['file']}")
        print(f"Mean angular error: {best['mean']}")
        print("Parameters:")
        for k in sorted(best["params"]):
            print(f"  {k}: {best['params'][k]}")
