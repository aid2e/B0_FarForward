import os
import pandas as pd
import numpy as np
def ReturnSummedParams(ParamDict, SigDict):
    for key, value in SigDict.items():
        if(value[-1] > 0):
            val = 0.
            for i in range(1, value[0] + 1):
                tmpKey = key.replace("_fill_", f"{i}").strip(" ")
                val += ParamDict[tmpKey]
                ParamDict[tmpKey] = val
    return ParamDict

def CheckConstraints(ParamDict, Constraints, SigDict):
    Passed = True
    for key, value in Constraints.items():
        expresssion = ""
        val = 0.
        for i in range(1, SigDict[key][0] + 1):
            tmpKey = key.replace("_fill_", f"{i}").strip(" ")
            val += ParamDict[tmpKey]
        expression = f"{val}" + value
        Passed = (Passed and eval(expression))
    return Passed
def SingularityExecute(data, designFile, settingsFile, NoOfEvents, eta_bin, mom_bin, isSLURM = False):
    SingularityCmd = "singularity exec -B "
    for bind, target in zip(data["MOUNT_PATH"], data["BIND_PATH"]):
        SingularityCmd += f"{bind}:{target},"
    SingularityCmd = SingularityCmd[:-1] + " "# Removing final comma
    SingularityCmd += data["SINGULARITY_IMAGE"] + " /usr/bin/bash"
    bash_template = ""
    outputROOTFile = "ExecuteSingularity.root"
    with open(os.path.join(data["src_dir"], "Templates", "SINGULARITY_EXEC.bash")) as f:
        bash_template = f.read()
    bash_template = bash_template.replace("PARAMS", designFile)
    bash_template = bash_template.replace("SETTINGS", settingsFile)
    bash_template = bash_template.replace("NEVENTS", str(NoOfEvents))
    bash_template = bash_template.replace("ETAMIN", str(eta_bin[0]))
    bash_template = bash_template.replace("ETAMAX", str(eta_bin[-1]))
    bash_template = bash_template.replace("MOMMIN", str(mom_bin[0]))
    bash_template = bash_template.replace("MOMMAX", str(mom_bin[-1]))
    bash_template = bash_template.replace("OUTPUTFILE",outputROOTFile)
    with open("command.sh", "w") as f:
        f.write(bash_template)
    cmd = SingularityCmd + " /work/command.sh"
    print("Singularity command", cmd)
    if(isSLURM):
        with open(os.path.join(data["src_dir"], "Templates", "SLURM_TEMPLATE.sh")) as f:
            slurm_template = f.read()
            slurm_template = slurm_template.replace("SINGULARITY_COMMAND", cmd)
        with open("slurm.sh", "w") as f:
            f.write(slurm_template)
        os.system("sbatch slurm.sh")
    else:
        os.system("ls -lhtr *")
        os.system("pwd")
        os.system(cmd)
    return outputROOTFile
def CalcMetric(eta_range, which_plot, defaultName, currentName):  #which_plot = "dp_p_p"
    
    if(os.path.exists(defaultName)==False):
        print("BaseLine params not there. Try to run CheckDefault(NoOfEvents, True) to generate base line")
        return np.nan
    
    default = pd.read_csv(defaultName, sep = ",")	
    current = pd.read_csv(currentName, sep = ",")
    default = default[default["etarange"] == eta_range]
    current = current[current["etarange"] == eta_range]
    default.fillna(0., inplace = True)
    current.fillna(0., inplace = True)
    default.replace(np.inf, 0., inplace = True)
    current.replace(np.inf, 0., inplace = True)
    
    sel_default = default[which_plot]
    sel_default = sel_default.mask(sel_default < 1/10**6, 1/10**6) # making sure the values are not zero
    sel_current = current[which_plot]
    sel_current = sel_current.mask(sel_current < 1/10**6, 1/10**6)
    
    sel_default_error = default["error_" + which_plot]
    sel_default_error = sel_default_error.mask(sel_default_error < 1/10**6, 1/10**6) # making sure the errors are not zero
    
    sel_current_error = current["error_" + which_plot]
    sel_current_error = sel_current_error.mask(sel_current_error < 1/10**6, 1/10**6)
    
    
    Ratios = sel_current/sel_default
    
    RatioErrorsSquare = (Ratios**2)*(sel_current_error**2/sel_current**2 + sel_default_error**2/sel_default**2) #absolute_error_square the errors are added in quadrature
    Weights = RatioErrorsSquare**-1 # W = 1/sigma**2

    pd_size = float(current[which_plot].size)

    if pd_size<=0.:
        pd_size = 1.   
    assert pd_size >0        
        
    #exit()
    #print("Ratios", Ratios)
    #print("Weights", Weights)
    metric = (Ratios*Weights).sum()/Weights.sum()
    #metric = (1/pd_size)*(current[which_plot]/default[which_plot]).sum()
        
    #print("metric: ", metric)

    return metric

def CalcMetricWithoutDefault(eta_range, which_plot, fileName):  #which_plot = "dp_p_p"
    
    if(os.path.exists(fileName)==False):
        print("BaseLine params not there. Try to run CheckDefault(NoOfEvents, True) to generate base line")
        return np.nan
    	
    current = pd.read_csv(fileName, sep = ",")
    current = current[current["etarange"] == eta_range]
    current.fillna(0., inplace = True)
    current.replace(np.inf, 0, inplace = True)

    #print(default)
    #print("")
    #print(current)
    #print("")
    sel_current = current[which_plot]
    sel_current = sel_current.mask(sel_current < 1/10**6, 1/10**6)
    
    sel_current_error = current["error_" + which_plot]
    sel_current_error = sel_current_error.mask(sel_current_error < 1/10**6, 1/10**6)
    
    
    
    Weights = sel_current_error**-2 # W = 1/sigma**2
    #print(sel_default)
    #print("")
    #print(sel_current)
    #exit()
    print(sel_current)
    print(Weights)
    pd_size = float(current[which_plot].size)

    if pd_size<=0.:
        pd_size = 1.   
    assert pd_size >0        

    #exit()
    
    metric = (sel_current*Weights).sum()/Weights.sum()
    #metric = (1/pd_size)*(current[which_plot]/default[which_plot]).sum()
        
    #print("eta_range: ", eta_range, "quantity : ", which_plot, "metric: ", metric)

    return metric


def CalcMeanMetric(which_plot, UseDefault=True, FileSig = ""): #which_plot = "dp_p_p"
    
    if(os.path.exists("default_params" + FileSig + ".csv")==False):
        print("BaseLine params not there" + " default_params" + FileSig + ".csv")
        return np.nan
    Metric = []
    defaultName = "default_params" + FileSig + ".csv"
    currentName = "params" + FileSig + ".csv"
    default = pd.read_csv(defaultName, sep = ",")
    current = pd.read_csv(currentName, sep = ",")
    if(current["OverFlowUnderFlowFlag"].sum() > 0):
        return 1000. # Return if you have this Overflow flag
    UniqueEta = default["etarange"].unique()
    
    for etarange in UniqueEta:
        if(UseDefault): Metric.append(CalcMetric(etarange, which_plot, defaultName, currentName))
        else: Metric.append(CalcMetricWithoutDefault(etarange, which_plot, currentName)) # can be used if we choose not to compare with Baseline Efficiency
    MeanMetric = sum(Metric)/len(Metric) # Currently using Arithmetic mean.
    #print("Mean metric: ", MeanMetric)
    return MeanMetric

def exception_obj(BarrelRadii,BarrelLength, Fwd_Disk_Rmin, Fwd_Disk_Rmax, Fwd_Disk_z): 
    '''SHould we still neeed this Exception??? SInce we are using cone for optimising'''
    IsException = False 

    # check if the Disk radii overlaps with the barrel radii at a given z
    for Brl_r, Brl_z in zip(BarrelRadii, BarrelLength):
        for disk_rmin, disk_rmax, disk_z in zip(Fwd_Disk_Rmin, Fwd_Disk_Rmax, Fwd_Disk_z):
            #print(Brl_r, Brl_z/2, disk_rmin, disk_rmax, disk_z)
            if(Brl_z/2>=disk_z):
                if(disk_rmax+disk_rmin > Brl_r): # make sure the disk fits inside the barrel
                    IsException = True 
                    
                else: pass

    """
            else:
                if(disk_rmax+disk_rmin < Brl_r): # make sure the disk radius is not less than the barrel
                    print("Disk Radius is smaller than Barrel radius even when Disk is outside thebarrel \n")


                else: pass

    """

    return IsException

def CalcCostProxy(DesignParams, Signature):
    costProxy = 0.
    for key, value in DesignParams.items():
        if (Signature in key):
            costProxy += 1./value
    return costProxy
    
    
    
    
