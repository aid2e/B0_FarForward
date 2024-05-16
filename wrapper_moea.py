import numpy as np
import os
import sys
import pandas as pd
from utils_optim_withCone import *

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.display import Display

import matplotlib
import matplotlib.pyplot as plt

from pymoo.factory import get_performance_indicator
from pymoo.performance_indicator.hv import Hypervolume

from joblib import Parallel, delayed

import pickle
import dill

import argparse

import time # added to output the process time for the code to run

from timeit import default_timer as timer
import multiprocessing as mpr


from obj_fun import *

#---------------------------------------------------------------------------------#

# declaring the values used for specific optimisation globally Before this was given in the MyProblem super().__init__() definition


#            th1e,  th1h, th2e,  th2h, urw1, bz1,  dbz2, dbz3, dbz4, fz1,  dfz2, dfz3, dfz4, dfz5
xl=np.array([25.26, 22.2, 25.26, 22.2, 20.0, 25.0, 10.0, 10.0, 10.0, 25.0, 10.0, 10.0, 10.0, 10.0]) # should modify this
xu=np.array([45.00, 45.0, 45.00, 45.0, 45.0, 50.0, 30.0, 30.0, 30.0, 50.0, 30.0, 30.0, 30.0, 10.0])

baseline_pars = np.array([3.3, 2.40, 21., 1.68, 16.62, 3.93, 25., 24., 24., 24., 24.]) # should modify this

June28_opt_pars = np.array([ 3.19219215,  6.9826853,  22.98043814, 10.10739695, 10.61907787,  6.63733708, 28.75463846,  6.10207502, 18.74279563, 45.90987584, 24.94115232])

n_var = 14
n_obj = 3
n_constr = 2
NoOfEvents = -999
n_cores = mpr.cpu_count()

signature = "" # this will be the signature of the txt files (log files) that are produced

class MyProblem(Problem):


    def __init__(self):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,# Need to check this
                         xl=xl, # Should check this
                         xu=xu) # Should check this



    #--------- parallelized (2) ---------#
    def _evaluate(self, X, out, *args, **kwargs):

        #-----------------------------------------------------#
        # the first evaluation has len(X) = population size   #
        # from the second evaluation, len(X) = offspring size #
        #-----------------------------------------------------#

        #print("np.shape(X): ", np.shape(X))
        #print("type(X): ", type(X))

        num_pop = len(X)

        print("num_pop: ", len(X))
        #exit()
        n_vars  = np.shape(X)[1]   # dimension of design space
        n_obj   = self.n_obj #2   # dimension of objective space
        n_const = self.n_constr #0   # number of constraints


        print("n_vars: ", n_vars)

        #exit()

        num_cores = n_cores #mpr.cpu_count() #10
        #if(num_cores is not isinstance(num_cores,int)):
        #    print("\n\n\n WARNING: not founding number of cores: setting it to 1.\n\n\n")
        #    num_cores=1
        if num_cores<0:
            num_cores = 4
        backend = 'multiprocessing'


        iterations = num_pop//num_cores
        remainder  = num_pop - iterations*num_cores

        print("iterations*num_cores: ", iterations*num_cores, ", remainder: ", remainder)

        #exit()

        design_space = np.zeros((0,n_vars))
        fin_funcs = np.zeros((0,n_obj))
        list_obj = []

        for i in range(n_obj):
            list_obj.append(np.zeros((0,1)))

        logfile = open(signature + "progress.txt", "a+")
        #-------------- parallelized iterations
        for i in range(iterations):
            
            print("...iter.: ", i)

            Xtmp = X[i*num_cores:(i+1)*num_cores,:]
            print("np.shape(X): ", np.shape(X))
            print("np.shape(Xtmp): ", np.shape(Xtmp))


            #Ytmp = Parallel(n_jobs=num_cores, backend=backend)(delayed(tot_obj)(v) for v in Xtmp) #10
            Ytmp = Parallel(n_jobs=num_cores, backend=backend, verbose = 10)(delayed(myfun)(v, NoOfEvents) for v in Xtmp) #10
            Ytmp = np.asarray(Ytmp)

            # Added now

            FailedSimIndex = []
            for index, objs in enumerate(Ytmp):
                objs = objs[:n_obj]
                #print("\n\n ###################\nEntered the loop\n#######################\n\n", objs, all(objs == 9999), type(objs), Xtmp[index])
                if(all(objs == 9999)): 
                    logfile.write("\n------\n Stuck with the Setting " + str(Xtmp[index]) + "\n------\n")
                    Xtmp[index] = -9999
                    FailedSimIndex.append(index)
                    print("\n\n############\nThis is a point which did not complete fully \n")
                    print(objs, "\n--------\n", Ytmp[index])
            """
            print("Indices to rerun is :", FailedSimIndex)
            if(float(len(FailedSimIndex))/float(len(Ytmp)) > 0.1):
                    print("Rerunning ", len(FailedSimIndex), " jobs since it failed for points", [Xtmp[j] for j in FailedSimIndex])
                    Ytmp1 = Parallel(n_jobs=len(FailedSimIndex), backend = backend, verbose = 10)(delayed(myfun)(Xtmp[j]) for j in FailedSimIndex)
            
                    for ii, index in enumerate(FailedSimIndex):
                        Ytmp[index] = Ytmp1[ii]
            else:

                for ii in FailedSimIndex:
                    Xtmp[index] = -9999
            """            
            fin_funcs = np.append(fin_funcs, Ytmp)
            design_space = np.append(design_space, Xtmp)

        logfile.write("\n Cores used for this iter is " + str(num_cores) + " Time at end of each iter at " + str(time.strftime("%H:%M:%S", time.localtime())) + "\n")
        logfile.close()
        print(np.shape(fin_funcs), np.shape(design_space))
        #exit()

        #------------- remainder

        if(iterations*num_cores<num_pop):
            print("...remainder")
            logfile = open(signature + "progress.txt", "a+")
            Xtmp = X[iterations*num_cores : num_pop, :]
            #Ytmp = Parallel(n_jobs=num_cores, backend=backend)(delayed(tot_obj)(v) for v in Xtmp) #10
            #Ytmp = Parallel(n_jobs=num_cores, backend=backend, verbose = 10)(delayed(myfun)(v) for v in Xtmp)
            Ytmp = Parallel(n_jobs=remainder, backend=backend, verbose = 10)(delayed(myfun)(v, NoOfEvents) for v in Xtmp)  # 6/3/2021
            Ytmp = np.asarray(Ytmp)
            FailedSimIndex = []
            for index, objs in enumerate(Ytmp):
                objs = objs[:n_obj]
                #print("\n\n ###################\nEntered the loop\n#######################\n\n", objs, all(objs == 9999), type(objs), Xtmp[index])
                if(all(objs == 9999)): 
                    logfile.write("\n------\n Stuck with the Setting " + str(Xtmp[index]) + "\n------\n")
                    Xtmp[index] = -9999
                    FailedSimIndex.append(index)
                    print("\n\n############\nThis is a point which did not complete fully \n")
                    print(objs, "\n--------\n", Ytmp[index], Xtmp[index])
            logfile.write("\n Cores used for this iter is " + str(num_cores) + "\n Time at end of each reminder at " + str(time.strftime("%H:%M:%S", time.localtime())) + "\n")
            logfile.close()
            """
            print("Indices to rerun is :", FailedSimIndex)
            
            if(float(len(FailedSimIndex))/float(len(Ytmp)) > 0.1):
                    print("Rerunning ", len(FailedSimIndex), " jobs since it failed for points", [Xtmp[j] for j in FailedSimIndex])
                    Ytmp1 = Parallel(n_jobs=len(FailedSimIndex), backend = backend, verbose = 10)(delayed(myfun)(Xtmp[j]) for j in FailedSimIndex)
            
                    for ii, index in enumerate(FailedSimIndex):
                        Ytmp[index] = Ytmp1[ii]
            else:
                for ii in FailedSimIndex:
                    Xtmp[index] = -9999
            """
            fin_funcs = np.append(fin_funcs, Ytmp)
            design_space = np.append(design_space, Xtmp)


        #------------- reshaping
        fin_funcs = fin_funcs.reshape(num_pop,(n_obj+n_const))
        design_space = design_space.reshape(num_pop,n_vars)

        print("\n\n::::::::::::::::::::::::::::::::")
        print("\n\n final reshaping \n\n")
        print("np.shape(fin_funcs): ", np.shape(fin_funcs))
        print("np.shape(design_space): ", np.shape(design_space))
        print("::::::::::::::::::::::::::::::::\n\n")

        #-----------------------   passing to dictionary  -------------------------#
        obj_range = range(0,n_obj,1)
        con_range = range(n_obj, n_obj+n_const,1)
        #n_obj+n_const-1,n_obj-1,-1

        lf = []
        lc = []

        for i in obj_range:
            #print("....", i)
            lf.append(fin_funcs[:, i])


        #print("")
        for i in con_range:
            #print("....", i)
            lc.append(fin_funcs[:, i])

        #print(fin_funcs[:, 0])

        if(len(obj_range)>0):
            out["F"] = np.column_stack(lf)
            #print("np.shape(out[F]): ", np.shape(out["F"]))
            #print("type(out[F]): ", type(out["F"]))
            #(15,2)
            #print(out["F"]), e.g., (15,2)
            #[[1.19076029 0.51869359]
            # [0.65939737 1.04496592]
            # ...
            # [0.81689562 3.50268191]]

        if(len(con_range)>0):
            out["G"] = np.column_stack(lc)



        #--------------------------------------------------------------------------#
        print("")
        print("........ stacked objectives ")

        print(out["F"])
        print("........ stacked constraint Functions ")
        print(out["G"])

        print("np.shape(out[\"F\"])",np.shape(out["F"]))
        print("np.shape(X)",np.shape(X))



class MyDisplay(Display):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("metric_a", np.mean(algorithm.pop.get("X")))
        self.output.append("metric_b", np.mean(algorithm.pop.get("CV")))

def ContinueWithCheckpoints(npyfilename, problem, Interval, n_gen, n_offspring):
    
    
    checkpoint, = np.load(npyfilename, allow_pickle = True).flatten()
    print("Loaded Checkpoint and it has {} gen : ".format(checkpoint.n_gen), checkpoint)
    print("The interval is : ", Interval)
    #sys.exit()
    
    with open(signature + "OptSettingSummary.txt", "a+") as optsum: 
        optsum.write("The loaded checkpoint " + npyfilename + " has {} generations in it at the start of this optimisation".format(checkpoint.n_gen))
    
    checkpoint.has_terminated = False

    completed_checkpoints = checkpoint.n_gen

    TotalGen = completed_checkpoints + n_gen

    res = minimize(problem,
                   checkpoint,
                   ("n_gen", completed_checkpoints + Interval),
                   verbose=True, copy_algorithm=False,
                   seed=123,
                   save_history=True) #200 n_gen
                   #display=MyDisplay()) you can choose what to print
                   #save_history with a deepcopy can be memory intensive
                   #used for plotting convergence.
                   #For MOO, report HyperVolume + Constraint Violation    
    count = 0
    np.save("checkpoint", res.algorithm)

    completed_checkpoints += Interval

    if not os.path.exists("./pkl_dir"):
        os.makedirs("./pkl_dir")
    out_t_filename = "./pkl_dir/" + signature + "_global_optmisation{}.pkl".format(completed_checkpoints)

    with open(out_t_filename,'wb') as multi_file_survey:
        out_list = [res.X, res.F, res]
        dill.dump(out_list, multi_file_survey)

    while(completed_checkpoints<TotalGen):
        
        checkpoint, = np.load("checkpoint.npy", allow_pickle = True).flatten()
        print("Loaded Checkpoint and it has {} gen :".format(res.algorithm.n_gen), checkpoint)
        checkpoint.has_terminated = False
        res = minimize(problem,
                       checkpoint,
                       ("n_gen", completed_checkpoints + Interval),
                       verbose=True, copy_algorithm=False,
                       seed=123,
                       save_history=True) #200 n_gen
                       #display=MyDisplay()) you can choose what to print
                       #save_history with a deepcopy can be memory intensive
                       #used for plotting convergence.
                       #For MOO, report HyperVolume + Constraint Violation 
        count+=1
        np.save("checkpoint", res.algorithm)
        completed_checkpoints = completed_checkpoints + min(Interval, abs(TotalGen - completed_checkpoints))
        out_t_filename = "./pkl_dir/" + signature + "_global_optmisation{}.pkl".format(completed_checkpoints)


        with open(out_t_filename,'wb') as multi_file_survey:
            out_list = [res.X, res.F, res]
            dill.dump(out_list, multi_file_survey)
    return res



def RunWithCheckpoints(problem, Interval, n_gen, pop_size, n_offspring):
    
    
    # Adding some lines to choose the population 1st August 2021
    
    PopX = np.zeros((pop_size, len(xl)))
    pop_count = 0
    while pop_count < pop_size:
        pop_design_points = DesignParams(xl, xu)
        if(sum(pop_design_points[2:6]) < 51. and sum(pop_design_points[6:11]) < 125.): # 
            PopX[pop_count] = pop_design_points
            pop_count+=1
    
    PopX[-1] = baseline_pars
    #PopX[-2] = June28_opt_pars
    algorithm = NSGA2(pop_size=pop_size, n_offsprings=n_offspring, sampling = PopX) #n_offsprings=10   #200,30
    
    res = minimize(problem,
                   algorithm,
                   ("n_gen", Interval),
                   verbose=True, copy_algorithm=False,
                   seed=123,
                   save_history=True) #200 n_gen
                   #display=MyDisplay()) you can choose what to print
                   #save_history with a deepcopy can be memory intensive
                   #used for plotting convergence.
                   #For MOO, report HyperVolume + Constraint Violation    
    count = 0
    np.save("checkpoint", algorithm)
    completed_checkpoints = Interval
    if not os.path.exists("./pkl_dir"):
        os.makedirs("./pkl_dir")

    out_t_filename = "./pkl_dir/" + signature + "_global_optmisation{}.pkl".format(completed_checkpoints)


    with open(out_t_filename,'wb') as multi_file_survey:
        out_list = [res.X, res.F, res]
        dill.dump(out_list, multi_file_survey)

    print("\nSuccessfuly completed 1st checkpoint\n")
    while(completed_checkpoints<n_gen):
        
        checkpoint, = np.load("checkpoint.npy", allow_pickle = True).flatten()
        print("Loaded Checkpoint and it has {} gen :".format(res.algorithm.n_gen), checkpoint)
        checkpoint.has_terminated = False
        res = minimize(problem,
                       checkpoint,
                       ("n_gen", completed_checkpoints + Interval),
                       verbose=True, copy_algorithm=False,
                       seed=123,
                       save_history=True) #200 n_gen
                       #display=MyDisplay()) you can choose what to print
                       #save_history with a deepcopy can be memory intensive
                       #used for plotting convergence.
                       #For MOO, report HyperVolume + Constraint Violation 
        count+=1
        np.save("checkpoint", res.algorithm)
        completed_checkpoints = completed_checkpoints + min(Interval, abs(n_gen - completed_checkpoints))
        out_t_filename = "./pkl_dir/" + signature + "_global_optmisation{}.pkl".format(completed_checkpoints)


        with open(out_t_filename,'wb') as multi_file_survey:
            out_list = [res.X, res.F, res]
            dill.dump(out_list, multi_file_survey)
    return res

#-----------------------------------#
#          Optimization             #
#-----------------------------------#
#------------------------------------------------------------------------------#

StartTime = time.time() # added to calculate the startime

if __name__=="__main__":


    n_gen = 6
    pop_size = 10
    n_offspring = 2
    wrapper_state = "warmup"
    numpy_checkpoint = ""

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-saveInt', '--saveInt',help='Checkpoint Interval', required = False)
    parser.add_argument('-calls','--calls',help='number of calls',required=True)
    parser.add_argument('-cores','--cores',help='number of cores',required=False)
    parser.add_argument('-population','--population',help='population size',required=False)
    parser.add_argument('-offspring','--offspring',help='offspring size',required=False)
    parser.add_argument('-wrapper_state', '--wrapper_state', help = "what to run can be one of [warmup, fresh, continue]", required = False)
    parser.add_argument('-numpy_checkpoint', '--numpy_checkpoint', help = "the lastest checkpoint", required = False)  
    parser.add_argument('-testing', '--testing', help = "[-testing test], testing option makes the NoOfEvents 1000, Use only when testing.", required = False)  
    

    args=parser.parse_args()
    


    if(args.calls!=None):
        n_gen = int(args.calls)

    
    if(args.cores!=None):
        n_cores = int(args.cores)

    if(args.population!=None):
        pop_size = int(args.population)

    if(args.offspring!=None):
        n_offspring = int(args.offspring)   
    
    Interval = 1
    if(args.saveInt!=None):
        Interval = int(args.saveInt)

    if(args.wrapper_state != None):
        wrapper_state = args.wrapper_state

    if(wrapper_state == "continue" and args.numpy_checkpoint != None):
        numpy_checkpoint = args.numpy_checkpoint

    
    if(wrapper_state == "warmup" or args.testing == "test"):
        NoOfEvents = 1000
    elif(wrapper_state == "fresh" or wrapper_state == "continue"):
        NoOfEvents = 80000
    else:
        print("\n ######## ERROR ######## \n NoOf Events Invalid, NoOfEvents is : {} \n #################\n".format(NoOfEvents))
        sys.exit()
    
    assert n_offspring<pop_size, "population must be larger than offspring"
    assert n_offspring>=2, "required at least two parents"
    assert n_gen>=5, "at least 5 calls required"

    problem = MyProblem()

    signature = wrapper_state  # all log files will have this signature before the log files
   
    with open(signature + "OptSettingSummary.txt", "w") as optsum:
        Comment = "The optimisation is running at : {} \n".format(os.uname()[1]) # getting the information of the farm node.
        Comment += "The Time at which the optimisation is started is : {} \n".format(str(time.strftime("%H:%M:%S", time.localtime())))
        Comment += "No of calls made in -calls is : {} \n".format(n_gen)
        Comment += "Population size is : {} \n".format(pop_size)
        Comment += "off spring size is : {} \n".format(n_offspring)
        Comment += "Interval is : {} \n".format(Interval)
        Comment += "The optmisation is run in the wrapper_state : {} \n".format(wrapper_state)
        Comment += "No of Events generated for each gene is : {} \n".format(NoOfEvents)
        if(wrapper_state == "continue"):
            Comment += "Loading from the numpy file : {} \n".format(numpy_checkpoint) 
        
        optsum.write(Comment)

    
    if(wrapper_state == "fresh" or wrapper_state == "warmup"):
        with open(signature + "OptSettingSummary.txt", "a+") as optsum:
            optsum.write("Calling the function : RunWithCheckpoints(problem, Interval = {}, n_gen = {}, pop_size = {}, n_offspring = {}) \n".format(Interval, n_gen, pop_size, n_offspring))
        res = RunWithCheckpoints(problem, Interval, n_gen, pop_size, n_offspring)
        
    
    elif(wrapper_state == "continue"):
        with open(signature + "OptSettingSummary.txt", "a+") as optsum:
            optsum.write("Calling the function : ContinueWithCheckpoints(numpy_checkpoint = {}, problem, Interval = {}, n_gen = {}, n_offspring = {}) \n".format(numpy_checkpoint,  Interval, n_gen, n_offspring))
        res = ContinueWithCheckpoints(numpy_checkpoint, problem, Interval, n_gen, n_offspring)
    else:
        print("wrapper_state can only be [fresh, warmup, continue]")
        print("wrapper state given is ", wrapper_state)
        sys.exit()


    
    #https://pymoo.org/getting_started.html
    #algorithm = NSGA2(pop_size=pop_size, n_offsprings=n_offspring) #n_offsprings=10   #200,30
    
    #print(type(algorithm))
    #res = minimize(problem,
    #               algorithm,
    #               ("n_gen", n_gen),
    #               verbose=True,
     #              seed=123,
     #              save_history=True) #200 n_gen
                   #display=MyDisplay()) you can choose what to print
                   #save_history with a deepcopy can be memory intensive
                   #used for plotting convergence.
                   #For MOO, report HyperVolume + Constraint Violation
        
    #plt.figure()
    #plot = Scatter()
    #plot.add(res.F, color="red")
    #plot.show()
    #plt.savefig("pareto2.png")

    print("\n\n :::::::: FINAL SURVEY :::::::: ")

    print("\n\nnp.shape(res.F): ", np.shape(res.F))

    print("\n res.X: ", res.X)
    print("\n res.F: ", res.F)
    print("\n res.pop: ", res.pop)
    print("\n res.history: ", res.history)
    
    print("\n\nnp.shape(res.pop): ", np.shape(res.pop))



    #--------------------------#
    #      store res           #
    #--------------------------#
    #------------------------------------------------------------------------------#
    
    if not os.path.exists("./pkl_dir"):
        os.makedirs("./pkl_dir")
    
    out_t_filename = "./pkl_dir/" + signature + "_global_optmisation.pkl"


    with open(out_t_filename,'wb') as multi_file_survey:
        out_list = [res.X, res.F, res]
        dill.dump(out_list, multi_file_survey)



with open(signature + "TimeTaken.txt", "w") as timefile: # output the total time
    timefile.write("Time taken for the total optimisation is : {}".format(time.time() - StartTime))

try:
    os.system("mv Failed_Setting_Commands.txt " + signature + "_Failed_Setting_Commands.txt")
except:
    pass    

