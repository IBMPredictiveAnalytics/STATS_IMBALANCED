import spss, spssaux, spssdata
from extension import Template, Syntax, processcmd
from spssdata import Spssdata, vdef
import random, sys
import numpy as np

#try:
    #import pandas as pd
#except:
    #print(installmsg % "pandas")
    #spss.Submit("""STATS PACKAGE INSTALL /PYTHON pandas.""")
#import pandas
#import random

#try:
    #from imblearn import over_sampling
#except:
    #print(installmsg % "imblearn")
    #spss.Submit("""STATS PACKAGE INSTALL /PYTHON imbalanced-learn""")

#from imblearn import over_sampling
#from imblearn import under_sampling
#from imblearn import combine
#from imblearn import __version__ as imbver

## debugging
try:
    import wingdbstub
    import threading
    wingdbstub.Ensure()
    wingdbstub.debugger.SetDebugThreads({threading.get_ident(): 1})
except:
    pass

# temporary
global _
try:
    _("---")
except:
    def _(msg):
        return msg

TFdict = {True: _("Yes"), False: _("No")}

# do STATS IMBALANCED command

def doimb(dep, indep, dataset, method,
    strategy=None, strategyvalue=None, strategylist=None,
    shrinkage=None, seed=None, freq=True, summary=False, 
    kneighbors=5, mneighbors=10, nneighbors=3, borderlinekind="type1",
    voting="soft", kindsel="alln", allowminority=False,
    replacement=False, nseeds=1, outstep=0.5
    ):
    
    # Installs for these imports should have been done with the STATS IMBALANCED
    # install, but sometimes that fails, so trying to bootstrap them in here.
    # Done here rather than at the top due to the need for the _ function.
    installmsg = _("""The %s package is not installed.  Attempting
    install with the STATS PACKAGE INSTALL extension command.  If
    that command is not installed, please install it via Extensions > Extension Hub
    and try again.""")   
    try:
        import pandas as pd
    except:
        print(installmsg % "pandas")
        spss.Submit("""STATS PACKAGE INSTALL /PYTHON pandas.""")
    import pandas as pd

    try:
        from imblearn import over_sampling
    except:
        print(installmsg % "imblearn")
        spss.Submit("""STATS PACKAGE INSTALL /PYTHON imbalanced-learn""")
    
    from imblearn import over_sampling
    from imblearn import under_sampling
    from imblearn import combine
    from imblearn import __version__ as imbver


    if isinstance(dataset, list):
        dataset = dataset[0]
      
    if dep in indep:
        raise ValueError(_(f"""The target variable appears in the independent list: {dep}"""))
    varnames = indep + [dep]
    # Validation

    if isinstance(dep, list):
        dep = dep[0]
    dsactivename = spss.ActiveDataset()

    if spss.GetWeightVar() is not None:
        print(_("Warning: this procedure does not support weights."))
    if spss.GetSplitVariableNames():
        print(_("Warning: this procedure does not support split file settings."))
    if dsactivename == dataset:
        raise ValueError(_("The output dataset name must be different from the input name"))
    # spss parser is handling "all" incorrectly
    if kindsel == "alln":
        kindsel = "all"

    if dsactivename == "*":
        makename = True
        # assign a temporary name and remove it at the end
        dsactivename = "D" + str(random.uniform(.05, 1))
        spss.Submit(f"""DATASET NAME {dsactivename}""")
    else:
        makename = False
        
    vardict = spssaux.VariableDict(caseless=True)

        # case correct variable names
    varnames = [vardict[v].VariableName for v in varnames]
    vartypes = [vardict[v].VariableType for v in varnames]
    strategyspec = makestrat(strategy, strategyvalue, strategylist, vardict[dep].VariableType)
    
    # get data with all missing values set to None
    # Need a better way to do this retrieval

    # dta comes in as tuple of tuples with string values the full length of the variable
    # Some methods do not support missing values of any type
    omitmv = method in ["clustercentroids", "onesidedselection", "editednearestneighbors",
        "smoteenn", "smotenc", "kmeanssmote", "borderlinesmote", "smote", "svmsmote", "adasyn", "allknn",
        "smoten"]
    dta = spssdata.Spssdata(indexes=varnames, names=False, omitmissing=omitmv, convertUserMissing=True).fetchall()
    validcases = len(dta)
    dta = pd.DataFrame(dta, columns=varnames)
        
    class Gendata:

        def __init__(self, dta, imbver, validcases):
            """dta is a pandas dataframe with the dep variable in the last column"""

            self.dta0, self.dta1 = dta.iloc[:, 0:dta.shape[1]-1], dta.iloc[:, -1]
            if self.dta1.isnull().values.any():
                print(_("""Error: Cases with missing values in the target variable
must be excluded from the data passed to this procedure."""))
                raise
            self.imbver = imbver
            self.validcases = validcases
          
        # oversample    
        def compute_random(self):
            s = over_sampling.RandomOverSampler(sampling_strategy=strategyspec,
                random_state=seed, shrinkage=shrinkage)
            res = s.fit_resample(self.dta0, self.dta1)
            self.params = s.get_params()
            return res
        
        def compute_smote(self):
            # strings not allowed
            if max(vartypes[:-1]) > 0:
                raise ValueError(_(f"Error: String variables cannot be used with this method ({method})"))             
            sm = over_sampling.SMOTE(sampling_strategy=strategyspec, random_state=seed, k_neighbors=kneighbors)
            res = sm.fit_resample(self.dta0, self.dta1)
            self.params = sm.get_params()
            return res
        
        #The borderline SMOTE — cf. to the BorderlineSMOTE with the parameters 
        # kind='borderline-1' and kind='borderline-2' — will classify each sample x(i)
        #to be (i) noise (i.e. all nearest-neighbors are from a different class than the one of 
        #), (ii) in danger (i.e. at least half of the nearest neighbors are from the same class than 
        #, or (iii) safe (i.e. all nearest neighbors are from the same class than x(i)
        #Borderline-1 and Borderline-2 SMOTE will use the samples in danger 
        # to generate new samples. 
        # In Borderline-1 SMOTE, x(z,i)
        #will belong to the same class than the one of the sample x(i)
        #. On the contrary, Borderline-2 SMOTE will consider x(z,i)
        #which can be from any class.        
        def compute_borderlinesmote(self):
            nonlocal borderlinekind
            borderlinekind = borderlinekind == "type1" and "borderline-1" or "borderline-2"
            sm = over_sampling.BorderlineSMOTE(sampling_strategy=strategyspec, random_state=seed, k_neighbors=kneighbors,
                m_neighbors=mneighbors, kind=borderlinekind)
            res = sm.fit_resample(self.dta0, self.dta1)
            self.params = sm.get_params()
            return res
        
        def compute_smotenc(self):
            if max(vartypes[:-1]) > 0:
                raise ValueError(_(f"Error: String variables cannot be used with this method: ({method})"))
            catfeatures = getcat(varnames, vardict)
            lencat = len(catfeatures)
            if lencat == 0 or lencat == len(varnames) - 1:
                raise ValueError(_(f"Error: At least one categorical and one scale variable are required to use this method: {method}."))
            sm = over_sampling.SMOTENC(catfeatures, sampling_strategy=strategyspec, random_state=seed,
                k_neighbors=kneighbors)
            res = sm.fit_resample(self.dta0, self.dta1)
            self.params = sm.get_params()
            return res
            
        def compute_smoten(self):
            if max(vartypes[:-1]) > 0:
                raise ValueError(_(f"Error: String variables cannot be used with this method ({method})"))             
            catfeatures = getcat(varnames, vardict)
            if len(catfeatures) != len(varnames) - 1:
                raise ValueError(_(f"Error: Only categorical variables can be used with this method ({method})"))
            smo = over_sampling.SMOTEN(categorical_encoder=None, sampling_strategy=strategyspec,
                random_state=seed, k_neighbors=kneighbors)
            res = smo.fit_resample(self.dta0, self.dta1)
            self.params = smo.get_params()
            return res            
            
        def compute_svmsmote(self):
            svm = over_sampling.SVMSMOTE(sampling_strategy=strategyspec, random_state=seed,
                k_neighbors=kneighbors, m_neighbors=mneighbors, out_step=outstep)
            res = svm.fit_resample(self.dta0, self.dta1)
            self.params = svm.get_params()
            return res
        
        def compute_adasyn(self):
            # no strings
            if max(vartypes[:-1]) > 0:
                raise ValueError(_(f"Error: String variables cannot be used with this method ({method})"))            
            ada = over_sampling.ADASYN(sampling_strategy=strategyspec, random_state=seed,
                n_neighbors=mneighbors)
            res = ada.fit_resample(self.dta0, self.dta1)
            self.params =ada.get_params()
            return res            
            
        def compute_kmeanssmote(self):
            # No strings
            if max(vartypes[:-1]) > 0:
                raise ValueError(_(f"Error: String variables cannot be used with this method ({method})"))
            # use defaults for kmeans_estimator, cluster_balance_threshold, density_exponent)
            kms = over_sampling.KMeansSMOTE(sampling_strategy=strategyspec, random_state=seed,
                k_neighbors=kneighbors)
            res = kms.fit_resample(self.dta0, self.dta1)
            self.params =kms.get_params()
            return res         
        
        #Method that under samples the majority class by replacing a cluster of majority 
        # samples by the cluster centroid of a KMeans algorithm. 
        # This algorithm keeps N majority samples by fitting the KMeans 
        # algorithm with N cluster to the majority class and using the 
        # coordinates of the N cluster centroids as the new majority samples.    
        def compute_clustercentroids(self):
            # no strings
            if max(vartypes[:-1]) > 0:
                raise ValueError(_(f"Error: String variables cannot be used with this method ({method})"))                
            us = under_sampling.ClusterCentroids(sampling_strategy=strategyspec, random_state=seed,
            voting=voting)
            res = us.fit_resample(self.dta0, self.dta1)
            self.params = us.get_params()
            return res
        
        # undersample
        def compute_editednearestneighbors(self):
            if max(vartypes[:-1]) > 0:
                raise ValueError(_(f"Error: String variables cannot be used with this method ({method})"))             
            # does not accept random_state
            enn =  under_sampling.EditedNearestNeighbours(sampling_strategy=strategyspec,
                kind_sel=kindsel)
            res = enn.fit_resample(self.dta0, self.dta1)
            self.params = enn.get_params()
            return res
        
        def compute_allknn(self):
            # random_state not supported by this procedure
            if max(vartypes[:-1]) > 0:
                raise ValueError(_(f"Error: String variables cannot be used with this method ({method})"))             
            ak = under_sampling.AllKNN(sampling_strategy=strategyspec, n_neighbors=nneighbors,
                kind_sel=kindsel, allow_minority=allowminority) 
            res =  ak.fit_resample(self.dta0, self.dta1)
            self.params = ak.get_params()
            return res
        
        def compute_randomundersampler(self):   # allows strings
            ru = under_sampling.RandomUnderSampler(sampling_strategy=strategyspec, random_state=seed,
                replacement=replacement)
            res =  ru.fit_resample(self.dta0, self.dta1)
            self.params = ru.get_params()
            return res
        
        def compute_onesidedselection(self):
            if max(vartypes[:-1]) > 0:
                raise ValueError(_(f"Error: String variables cannot be used with this method ({method})"))             
            ###seed2 = seed
            u = under_sampling.OneSidedSelection(sampling_strategy=strategyspec, random_state=seed,
                n_neighbors=mneighbors, n_seeds_S=nseeds)
            res =  u.fit_resample(self.dta0, self.dta1)
            self.params = u.get_params()
            return res
        
        # combine
        
        # use default smote and een objects
        def compute_smoteenn(self):
            if max(vartypes[:-1]) > 0:
                raise ValueError(_(f"Error: String variables cannot be used with this method ({method})"))
            smo =  combine.SMOTEENN(sampling_strategy='auto', random_state=seed)
            res = smo.fit_resample(self.dta0, self.dta1)
            self.params = smo.get_params()
            return res
        
        methodmap = {"random": compute_random,
        "borderlinesmote": compute_borderlinesmote,
        "smote": compute_smote,
        "smotenc": compute_smotenc,
        "smoten": compute_smoten,
        "svmsmote": compute_svmsmote,
        "adasyn": compute_adasyn,
        "kmeanssmote": compute_kmeanssmote,
        "clustercentroids": compute_clustercentroids,
        "editednearestneighbors": compute_editednearestneighbors, 
        "smoteenn": compute_smoteenn, 
        "allknn": compute_allknn,
        "randomundersampler": compute_randomundersampler,
        "onesidedselection": compute_onesidedselection}
        
        def dogen(self):
            """Invoke the appropriate method and return the augmented dataframe"""
            
            try:
                res = Gendata.methodmap[method](self)
            except:
                print(sys.exc_info()[1].args[0])
                raise 
            # put the dependent variable back in the first dataframe
            res = res[0].join(res[1].iloc[:])
            makeparametertable(method, self.params, dataset, varnames, self.imbver, self.validcases)
            return res        
        
    # tasks
    spss.StartProcedure(_("Balance Data"))

    try:
        gen = Gendata(dta, imbver, validcases)
        dta = gen.dogen()
    except:
        spss.EndProcedure()
        return
        
    # create new dataset
    createnewdataset(varnames, dta, dataset, dsactivename,
        vardict)
    spss.EndProcedure()
    if freq or summary:
        spss.Submit(f"""DATASET ACTIVATE {dataset}.""")
    if freq:
        spss.Submit(f"""FREQUENCIES VARIABLES={dep}.""")
    if summary:
        vardict = spssaux.VariableDict()
        catvars = [vardict[v].VariableName for v in getcat(varnames, vardict)]
        scalevars = [v for v in varnames[:-1] if not v in catvars]
        spss.Submit(f"""SORT CASES BY {dep}.
SPLIT FILES BY {dep}.""")
        if catvars:
            spss.Submit(f"""FREQUENCIES VARIABLES={" ".join(catvars)}""")
        if scalevars:
            spss.Submit(f"""DESCRIPTIVES VARIABLES = {" ".join(scalevars)}.""")
        spss.Submit("""SPLIT FILES OFF.""")
    
    spss.Submit(f"""DATASET ACTIVATE {dsactivename}.""")
    # If temporary dataset name was assigned to the active file, remove it (which leaves ds open)
    if makename:
        spss.Submit(f"""DATASET CLOSE {dsactivename}.""")


omitteditems = set(["n_jobs"])

def makeparametertable(method, parms, dataset, varnames, imbver, validcases):
    """Produce table of sampling parameters"""
    
    target = varnames[-1]
    pt = spss.BasePivotTable(_("Sampling Parameters"), "SamplingParameters",
        caption = f"""Using Imbalanced-learn version {imbver}""")
    rows = []
    cells = []
    rows.append(_("Method"))
    cells.append(method)
    rows.append(_("Valid Input Cases"))
    cells.append(validcases)
    rows.append(_("Output Dataset"))
    cells.append(dataset)
    rows.append(_("Target"))
    cells.append(target)
    for k, v in parms.items():
        if k in omitteditems:
            continue
        rows.append(k)
        if k == "categorical_features":
            v = "\n".join(varnames[x] for x in v)
        if isinstance(v, dict):
            # merge dictionary to a single string
            allvalues = []
            for kk, vv in v.items():
                allvalues.append(f"""{kk}:{vv}""")
            cells.append("\n".join(allvalues))
        else:
            if v is None:
                cells.append("---")
            else:
                cells.append(v)

    for i, v in enumerate(cells):
        if not isinstance(cells[i], str):
            cells[i] = str(cells[i])
    pt.SimplePivotTable(rowlabels=rows, 
        collabels=[_("Values")], 
        cells=cells)
    

    ###spss.EndProcedure()

def getcat(varnames, vardict):
    """Return a list of categorical variable indexes for independent variables"""
    
    catindexes = []
    for i, v in enumerate(varnames[:-1]):
        if vardict[v].VariableLevel in ['nominal', 'ordinal']:
            catindexes.append(i)
    return catindexes
    
    
def createnewdataset(varnames, dta, dataset, activedataset,
        vardict):
    """Create a new, possibly temporary dataset with variables dtaid, and varnames
    
    varnames is the names of the input variables
    dataset is the name for the output dataset
    activedataset is the name of the active (input) dataset
    vardict is a variable dictionary"""    

    spss.EndProcedure()
    
    
    # create new dataset for output
    curs = Spssdata(accessType="n", maxaddbuffer=len(varnames) * 8)
    for vname in varnames:
        curs.append(vdef(vname, vtype=vardict[vname].VariableType))
    curs.commitdict()

    # create cases for transformed variables
    # Numpy nan's must be converted to None for SPSS
    for i in range(dta.shape[0]):
        for j, v in enumerate(varnames):
            val = dta.iloc[i, j]
            try:                
                if np.isnan(val):
                    val = None
            except:
                pass   # ignore invalid type error as value would not be NaN
            curs.appendvalue(v, val)
        curs.CommitCase()
    curs.CClose()
    spss.Submit(f"""DATASET NAME {dataset}.""")
    
    varstr = " ".join(varnames)
    # omitting WIDTH attribute as it may generate false warning
    cmd = f"""APPLY DICTIONARY FROM {activedataset}
    /TARGET VARIABLES= {varstr}
    /VARINFO ALIGNMENT ATTRIBUTES FORMAT LEVEL ROLE VARLABEL VALLABELS MISSING."""
    spss.Submit(cmd)
    spss.Submit(f"""DATASET ACTIVATE {activedataset}.""")



    
    
sdict = {"notmajority": "not majority", "minority": "minority", "notminority": "not minority",
    "all": "all"}

def makestrat(strategy, strategyval, strategylist, dvtype):
    """return strategy argument for sampling functions
    
    strategy is a lower-cased string as in sdict
    strategyval is a single float.
    It is the desired ratio of the number of samples in the
      minority class over the number of samples in the majority class after resampling.
    strategylist is a list of literals followed by integer counts
    (Natural pairs can't be used, because the CDB table control pastes by column)
      The keys correspond to the targeted classes.
      The values correspond to the desired number of samples for each targeted class.
    Only one can be other than None or [None]
    
    """
    
    items = [strategy is not None, strategyval is not None, strategylist != None]
    s = sum(items)
    if s > 1:
        raise ValueError(_(f"Only one strategy is allowed.  {s} were found"))
    if s == 0:
        return "auto"
    if strategy:
        if not strategy in sdict:
            raise ValueError(_(f"Invalid strategy type: {strategy}"))
        return sdict[strategy]
    if strategyval is not None:
        return strategyval   # already a float
    
    # a strategylist looks like catval catval catval ...integer integer integer ...
    if strategylist:
        if len(strategylist) % 2 != 0:
            raise ValueError(_("""Number of category values does not match number of categories."""))
        ncats = len(strategylist) // 2
        dkeys = [strategylist[item] for item in range(ncats)]
        try:
            dvalues = [int(float(strategylist[item])) for item in range(ncats, len(strategylist))]
        except:
            raise ValueError(_("""A category count for the target variable is not an integer."""))        
        if dvtype == 0:
            try:
                dkeys = [int(k) for k in dkeys]
            except:
                raise ValueError(_("""A category value for a numeric target variable is not an integer."""))
        else:
            dkeys = [adjustcategory(k, dvtype) for k in dkeys]
            
        return dict(zip(dkeys, dvalues))
        
def adjustcategory(value, vartype):
    """check and adjust category value according to the variable type
    
    value is the category value as a string
    vartype is the target variable type"""
    
    if vartype == 0:
        try:
            return float(value)
        except:
            raise ValueError(_(f"""A category value for a numeric target variable is not a number: {value}."""))
    else:
        value2 = value.rstrip()
        catlen = len(value2)
        if catlen > vartype:
            raise ValueError(_(f"A strategy category value is too long for the target variable: {value}"))
        # return value padded out to declared variable length
        return value2 + (vartype - catlen) * " "    # noop if length already matches
    
def  Run(args):
    """Execute the STATS IMBALANCED command"""

    args = args[list(args.keys())[0]]

    oobj = Syntax([
        Template("DEP", subc="", ktype="existingvarlist", var="dep", islist=False), 
        Template("INDEP", subc="",  ktype="existingvarlist", var="indep", islist=True),
        Template("DATASET", subc="", ktype="varname", var="dataset", islist=False), 
        Template("METHOD", subc="", ktype="str", var="method", islist=False,
            vallist=[ # oversample
                "random", "borderlinesmote", "smote", "smotenc", "smoten",
                "svmsmote", "adasyn", "kmeanssmote",
                # undersample
                "clustercentroids", "randomundersampler", "onesidedselection",
                "editednearestneighbors", "allknn", 
                # combine
                "smoteenn"]), 
        Template("STRATEGY", subc="", ktype="str", var="strategy", islist=False,
            vallist=["notmajority", "minority", "notminority", "all"]),
        Template("STRATEGYVAL", subc="", ktype="float", var="strategyvalue", islist=False,
            vallist=[.000001]), 
        Template("STRATEGYLIST", subc="", ktype="literal", var="strategylist", islist=True),
        
        Template("SHRINKAGE", subc="OPTIONS", ktype="float", var="shrinkage"),
        Template("SEED", subc="OPTIONS", ktype="int", var="seed"), 
        Template("KNEIGHBORS", subc="OPTIONS", ktype="int", var="kneighbors", 
            vallist=[1, ]),
        Template("NNEIGHBORS", subc="OPTIONS", ktype="int", var="nneighbors", 
            vallist=[1, ]),        
        Template("MNEIGHBORS", subc="OPTIONS", ktype="int", var="mneighbors",
            vallist=[1]),
        Template("BORDERLINEKIND", subc="OPTIONS", ktype="str", var="borderlinekind",
            vallist=["type1", "type2"]),
        Template("VOTING", subc="OPTIONS", ktype="str", var = "voting",
            vallist=["hard", "soft"]),
        Template("KINDSEL", subc="OPTIONS", ktype="str", var="kindsel",
            vallist = ["alln", "mode"]),
        Template("ALLOWMINORITY", subc="OPTIONS", ktype="bool", var="allowminority"), 
        Template("REPLACEMENT", subc="OPTIONS", ktype="bool", var="replacement"),
        Template("NSEEDS", subc="OPTIONS", ktype="int", var="nseeds"),
        Template("OUTSTEP", subc="OPTIONS", ktype="float", var="outstep"),
        Template("TARGETFREQ", subc="OPTIONS", ktype="bool", var="freq"),
        Template("SUMMARIES", subc="OPTIONS", ktype="bool", var="summary")
        
        ])
        
        
        
    #enable localization
    global _
    try:
        _("---")
    except:
        def _(msg):
            return msg

    # A HELP subcommand overrides all else
    if "HELP" in args:
        #print helptext
        helper()
    else:
        processcmd(oobj, args, doimb, vardict=spssaux.VariableDict())

def helper():
    """open html help in default browser window
    
    The location is computed from the current module name"""
    
    import webbrowser, os.path
    
    path = os.path.splitext(__file__)[0]
    helpspec = "file://" + path + os.path.sep + \
         "markdown.html"
    
    # webbrowser.open seems not to work well
    browser = webbrowser.get()
    if not browser.open_new(helpspec):
        print(("Help file not found:" + helpspec))
try:    #override
    from extension import helper
except:
    pass        

def attributesFromDict(d):
    """build self attributes from a dictionary d."""

    # based on Python Cookbook, 2nd edition 6.18

    self = d.pop('self')
    for name, value in d.items():
        setattr(self, name, value)