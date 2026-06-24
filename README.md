## B0 optimization with IDDS/PANDA using epic_scheduler

Step by step running.

### EIC container

./eic/eic-shell 

### Virtual environment 

```
cd B0_FarForward
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip black ruff
pip install idds-client idds-common idds-workflow panda-client
```

### Scheduler

```
cd scheduler_epic
pip install -e .
pip install -e .[slurm,panda]
pip install -e .[dev]
```

### Run

```
cd B0_FarForward
source setup.sh
source setup_panda_bnl.sh
python3 b0_panda_run.py 
```

### Log files and PanDA status checking

With the associated reqid (e.g. 3547) follow this link:

https://pandamon01.sdcc.bnl.gov/tasks/?reqid=3065 


Get the jeditaskid and follow this link (e.g 37217):

https://pandamon01.sdcc.bnl.gov/jobs/?jeditaskid=37217&mode=nodrop&display_limit=100


Get the pandaid for a given job (e.g. 888201) and find the log files here:

https://pandamon01.sdcc.bnl.gov/job?pandaid=888201