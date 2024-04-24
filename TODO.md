
1. LI: Attribute grouping (needs to filter through the whole pipeline)
   1. attribute group <-> multiple projects. Want one model / attribute group
2. Measure 
   1. Custom metric
   2. Model selection
   3. PS: Decay testing (multiple houldouts scoring)
   4. PS: Stability testing (back test retraining)
3. Data ingest and prep flexibility.
   1. one attribute group <-> one data set
   2. data source <-> multiple attribute groups
   3. (user defined data source function, data prep function registry) -> into config csv
4. Deploy -> output = function that sits outside DR and pipeline and scores. Includes data ingest from above. Scheduling out of scope
5. 
6. 
7. 
8. 
9.  Visualise experiments (Guang kedro viz experiment tracking? MLFlow?)
10. 