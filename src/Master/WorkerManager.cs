using System;
using System.Collections.Generic;
using System.Text;

namespace Mozkomor.GrinGoldMiner
{
    class WorkerManager
    {
        static List<Worker> workers = new List<Worker>();

        public static void Init(Config config)
        {
            foreach(var gpu in config.GPUOptions)
            {
                Worker w = new Worker(gpu.GPUType, gpu.DeviceID, gpu.PlatformID);
                workers.Add(w);
                w.Start();
            }
            

            //var cards = w.Start();
            ;
        }

        //worker found solution
        public static void SubmitSolution(SharedSerialization.Solution sol)
        {
            //todo wrap Solution into richer class with internal info
            ConnectionManager.SubmitSol(sol);
        }

        //new job received from stratum connection
        public static void newJobReceived(SharedSerialization.Job job)
        {
            //update workers..
            foreach (var worker in workers)
            {
                worker.SendJob(job);
            }
        }
    }
}
