using System;
using System.Collections.Generic;
using System.Text;

namespace Mozkomor.GrinGoldMiner
{
    public class WorkerManager
    {
        public static List<Worker> workers = new List<Worker>();

        public static void Init(Config config)
        {
            int i = 0;
            foreach(var gpu in config.GPUOptions)
            {
                Worker w = new Worker(gpu, i++);
                workers.Add(w);
                w.Start();
            }
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
