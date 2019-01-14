using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Mozkomor.GrinGoldMiner
{
    public class WorkerManager
    {
        public static List<Worker> workers = new List<Worker>();
        public static DateTime lastJob = DateTime.Now;

        public static void Init(Config config)
        {
            if (config == null) return;

            int i = 0;
            foreach(var gpu in config.GPUOptions)
            {
                Worker w = new Worker(gpu, i++);
                workers.Add(w);
                w.Start(config);
                Task.Delay(4000).Wait();
            }
        }

        //worker found solution
        public static void SubmitSolution(SharedSerialization.Solution sol)
        {
            //todo wrap Solution into richer class with internal info
            // diff check !!
            ConnectionManager.SubmitSol(sol);
        }

        private static string lockPush = "";
        //new job received from stratum connection
        public static void newJobReceived(SharedSerialization.Job job)
        {
            lock (lockPush)
            {
                lastJob = DateTime.Now;
                //update workers..
                foreach (var worker in workers)
                {
                    worker.SendJob(job);
                }
            }
        }

        //tell workers to chill
        internal static void PauseAllWorkers()
        {
            foreach (var worker in workers)
            {
                worker.SendJob(new SharedSerialization.Job()
                    {
                    pre_pow = "",
                    jobID = 666,
                    timestamp = DateTime.Now
                });
            }
        }
    }
}
