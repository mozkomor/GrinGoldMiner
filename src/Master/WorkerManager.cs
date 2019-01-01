using System;
using System.Collections.Generic;
using System.Text;

namespace Mozkomor.GrinGoldMiner
{
    class WorkerManager
    {
        static List<Worker> workers = new List<Worker>();

        public static void Init()
        {
            Worker w = new Worker(SharedSerialization.WorkerType.NVIDIA, 0);
            workers.Add(w);

            var cards = w.GetDevices();
            ;
        }
    }
}
