using System;
using System.Collections.Generic;
using System.Text;

namespace SharedSerialization
{
    [SerializableAttribute]
    public class MinerInfo
    {
        public string MinerName { get; set; }
        public string MinerUniqueName => MinerName;
        public List<WorkerInfo> Workers { get; set; }
    }

    [SerializableAttribute]
    public class WorkerInfo
    {
        public string GPUStatus { get; set; }
        public float GraphsPerSecond { get; set; }
        public uint TotalSols { get; set; }
        public double Fidelity { get; set; }
        //public LogMessage LastLog { get; set; }
        public int ID { get; set; }
        public DateTime Time { get; set; }
        public GPUOption GPUOption { get; set; }
        public DateTime lastSolution { get; set; }
        public int Errors { get; set; }
        public string GPUName { get; set; }
        public LogMessage LastLog = new LogMessage() { message = "-", time = DateTime.MinValue };
        public LogMessage LastDebugLog;
        public LogMessage LastErrLog = null;
    }

    [SerializableAttribute]
    public enum GPUStatus : int
    {
        STARTING,
        DISABLED,
        ONLINE,
        OFFLINE,
        ERROR,
        OOM
    }


}
