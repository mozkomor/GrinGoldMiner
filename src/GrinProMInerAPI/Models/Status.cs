using SharedSerialization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace GrinProMiner.Models
{
    public class Status
    {
        public StratumConnectionInfo ActiveConnection { get; set; }
        public string LastJob { get; set; } //seconds
        public string LastShare { get; set; } //seconds
        public ShareStats Shares { get; set; }
        public List<WorkerInfo> Workers { get; set; }
    }

    public class SimpleStatus
    {
        public string ConnectionAddress { get; set; }
        public string ConnectionStatus { get; set; }
        public string LastJob { get; set; } //seconds
        public string LastShare { get; set; } //seconds
        public ShareStats Shares { get; set; }
        public List<SimpleWorkerInfo> Workers { get; set; }
    }

    public class SimpleWorkerInfo
    {
        public int ID { get; set; }
        public string Platform { get; set; }
        public string GPUName { get; set; }
        public string Status { get; set; } 
        public float GraphsPerSecond { get; set; }
        public uint TotalSols { get; set; }
        public float Fidelity { get; set; }
        public string LastSolution { get; set; }
    }

    public class StratumConnectionInfo
    {
        public string Address { get; set; }
        public string Port { get; set; }
        public string Status { get; set; }
        public string Login { get; set; }
        public string Password { get; set; }
        public string LastCommunication { get; set; }
        public string LastJob { get; set; }
    }

    public class ShareStats
    {
        public uint Found { get; set; }
        public uint Submitted { get; set; }
        public uint Accepted { get; set; }
        public uint TooLate { get; set; }
        public uint FailedToValidate { get; set; }
    }

    public class ShareInfo
    {
       public string Reason { get; set; }
    }

}
