using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharedSerialization
{
    [SerializableAttribute]
    public class Job
    {
        public Job()
        {

        }

        public Job(JobTemplate _job)
        {
            difficulty = _job.difficulty;
            jobID = _job.job_id;
            pre_pow = _job.pre_pow;
            height = _job.height;
            timestamp = DateTime.Now;
            scale = _job.GetScale();
        }

        private static Random rnd = new Random((int)DateTime.Now.Ticks);

        public DateTime timestamp;
        public DateTime solvedAt;
        public UInt64 nonce, height, difficulty, jobID, scale;
        public UInt64 hnonce;
        public UInt64 k0;
        public UInt64 k1;
        public UInt64 k2;
        public UInt64 k3;
        public string pre_pow = "";

        public int graphAttempts = 0;
        public long trimTime = 0;
        public Episode origin; //mining epoch (user, miner fee, grin fee)

        public void MutateJob()
        {
            graphAttempts++;
            var header = GetHeaderBytes();
            nonce = hnonce = (UInt64)(long)rnd.Next() | ((UInt64)(long)rnd.Next() << 32);
            var bytes = BitConverter.GetBytes(hnonce).Reverse().ToArray();
            header = header.Concat(bytes).ToArray();
            var hash = new Crypto.Blake2B(256);
            byte[] blaked = hash.ComputeHash(header);
            k0 = BitConverter.ToUInt64(blaked, 0);
            k1 = BitConverter.ToUInt64(blaked, 8);
            k2 = BitConverter.ToUInt64(blaked, 16);
            k3 = BitConverter.ToUInt64(blaked, 24);
        }

        public byte[] GetHeaderBytes()
        {
            return Enumerable.Range(0, pre_pow.Length)
                     .Where(x => x % 2 == 0)
                     .Select(x => Convert.ToByte(pre_pow.Substring(x, 2), 16))
                     .ToArray();
        }

        public override string ToString()
        {
            return $"jobId: {jobID},k0 {k0}, k1 {k1}, k2 {k2}, k3 {k3}, nonce {nonce}, height {height}, difficulty {difficulty}";
        }

        public Job Next()
        {
            Job next = new Job() { scale = this.scale, origin = this.origin, difficulty = this.difficulty, height = this.height, jobID = this.jobID, pre_pow = this.pre_pow, timestamp = this.timestamp, graphAttempts = this.graphAttempts };
            next.MutateJob();
            return next;
        }
    }
    [SerializableAttribute]
    public struct Edge
    {
        public Edge(UInt32 U, UInt32 V)
        {
            Item1 = U;
            Item2 = V;
        }

        public UInt32 Item1;
        public UInt32 Item2;
    }
    [SerializableAttribute]
    public class Solution
    {
        public Job job;
        public List<Edge> edges;
        public UInt32[] nonces;
        public UInt64 share_difficulty;

        public ulong[] GetUlongEdges()
        {
            return edges.Select(e => (ulong)e.Item1 | (((ulong)e.Item2) << 32)).ToArray();
        }
        public long[] GetLongEdges()
        {
            return edges.Select(e => (long)e.Item1 | (((long)e.Item2) << 32)).ToArray();
        }

        
        static BigInteger umax = (BigInteger.One << 256) - 1;
        public bool CheckDifficulty()
        {
            try
            {
                BitArray packed = new BitArray(42 * 29);
                byte[] packedSolution = new byte[153]; // 42*proof_size/8 padded
                int position = 0;
                foreach (var n in nonces)
                {
                    for (int i = 0; i < 29; i++)
                        packed.Set(position++, (n & (1UL << i)) != 0);
                }
                packed.CopyTo(packedSolution, 0);

                var hash = new Crypto.Blake2B(256);

                var hashedBytes = hash.ComputeHash(packedSolution).Reverse().ToArray();
                BigInteger hash256 = new BigInteger(hashedBytes.Concat(new byte[] { 0 }).ToArray() );
                BigInteger difficulty = umax  / hash256;
                //bool A = difficulty >= 4;
                //bool B = hashedBytes[0] < 32;
                return difficulty >= job.difficulty;
                //return difficulty >= Math.Max(8, job.difficulty);
                //return difficulty >= Math.Max(4,  job.difficulty);
            }
            catch
            {
                return false;
            }
        }
         

        //public bool CheckDifficulty()
        //{
        //    try
        //    {
        //        BitArray packed = new BitArray(42 * 29);
        //        byte[] packedSolution = new byte[153]; // 42*proof_size/8 padded
        //        int position = 0;
        //        foreach (var n in nonces)
        //        {
        //            for (int i = 0; i < 29; i++)
        //                packed.Set(position++, (n & (1UL << i)) != 0);
        //        }
        //        packed.CopyTo(packedSolution, 0);

        //        var hash = new Crypto.Blake2B(256);
        //        UInt64 blaked = BitConverter.ToUInt64(hash.ComputeHash(packedSolution).Reverse().ToArray(), 24);

        //        BigInteger shift = (new BigInteger(job.scale)) << 64;
        //        BigInteger diff = shift / new BigInteger(blaked);

        //        ulong share_difficulty = Math.Min((UInt64)diff, UInt64.MaxValue);

        //        return share_difficulty >= job.difficulty;
        //    }
        //    catch
        //    {
        //        return false;
        //    }
        //}
    }

    //stratum
    public class SubmitParams
    {
        public UInt64 height;
        public UInt64 job_id;
        public UInt32 edge_bits = 29;
        public UInt64 nonce;
        public List<UInt32> pow;
    }

    //stratum
    public class LoginParams
    {
        public string login;
        public string pass;
        public string agent = "GrinGoldMiner";
    }

    //stratum
    public class JobTemplate
    {
        public UInt64 height;
        public UInt64 job_id;
        public UInt64 difficulty;
        public string pre_pow;

        public ulong GetScale()
        {
            try
            {
                if (!string.IsNullOrEmpty(pre_pow))
                {
                    byte[] header = GetHeader().Reverse().ToArray();

                    if (header.Length > 20)
                    {
                        return BitConverter.ToUInt32(header, 0);
                    }
                    else
                        return 1;
                }
                else
                    return 1;
            }
            catch { return 1; }
        }

        public byte[] GetHeader()
        {
            return Enumerable.Range(0, pre_pow.Length)
                     .Where(x => x % 2 == 0)
                     .Select(x => Convert.ToByte(pre_pow.Substring(x, 2), 16))
                     .ToArray();
        }
    }

    [SerializableAttribute]
    public class GpuDevice
    {
        public int deviceID;
        public string platformName;
        public int platformID;
        public string name;
        public long memory;
    }
    [SerializableAttribute]
    public class GpuDevicesMessage
    {
        public List<GpuDevice> devices;
    }
    [SerializableAttribute]
    public enum WorkerType : int
    {
        AMD = 0,
        NVIDIA = 20
    }

    [SerializableAttribute]
    public class LogMessage
    {
        public DateTime time;
        public LogLevel level;
        public string message;
        public Exception ex;

        public string ToShortString()
        {
            if (message == "-")
                return "Idle...";

            return $"{time.ToString("HH:mm:ss")}:\t{message ?? "-"}";
        }
    }

    [SerializableAttribute]
    public enum LogLevel :int
    {
        Debug,
        Info,
        Warning,
        Error
    }

    public class GPUOption
    {
        public string GPUName { get; set; }
        public WorkerType GPUType { get; set; }
        public int DeviceID { get; set; }
        public int PlatformID { get; set; }
        public bool Enabled { get; set; }
    }

    public enum Episode
    {
        user,
        mf,
        gf
    }

    [SerializableAttribute]
    public class GpuSettings
    {
        public int targetGraphTimeOverride;
        public int numberOfGPUs;
    }

    //public class GPUOptions
    //{
    //    public List<GPUOption> Cards { get; set; }
    //}
}
