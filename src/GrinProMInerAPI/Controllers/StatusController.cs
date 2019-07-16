using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Mozkomor.GrinGoldMiner;
using GrinProMiner.Models;
using SharedSerialization;

namespace GrinProMiner.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class StatusController : ControllerBase
    {
        [Route("all")]
        public ActionResult<Status> GetAll()
        {
            var lastJob = WorkerManager.lastJob.ToString();

            var conn = ConnectionManager.GetCurrConn();

            if (conn == null)
                return StatusCode(404, "No Straturm Connection active.");


            var lastSHare = conn.lastShare;
            var totalShares = conn.totalShares;
            var accepted = conn.sharesAccepted;
            var rejected = conn.sharesRejected;
            var tooLate = conn.sharesTooLate;
            ///TODO do WorkerManagera dat SolutionsFound (kolik dohromady ze vsech karet) a 
            /// SolutionsSubmitted (to bude az co projde pres diff)

            Status status = new Status();

            status.LastShare = lastSHare.ToString();
            status.LastJob = lastJob;

           

            List<WorkerInfo> workers = new List<WorkerInfo>();
            foreach (var worker in WorkerManager.GetWorkersInfo())
            {
                workers.Add(worker);
            }
            status.Workers = workers;

            var sc1 = conn;
            bool isFee = ConnectionManager.IsInFee();
            var c1 = new StratumConnectionInfo()
            {
                Address = isFee ? "FEE" : sc1.ip,
                Port = isFee ? "FEE" : sc1.port.ToString(),
                Login = isFee ? "FEE" : sc1.login,
                Password = isFee ? "FEE" : sc1.password,
                Status = sc1.IsConnected == true ? "Connected" : "Disconnectd",
                LastCommunication = sc1.lastComm.ToString(),
                LastJob = sc1.CurrentJob?.timestamp.ToString(),
            };
            status.ActiveConnection = c1;


            ShareStats ss = new ShareStats();
            ss.Accepted = accepted;
            ss.FailedToValidate = rejected;
            ss.Found = (uint)workers.Sum(w => w.TotalSols);
            ss.Submitted = totalShares;
            ss.TooLate = tooLate;

            status.Shares = ss;

            return status;
        }

        [Route("")]
        public ActionResult<SimpleStatus> Get()
        {
            var lastJob = WorkerManager.lastJob.ToString("yyyy-MM-ddTHH:mm:ssK");

            var conn = ConnectionManager.GetCurrConn();

            if (conn == null)
                return StatusCode(404, "No Straturm Connection active.");

            
            var lastSHare = conn.lastShare;
            var totalShares = conn.totalShares;
            var accepted = conn.sharesAccepted;
            var rejected = conn.sharesRejected;
            var tooLate = conn.sharesTooLate;
            ///TODO do WorkerManagera dat SolutionsFound (kolik dohromady ze vsech karet) a 
            /// SolutionsSubmitted (to bude az co projde pres diff)

            SimpleStatus status = new SimpleStatus();

            status.LastShare = lastSHare.ToString("yyyy-MM-ddTHH:mm:ssK");
            status.LastJob = lastJob;
            
            List<SimpleWorkerInfo> workers = new List<SimpleWorkerInfo>();
            foreach (var worker in WorkerManager.GetWorkersInfo())
            {
                SimpleWorkerInfo wi = new SimpleWorkerInfo();

                wi.GPUName = worker.GPUName;
                wi.Platform = worker.GPUOption.PlatformID.ToString();
                wi.Status = worker.GPUStatus;
                wi.GraphsPerSecond = worker.GraphsPerSecond;
                wi.ID = worker.ID;
                wi.TotalSols = worker.TotalSols;
                wi.LastSolution = worker.lastSolution.ToString("yyyy-MM-ddTHH:mm:ssK");
                wi.Fidelity = (float)worker.Fidelity;
                workers.Add(wi);
            }
            status.Workers = workers;
            
            if (ConnectionManager.IsInFee())
            {
                status.ConnectionAddress = $"FEE (GrinPro collects 1% as fee for the Grin Development Fund and 1% for further miner development.)";
                status.ConnectionStatus = conn.IsConnected == true ? "Connected" : "Disconnectd";
            }
            else
            {
                status.ConnectionAddress = $"{conn.ip}:{conn.port}";
                status.ConnectionStatus = conn.IsConnected == true ? "Connected" : "Disconnectd";
            }

            ShareStats ss = new ShareStats();
            ss.Accepted = accepted;
            ss.FailedToValidate = rejected;
            ss.Found = (uint)workers.Sum(w => w.TotalSols);
            ss.Submitted = totalShares;
            ss.TooLate = tooLate;

            status.Shares = ss;

            return status;
        }
    }
}