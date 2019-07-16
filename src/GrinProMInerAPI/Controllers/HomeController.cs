using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using GrinProMiner.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Mozkomor.GrinGoldMiner;

namespace GrinProMiner.Controllers
{
    [Route("")]
    [Controller]
    public class HomeController : Controller
    {
        [Route("api")]
        public string Get()
        {
            return "You can use GET METHODS: /api/status, /api/workers, /api/workers/{id}, /api/connections, /api/connections/active /api/config and POST METHODS: /api/connections/active, /api/config. API DOCS: https://grinpro.io/api.html";
        }

        [Route("")]
        public IActionResult GetHome()
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


            return View("Views/Home.cshtml",status); 
        }
    }
}