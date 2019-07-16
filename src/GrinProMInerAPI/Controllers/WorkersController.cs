using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Mozkomor.GrinGoldMiner;
using SharedSerialization;

namespace GrinProMiner.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class WorkersController : ControllerBase
    {
        [HttpGet]
        public ActionResult<IEnumerable<WorkerInfo>> Get()
        {
            List<WorkerInfo> workers = new List<WorkerInfo>();
            foreach (var worker in WorkerManager.GetWorkersInfo())
            {
                //WorkerInfo wi = new WorkerInfo();
                //wi.GPUOption = worker.gpu;
                //wi.GPUStatus = worker.GetStatus().ToString();
                //wi.GraphsPerSecond = worker.currentGPS;
                //wi.ID = worker.ID;
                //wi.LastLog = null;
                //wi.Time = DateTime.Now;
                //wi.TotalSols = worker.totalSols;
                //wi.lastSolution = worker.lastSolTime;
                //wi.Errors = worker.errors;
                workers.Add(worker);
            }
            return workers;
        }

        [HttpGet("{id}")]
        public ActionResult<WorkerInfo> Get(int id)
        {
            var worker = WorkerManager.GetWorkersInfo().Where(x => x.ID == id).FirstOrDefault();

            if (worker == null)
                return StatusCode(404, $"No worker with id {id} found");

            //WorkerInfo wi = new WorkerInfo();
            //wi.GPUOption = worker.gpu;
            //wi.GPUStatus =worker.GetStatus().ToString();
            //wi.GraphsPerSecond = worker.currentGPS;
            //wi.ID = worker.ID;
            //wi.LastLog = null;
            //wi.Time = DateTime.Now;
            //wi.TotalSols = worker.totalSols;
            //wi.lastSolution = worker.lastSolTime;
            //wi.Errors = worker.errors;

            return worker;
        }
    }
}