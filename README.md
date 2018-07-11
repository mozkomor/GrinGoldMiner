# GPU Miner for Cuckoo Cycle PoW - nvidia beta version

This miner (GGM) is currently (testnet3) in early beta stage, OpenCL version is expected to arrive soon with multiple improvements: beta amd support, speed boost, more reliable inter-process socket connection, easier linux install, multi-gpu, new UI, binary package (windows, linux).

Beta miner requires 8GB GPU for cuckoo30. 

GGM is double bounty-winning low-power algorithm variant, primarily intended for grin mining on testnet3 for those who are running on windows. Remember that testnet coins have no value so lower your power-limit to 50% when mining for long periods of time. Performance table have been removed from this page, because cuckoo cycle needs to slightly change in order to be competitive on gpus. Once the final algorithm is known, new benchmarks will be published. We advice not to build grin mining rigs atm. For now, 1080 Ti is twice as fast as 1070 (4 gps vs 2.3 gps). 

---------------------------------------------------

If you want to support or speedup development of this cross platform miner, contact us at mozkomorkaban@gmail.com.

Based on great work of John Tromp, the inventor of Cuckoo cycle PoW  https://github.com/tromp/cuckoo

Designed for https://github.com/mimblewimble/grin

License for the software was changed after publication to be compatible with the bounty. Closed sourced miners based on this work will now be required to pay full 50% of the fee to coin developers and keep the other 50%.

Special thanks to John Tromp, Kristy-Leigh Minehan (OhGodACompany) and Manli Technology for sending us new shiny HW (1080 Ti) for development and tweaking, much appreciated!
