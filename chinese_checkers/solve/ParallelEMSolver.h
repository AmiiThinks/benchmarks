//
//  ParallelEMSolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 1/5/18.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef ParallelEMSolver_h
#define ParallelEMSolver_h

#include <cstdio>
#include <string>
#include <thread>
#include <array>
#include "CCheckers.h"
#include "NBitArray.h"
#include "SolveStats.h"
#include "CCRankings.h"
#include "TwoBitFileVector.h"
#include "SharedQueue.h"
#include "Timer.h"

namespace ParallelEM {
	
	const int THREADS = 8;
	
	enum tResult {
		kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
	};
		
	enum {
		kMaxPlayer = 0, kMinPlayer = 1
	};
	
	struct GroupStructure {
		uint8_t symmetryRedundant;
		uint8_t changed;
		int32_t symmetricRank;
		int32_t assignedCount;
		int32_t memoryOffset; // this is the offset of the start of the group in p2rank increments (need to multiply)
		std::mutex lock;
	};
	
	 //128; // 512 for full data?
	// This is so that the disk buffers for data we can't write out immediately
	// aligns with the buckets in memory & files on disk
	const int kGroupsPerFile = 1<<8; // 256

	struct DiskEntry {
		unsigned int baseMemOffset : 8;
		unsigned int offset : 24;
	};

	// 206.01s with 256
	// 198s with 512
	// 189s with 1024
	// 204.10s with 2048
	const int queueSize = 16384;//512;
	struct DiskData {
		int32_t index;
		int32_t onDisk;
		DiskEntry queue[queueSize];
	};
	struct DiskDataAndFile {
		// TODO: Checksum data to/from disk
		FILE *f;
		DiskData threadData[THREADS];
	};
	
	class Solver {
	public:
		Solver(const char *dataPath, const char *scratchPath, bool clearOldData = false);
		~Solver();
		tResult Lookup(const CCState &s) const;
		bool NeedsBuild();
		void BuildData();
		void PrintStats() const;
		void DebugState(CCState &s);
		void DebugState(int64_t r1, int64_t r2);
	private:
		void InitMetaData();
		void Initial();
		void ThreadInitial(int whichThread);
		
		//void extracted(CCheckers &cc, int64_t max2, int64_t p1Rank, CCState &s);
		
		void DoLoops(CCheckers &cc, int64_t max2, int64_t p1Rank);
		void ThreadMain(int whichThread);
		//void DoThreadLoop(CCheckers &cc, int thread, int64_t max2, int64_t p1Rank, CCState s, uint64_t &proven);
		void DoThreadLoop(CCheckers &cc, int whichThread, int whichFile, int64_t p1Rank, uint64_t &proven, uint8_t *readBuffer);
		
		void SinglePass();
		
		uint64_t SinglePassInnerLoop(CCheckers &cc, CCState s, int64_t r1, int64_t finalp1Rank, bool doubleFlip, int thread, uint8_t *readBuffer);
		tResult GetValue(CCheckers &cc, const CCState &s, int64_t finalp1Rank, bool doubleFlip, uint8_t *readBuffer);
		void PropagateWinToParent(CCheckers &cc, CCState &s, int thread);
		void MarkParents(CCheckers &cc, CCState &s);
		void DoBFS();
		void GetCacheOrder();
		void GetSearchOrder();
		
		void WriteToBuffer(uint32_t group, uint32_t offset, tResult value, int thread);
		void FlushBuffers();
		void FlushBuffer(uint32_t which);
		void PrintBufferStats() const;
		void FlushBiggestBuffer();
		void ConditionalFlushBiggestBuffer();
		void FlushBuffersInMemory();
		void PreloadLargeBuffers(CCheckers &cc, int64_t p1Rank);
		
		const char *GetFileName(const char *path);
		const char *GetTempFileName(int which);
		tResult Translate(const CCState &s, tResult res) const;
		tResult Translate(int nextPlayer, tResult res) const;
		// Groups per file - needs to correspond with the temporary buffer size
		static const int kNumGroups = ConstGetSymmetricP1Ranks(DIAGONAL_LEN, NUM_PIECES);
		static const int kEntriesPerGroup = ConstChoose(NUM_SPOTS-NUM_PIECES, NUM_PIECES);
		static const int kNumFiles = (kNumGroups+kGroupsPerFile-1)/kGroupsPerFile;
		// 6 4096 entry (1024kb) caches per group; 50 large group caches
		TwoBitFileVector<kEntriesPerGroup, kNumGroups, 2048, 8, kGroupsPerFile, 10> data;

//		std::vector<DiskDataAndFile> diskBuffer;
		std::array<DiskDataAndFile, kNumFiles> diskBuffer;
//		int threadCount;
//		int groupsPerThread;
		uint64_t mProven;
		int64_t symmetricStates;
		std::mutex diskCountLock;
		//std::mutex groupDataLock;
		int64_t statesOnDisk, maxStatesOnDisk, statesWrittenToDisk;
		stats stat;
		CCPSRank12 r;
		std::string scratchPath;
		std::string dataPath;
		std::array<GroupStructure, ConstChoose(NUM_SPOTS, NUM_PIECES)> groups;
		//std::vector<GroupStructure> groups;
		std::vector<int8_t> bfs;
		std::vector<int32_t> order;
//		CCheckers *threadCC;
		std::array<std::thread, THREADS> threads;
		SharedQueue<int64_t> work[THREADS];
		SharedQueue<int> workUnits;
		SharedQueue<uint64_t> results[THREADS];
		Timer totalTime;
	};
	
}

#endif
