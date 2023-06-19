//
//  InitialParallelSolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 11/5/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef ParallelSolver_h
#define ParallelSolver_h

#include <stdio.h>

#include <stdio.h>
#include "CCheckers.h"
#include "NBitArray.h"
#include "SolveStats.h"
#include "CCRankings.h"
#include "SharedQueue.h"
#include <thread>

/*
 * This solver has a very basic parallelism. It just divides each iteration into small chunks,
 * and the threads process the chunks independentaly.
 */
namespace ParallelOne {
	
	enum tResult {
		kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
	};
	
	enum {
		kMaxPlayer = 0, kMinPlayer = 1
	};
	
	enum tMeta {
		kHasNoProvenChildren = 0,
		kHasProvenChildren = 1
	};
	
	struct GroupStructure {
		int32_t memoryOffset;
		int32_t symmetricRank;
	};
	
	class Solver {
	private:
		struct result {
			int32_t r1, r2;
		};
		struct range {
			int64_t p1Group;
			int64_t from, to;
		};

	public:
		Solver(const char *path, bool forceBuild = false);
		tResult Lookup(const CCState &s) const;
		void BuildData(const char *path, int numThreads);
		void PrintStats() const;
	private:
		void InitMetaData();
		void Initial(int numThreads);
		
		//		int64_t GetBestNextGroup();
		void SinglePass(int numThreads);

		void ThreadMain();
		void WriteResult(std::vector<result> &data, tResult valueToWrite);
		void SinglePassInnerLoop(CCheckers &cc, int64_t p1Rank, int64_t p2Rank, CCState &s, int64_t finalp1Rank, bool doubleFlip,
								 std::vector<result> &wins, std::vector<result> &losses);
		tResult GetValue(CCheckers &cc, const CCState &s, int64_t finalp1Rank, bool doubleFlip);
		void PropagateWinToParent(CCheckers &cc, CCState &s, std::vector<result> &wins);
		void MarkParents(CCheckers &cc, CCState &s);
		
		void DoBFS();
		void GetSearchOrder();

		const char *GetFileName(const char *path);
		tResult Translate(const CCState &s, tResult res) const;
		tResult Translate(int nextPlayer, tResult res) const;
		NBitArray<2> data;
		uint64_t proven;
		int64_t symmetricStates;
		stats stat;
		CCPSRank12 r;
		GroupStructure *groups;
		std::vector<int8_t> bfs;
		std::vector<int64_t> order;
		std::vector<bool> open;
		
		const int WORK_SIZE = 500;
		const int CACHE_WRITE_SIZE = 1000;
		const int WORK_QUEUE_BUFFER = 200;
		uint32_t memoryMult;
		SharedQueue<range> workQueue;
		SharedQueue<uint8_t> doneQueue;
		SharedQueue<uint8_t> startQueue;
		std::mutex lock;
		std::vector<std::thread> threads;
};
	
}


#endif /* CacheFriendlySolver_h */
