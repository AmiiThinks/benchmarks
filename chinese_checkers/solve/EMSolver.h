//
//  EMSolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 1/5/18.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef EMSolver_h
#define EMSolver_h

#include <stdio.h>
#include <stdio.h>
#include <string>
#include "CCheckers.h"
#include "NBitArray.h"
#include "SolveStats.h"
#include "CCRankings.h"
#include "TwoBitFileVector.h"

namespace EM {
	
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
		uint8_t symmetryRedundant;
		uint8_t changed;
		uint8_t handled;
		int32_t symmetricRank;
		int32_t assignedCount;
		int32_t memoryOffset; // this is the offset of the start of the group in p2rank increments (need to multiply)
	};
	
	 //128; // 512 for full data?
	// This is so that the disk buffers for data we can't write out immediately
	// aligns with the buckets in memory.
	const int kGroupsPerFile = 1<<8; // 256

	struct DiskEntry {
		unsigned int baseMemOffset : 8;
		unsigned int offset : 24;
	};

	// 206.01s with 256
	// 198s with 512
	// 189s with 1024
	// 204.10s with 2048
	const int queueSize = 512;
	struct DiskData {
		// TODO: Checksum data to/from disk
		FILE *f;
		int32_t index;
		int32_t onDisk;
		DiskEntry queue[queueSize];
	};
	
	class Solver {
	public:
		Solver(const char *dataPath, const char *scratchPath, bool clearOldData = false);
		tResult Lookup(const CCState &s) const;
		const char *LookupText(const CCState &s) const;
		bool NeedsBuild();
		void BuildData();
		void PrintStats() const;
		void DebugState(CCState &s);
		void DebugState(int64_t r1, int64_t r2);
	private:
		void InitMetaData();
		void Initial();
		
		void DoLoops(CCheckers &cc, int64_t max2, int64_t p1Rank, CCState &s);

		void SinglePass();
		void SinglePassInnerLoop(CCheckers &cc, CCState s, int64_t r1, int64_t finalp1Rank, bool doubleFlip);
		tResult GetValue(CCheckers &cc, const CCState &s, int64_t finalp1Rank, bool doubleFlip);
		void PropagateWinToParent(CCheckers &cc, CCState &s);
		void MarkParents(CCheckers &cc, CCState &s);
		void DoBFS();
		void GetSearchOrder();

		void WriteToBuffer(uint32_t group, uint32_t offset, tResult value);
		void FlushBuffers();
		void FlushBuffer(uint32_t which);
		void PrintBufferStats() const;
		void FlushBiggestBuffer();

		const char *GetFileName(const char *path);
		const char *GetTempFileName(int which);
		tResult Translate(const CCState &s, tResult res) const;
		tResult Translate(int nextPlayer, tResult res) const;
		// Groups per file - needs to correspond with the temporary buffer size
		TwoBitFileVector<ConstChoose(NUM_SPOTS-NUM_PIECES, NUM_PIECES), ConstGetSymmetricP1Ranks(DIAGONAL_LEN, NUM_PIECES), 4096/*2048*/, 6, kGroupsPerFile> data;

		std::vector<DiskData> diskBuffer;
		uint64_t proven;
		int64_t symmetricStates;
		int64_t statesOnDisk, maxStatesOnDisk;
		stats stat;
		CCPSRank12 r;
		std::string scratchPath;
		std::string dataPath;
		std::vector<GroupStructure> groups;
		std::vector<int8_t> bfs;
		std::vector<int32_t> order;
	};
	
}

#endif
