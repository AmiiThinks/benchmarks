//
//  EMSolver.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 11/5/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#include "EMSolver.h"
#include <cassert>
#include <cstring>
#include "Timer.h"

namespace EM {
	static const char *resultText[] = {"Draw", "Loss", "Win", "Illegal"};

#pragma mark - Initialization -

	
	Solver::Solver(const char *dataPath, const char *scratchPath, bool forceBuild)
	:scratchPath(scratchPath), data(dataPath, forceBuild)
	{
		if (NUM_PIECES == 2 || NUM_PIECES == 5)
		{
			printf("%d pieces not symmetric; aborting\n", NUM_PIECES);
			exit(0);
		}
		printf("### Chinese Checkers Solver! Board: %d Pieces: %d\n", NUM_SPOTS, NUM_PIECES);
		printf("--> Starting EM Solver [%s] --\n", __FILE__);
		printf("--> Ranking: %s --\n", r.name());

		InitMetaData();
		printf("--> Done initializing meta-data.\n");
		if (data.NeedsBuild())
			printf("--> Waiting to build solve data\n");
		else
			printf("--> Build data ready for queries.\n");
	}

	void Solver::InitMetaData()
	{
		CCheckers cc;
		CCState s;
		int32_t memoryOffset = 0;

		printf("--> Initializing data and detecting symmetry\n");
		//uint64_t maxRank = r.getMaxRank();
		groups.resize(r.getMaxP1Rank());
		
		// Note that in single-player CC we have to (maybe?) generate
		// the children of both the regular and flipped states
		
		int64_t totalStates = 0;
		statesOnDisk = maxStatesOnDisk = 0;
		symmetricStates = 0;
		for (int32_t x = 0; x < groups.size(); x++)
			groups[x].symmetricRank = -1;
		for (int32_t x = 0; x < groups.size(); x++)
		{
			groups[x].changed = 0;
			groups[x].handled = false;
			groups[x].assignedCount = 0;
			r.unrank(x, 0, s);
			CCState sym;
			cc.FlipPlayer(s, sym, 0);

			// Since we are only going up to 7x7 (6 piece) CC, we can narrow to 32 bits here
			int32_t otherRank = static_cast<int32_t>(r.rankP1(sym));
			if (otherRank < x)
			{
				groups[x].symmetryRedundant = true;
				groups[x].symmetricRank = otherRank;
				groups[otherRank].symmetricRank = x;
			}
			else {
				groups[x].memoryOffset = memoryOffset;
				memoryOffset++;//= r.getMaxP2Rank();
				groups[x].symmetryRedundant = false;
				//printf("[%d] !Symmetric: ", x); s.PrintASCII();
				symmetricStates += r.getMaxP2Rank();
			}
			totalStates += r.getMaxP2Rank();
		}
		
		// 6,482 ? (non symmetric groups?)
//		data.Resize(symmetricStates);
//		data.Clear();
		printf("--> Test: (49 3) = %llu\n", ConstChoose(49, 3));
		printf("--> SYMMETRY: %lld total states; %lld after symmetry reduction\n", totalStates, symmetricStates);
		printf("--> STATIC: %llu fully symmetric groups\n", ConstGetNonMiddleSymmetry(DIAGONAL_LEN, NUM_PIECES, 0));
		printf("--> STATIC SYMMETRY: %llu non-symmetric groups; %lld total states\n",
			   ConstGetSymmetricP1Ranks(DIAGONAL_LEN, NUM_PIECES), (uint64_t)ConstChoose(NUM_SPOTS-NUM_PIECES, NUM_PIECES)*(uint64_t)ConstGetSymmetricP1Ranks(DIAGONAL_LEN, NUM_PIECES));
		printf("--> Running single-agent BFS to order search\n");
		DoBFS();
		GetSearchOrder();

		memoryOffset = 0;
		printf("--> Constructing ordering from BFS\n");
		for (int32_t x = 0; x < order.size(); x++)
		{
			if (!groups[order[x]].symmetryRedundant)
			{
				//printf(" [%d] Group %d is at memory %llu\n", x, order[x], memoryOffset);
				groups[order[x]].memoryOffset = memoryOffset;
				memoryOffset += 1;//r.getMaxP2Rank();
			}
		}
		printf("--> %lu reduced groups [from %lu - %1.3fx]\n", order.size(), groups.size(), groups.size()/(float)order.size());
		printf("--> Disk buffer entry size: %lu [should be 4 if bit fields are implemented as expected]\n", sizeof(DiskEntry));
		diskBuffer.resize(1+(memoryOffset>>8));
		for (int x = 0; x < diskBuffer.size(); x++)
		{
			diskBuffer[x].f = fopen(GetTempFileName(x), "w+b"); // open and truncate file
			assert(diskBuffer[x].f != 0);
			//setvbuf(diskBuffer[x].f, NULL, _IONBF, 0);
			diskBuffer[x].index = 0;
			diskBuffer[x].onDisk = 0;
		}
	}

	void Solver::GetSearchOrder()
	{
		int newItems;
		int d = 0;
		order.clear();
		do {
			newItems = 0;
			//printf("Search order for %d\n", d);
			for (int32_t x = bfs.size()-1; x >= 0; x--)
			{
				if (bfs[x] == d) // No redundant states in the order now
				{
					if (!groups[x].symmetryRedundant)
					{
						order.push_back(x);
						newItems++;
						//if (d == 0) printf("%d not redundant\n", x);
					} else {
						//if (d == 0) printf("%d redundant\n", x);
					}
				}
			}
			d++;
		} while (newItems > 0);
	}
	
	void Solver::DoBFS()
	{
		CCheckers cc;
		CCState s;
		Timer t;
		bfs.resize(r.getMaxP1Rank());
		std::fill(bfs.begin(), bfs.end(), -1);
		
//		cc.Reset(s);
//		printf("Reset: "); s.PrintASCII();
//		cc. SymmetryFlipHoriz(s);
		//s.Reverse();
		// Set the start state to 0
		
//		bfs[r.rankP1(s)] = 0;
		cc.ResetP1Goal(s);
		bfs[r.rankP1(s)] = 0;
		//r.unrankP1(r.getMaxP1Rank()-1, s);
		//printf("Start: "); s.PrintASCII();
		
		int depth = 0;
		int written = 0;
		int total = 1;
		printf("--> BFS: ");
		t.StartTimer();
		do {
			written = 0;
			for (int x = 0; x < r.getMaxP1Rank(); x++)
			{
				if (bfs[x] == depth)
				{
					r.unrankP1(x, s);
					CCMove *m = cc.getMoves(s);
					for (CCMove *tmp = m; tmp; tmp = tmp->next)
					{
						cc.ApplyMove(s, tmp);
						s.toMove = kMaxPlayer;
						int64_t rank = r.rankP1(s);
						if (bfs[rank] == -1)
						{
							bfs[rank] = depth+1;
							written++;
						}
						s.toMove = kMinPlayer;
						cc.UndoMove(s, tmp);
					}
					cc.freeMove(m);
				}
			}
			total += written;
			printf("%d ", depth);
			//printf("Depth %d complete. %d new. %d of %d complete\n", depth, written, total, r.getMaxP1Rank());
			depth++;
		} while (written != 0);
		t.EndTimer();
		printf("\n--> BFS complete to depth %d in %1.2fs\n", depth, t.GetElapsedTime());
	}
	
#pragma mark - mark initial wins/losses -

	void Solver::Initial()
	{
		CCheckers cc;
		CCState s, tmp;
//		uint64_t maxRank = r.getMaxRank();
//		int64_t max1 = r.getMaxP1Rank();
		int64_t max2 = r.getMaxP2Rank();
		
		for (int64_t x = 0; x < order.size(); x++)
		{
			int64_t p1Rank = order[x];
			
			if (groups[p1Rank].symmetryRedundant)
				continue;
			
			FlushBuffer(groups[p1Rank].memoryOffset>>8);

			r.unrankP1(p1Rank, s);

			int startCount = cc.GetNumPiecesInStart(s, 0);
			int goalCount = cc.GetNumPiecesInGoal(s, 0);
			// 1. if no pieces in home, then only one possible goal.
			if (startCount == 0 && goalCount == 0)
			{
				bool result = cc.SetP2Goal(s);
				assert(result == true);
				//r.unrankP2(0, s);
				stat.unrank++;
				
				if (cc.Winner(s) == -1)
					continue;
				//if (data.SetIf(groups[p1Rank].memoryOffset*r.getMaxP2Rank(), kLoss, kDraw))
				if (data.SetIf(groups[p1Rank].memoryOffset, 0, kLoss, kDraw))
				{
					groups[p1Rank].assignedCount++;
					groups[p1Rank].changed = true;
					proven++;
					PropagateWinToParent(cc, s); // This is a win at the parent!
				}
				continue;
			}
			
			// 2. if pieces in home, then try all goals (could do better)
			for (int64_t p2Rank = 0; p2Rank < max2; p2Rank++)
			{
				if (p2Rank > 0)
				{
					// clear p2 pieces
					for (int x = 0; x < NUM_PIECES; x++)
					{
						s.board[s.pieces[1][x]] = 0;
					}
				}
				r.unrankP2(p2Rank, s);
				stat.unrank++;

				if (!cc.Legal(s))
				{
					auto v = data.Get(groups[p1Rank].memoryOffset, p2Rank);
					if (v == kDraw)
					{
						proven++;
						groups[p1Rank].assignedCount++;
						groups[p1Rank].changed = true;
					}
					//printf("ILL: "); s.PrintASCII();
					data.Set(groups[p1Rank].memoryOffset, p2Rank, kIllegal);
					//data.Set(groups[p1Rank].memoryOffset*r.getMaxP2Rank()+p2Rank, kIllegal);
					continue;
				}
				
				switch (cc.Winner(s))
				{
					case -1: // no winner
						break;
					case 0: // not possible, because it's always player 0's turn
						// Actually, this is possible in one situation - but we consider
						// it illegal to make a suicide move to lose the game
						assert(!"(0) This isn't possible");
						break;
					case 1:
						//if (data.SetIf(groups[p1Rank].memoryOffset*r.getMaxP2Rank()+p2Rank, kLoss, kDraw))
						if (data.SetIf(groups[p1Rank].memoryOffset, p2Rank, kLoss, kDraw))
						{
							proven++;
							groups[p1Rank].assignedCount++;
							groups[p1Rank].changed = true;
							tmp = s;
							PropagateWinToParent(cc, tmp); // This is a win at the parent!
						}
						else if (data.SetIf(groups[p1Rank].memoryOffset, p2Rank, kLoss, kWin))
						{
							//proven++;
							//groups[p1Rank].assignedCount++;
							groups[p1Rank].changed = true;
							tmp = s;
							PropagateWinToParent(cc, tmp); // This is a win at the parent!
						}
						break;
				}
			}
		}

	}
	
#pragma mark - main solver -

	/*
	 * Note that s will be modified in this function
	 */
	void Solver::PropagateWinToParent(CCheckers &cc, CCState &s)
	{
//		return;
		
		CCState tmp;
		// Always called from Loss at max node
		assert(s.toMove == kMaxPlayer);
		stat.backwardExpansions++;

		// Flip the whole board then generate all moves. More efficient than
		// Generating moves and then flipping each time.
		cc.SymmetryFlipVert(s);
		assert(s.toMove == kMinPlayer);
		CCMove *m = cc.getReverseMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			int64_t p1Rank, p2Rank;

			cc.ApplyReverseMove(s, t);

			p1Rank = r.rankP1(s);

			stat.rank++;
			if (!groups[p1Rank].symmetryRedundant)
			{
				p2Rank = r.rankP2(s);
				WriteToBuffer(p1Rank, p2Rank, kWin);
			}

			// Only flip pieces
			cc.SymmetryFlipHoriz_PO(s, tmp);

			p1Rank = r.rankP1(tmp);

			stat.rank++;
			if (!groups[p1Rank].symmetryRedundant)
			{
				p2Rank = r.rankP2(tmp);
				WriteToBuffer(p1Rank, p2Rank, kWin);
			}
			cc.UndoReverseMove(s, t);
		}
		cc.freeMove(m);
	}

	
	/*
	 * Returns the value for the parent, so it has to be flipped.
	 */
	tResult Solver::GetValue(CCheckers &cc, const CCState &s, int64_t finalp1Rank, bool doubleFlip)
	{
		CCState tmp;
		// Max player to move
		if (doubleFlip)
			cc.SymmetryFlipHorizVert(s, tmp); // Same flip used initially to flip players - returns to initial p1rank
		else
			cc.SymmetryFlipVert(s, tmp); // Same flip used initially to flip players - returns to initial p1rank
		assert(tmp.toMove == kMinPlayer);

		CCMove *m = cc.getMoves(tmp);
		stat.forwardExpansions++;
		tResult result = kLoss;
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(tmp, t); // Now max player to move
			int64_t p1Rank, p2Rank;
			tResult val;
			// We re-use the initial p1Rank and just compute p2 here
			assert(tmp.toMove == kMaxPlayer);
			p1Rank = finalp1Rank;
			p2Rank = r.rankP2(tmp);
			tResult tmpVal = (tResult)data.GetReadOnlyBuffer(groups[p1Rank].memoryOffset, p2Rank);
			val = Translate(kMaxPlayer, tmpVal); // Invert back to get value for min player
			stat.rank++;
			cc.UndoMove(tmp, t);

	
			switch (val)
			{
				case kWin:
					if (tmp.toMove == kMaxPlayer)
					{
						assert(!"(a) Shouldn't get here - it's min's turn\n");
						cc.freeMove(m);
						return kIllegal;
					}
					break;
				case kLoss: // loss at child will be win for parent
					if (tmp.toMove == kMinPlayer)
					{
						cc.freeMove(m);
//						assert(!"(b) Shouldn't get here - a loss to min should have been propagated immediately(?).\n");
						// Previously we didn't get here because wins are propagated immediately to parents
						// But, when we send such states to disk, they are missed in memory and may be found later,
						// so we need to handle them here.
						return kWin;
					}
					break;
				case kDraw: // still unknown // TODO: By immediately propagating wins we can skip out once we find a draw.
					cc.freeMove(m);
					return kDraw;
//					result = kDraw;
					break;
				case kIllegal: // ignore
					break;
			}

		}
		cc.freeMove(m);
		// Parent (max node) is a loss only if all children are losses (wins for player to move)
		return result;
	}
	
	void Solver::SinglePassInnerLoop(CCheckers &cc, CCState s, int64_t r1, int64_t finalp1Rank, bool doubleFlip)
	{
		assert(s.toMove == kMaxPlayer);
		if (groups[r1].symmetryRedundant)
		{
			// We can look up the children, but we can't store the parent value. Skip.
			return;
		}
		int64_t r2 = r.rankP2(s);

		// Check whether we need to prove this state
		if (data.Get(groups[r1].memoryOffset, r2) != kDraw)
			return;

		tResult result = GetValue(cc, s, finalp1Rank, doubleFlip);
		
		switch (result)
		{
			case kWin:
				// Normally we don't see wins at max nodes, since they are immediately propagated. But, with delayed
				// processing of parents we might miss some.
				//if (data.SetIf(groups[r1].memoryOffset*r.getMaxP2Rank()+r2, kWin, kDraw))
				if (data.SetIf(groups[r1].memoryOffset, r2, kWin, kDraw))
				{
					groups[r1].assignedCount++;
					groups[r1].changed = true;
					proven++;
				}
				break;
			case kLoss:
				// Loss symmetrically becomes win at parent
				//if (data.SetIf(groups[r1].memoryOffset*r.getMaxP2Rank()+r2, kLoss, kDraw))
				if (data.SetIf(groups[r1].memoryOffset, r2, kLoss, kDraw))
				{
					PropagateWinToParent(cc, s);
					groups[r1].assignedCount++;
					groups[r1].changed = true;
					proven++;
				}
				break;
			case kDraw: // still unknown
				break;
			case kIllegal:
				assert(!"(d) Shouldn't get here");
				break;
		}
	}

	void Solver::DoLoops(CCheckers &cc, int64_t max2, int64_t p1Rank, CCState &s)
	{
		// 1. Load (if needed) the chunk [p1Rank] we are currently reading from
		// 2. Pull any cached states from disk into memory

		// temp disk usage is too big
		// if 75% full, flush back to 50% full
		if ((statesOnDisk*4) > (symmetricStates/4)*0.75)
		{
			auto i = statesOnDisk;
			while ((statesOnDisk*4) > (symmetricStates/4)*0.5)
				FlushBiggestBuffer();
			printf("[f%llu]\n", i-statesOnDisk);
		}
		FlushBuffer(groups[p1Rank].memoryOffset>>8);
		
		// Read is only valid until we flush buffers
		data.LoadReadOnlyBuffer(groups[p1Rank].memoryOffset);
		r.unrank(p1Rank, max2-1, s);
		cc.SymmetryFlipVert(s);
		s.toMove = kMaxPlayer;

		int64_t r1;//, r2;
		r.GetFirstP1RelP2(s, r1);
		for (int64_t p2Rank = 0; p2Rank < max2; p2Rank++)
		{
			// Pass #1. These are all the parents that lead directly to the p1Rank as second-player position
			SinglePassInnerLoop(cc, s, r1, p1Rank, false);
			r.IncrementP1RelP2(s, r1);
		}
		if (groups[p1Rank].symmetricRank != -1) // There is a symmetric group that leads here
		{
			r.unrank(groups[p1Rank].symmetricRank, max2-1, s);
			cc.SymmetryFlipVert(s);
			s.toMove = kMaxPlayer;

			r.GetFirstP1RelP2(s, r1);
			
			for (int64_t p2Rank = 0; p2Rank < max2; p2Rank++)
			{
				// Pass #2. These are all the parents that lead directly to the reversed p1Rank as second-player position
				// These are new states that we have to consider (compared to older solving approaches) that have all
				// their successors in this group
				SinglePassInnerLoop(cc, s, r1, p1Rank, true);
				r.IncrementP1RelP2(s, r1);
			}
		}
	}

	void Solver::SinglePass()
	{
		CCheckers cc;
		CCState s;
		int64_t max2 = r.getMaxP2Rank();
		float perc = 0.05f;
		
		for (int64_t x = 0; x < order.size(); x++)
		{
			int64_t p1Rank = order[x];
			if ((float)x/order.size() > perc)
			{
				printf("%1.1f%% ", x*100.0/order.size());
				perc += 0.05;
				PrintBufferStats();
			}
			if (!groups[p1Rank].changed)
			{
				continue;
			}
			if (groups[p1Rank].symmetryRedundant)
			{
				// No states in this group to analyze
				continue;
			}
			groups[p1Rank].changed = false;
			
			DoLoops(cc, max2, p1Rank, s);
		}
		printf("\n");
		//FlushBuffers();
	}
	
	bool Solver::NeedsBuild()
	{
		return data.NeedsBuild();
	}
	
	// Full size:
	// TODO: Validate this and pre-code!
	// --> SYMMETRY: 85251690988464 total; 42645604101646 symmetric
	void Solver::BuildData()
	{
		Timer t, total;
		CCheckers cc;
//		printf("--> Initializing Meta Data\n");
//		InitMetaData();
		proven = 0;
		
		total.StartTimer();
		t.StartTimer();
		printf("** Filling in initial states\n");
		Initial();
		t.EndTimer();
		printf("Round 0; %llu new; %llu of %llu proven; %1.2fs elapsed; %llu on disk (%llu max)\n", proven, proven, symmetricStates, t.GetElapsedTime(), statesOnDisk, maxStatesOnDisk);
		
		uint64_t oldProven;
		
		CCState start;
		cc.Reset(start);
		int64_t startR1, startR2;
		bool startProven = false;
		r.rank(start, startR1, startR2);
		printf("** Starting Main Loop\n");
		int iteration = 0;
		do {
			iteration++;
			t.StartTimer();
			oldProven = proven;
			SinglePass();
			t.EndTimer();
			printf("Round %d; %llu new; %llu of %llu proven; %1.2fs elapsed; %llu on disk (%llu max)\n", iteration, proven-oldProven, proven, symmetricStates, t.GetElapsedTime(), statesOnDisk, maxStatesOnDisk);

			if (!startProven && data.Get(groups[startR1].memoryOffset, startR2) != kDraw)
			{
				printf("Start state proven to be %s\n", resultText[data.Get(groups[startR1].memoryOffset, startR2)]);
				startProven = true;
			}
			if (proven == oldProven)
				FlushBuffers();
		} while (proven != oldProven);
		total.EndTimer();
		printf("%1.2fs total time elapsed\n", total.EndTimer());
		
		// TODO: Restore this
//		data.Write(GetFileName());
		PrintStats();
	}
	
#pragma mark - disk buffer code -

	void Solver::WriteToBuffer(uint32_t group, uint32_t offset, tResult value)
	{
//		assert(value == kWin);
		// offset needs 43!/37!6! entries which requires 23 bits
		// Let's use 24 bits for offset and 8 bits for collapsing groups
		auto &i = diskBuffer[groups[group].memoryOffset>>8];
		if (i.index >= queueSize) {
			fwrite(i.queue, sizeof(DiskEntry), queueSize, i.f);
			i.index = 0;
			i.onDisk++;
			statesOnDisk+=queueSize;
			maxStatesOnDisk = std::max(statesOnDisk, maxStatesOnDisk);
		}
		i.queue[i.index] = { static_cast<uint32_t>(groups[group].memoryOffset&0xFF), offset };
		i.index++;
	}


	void Solver::PrintBufferStats() const
	{
		int max_b = 0, min_b = 0xFFFFFFF;
		for (int x = 0; x < diskBuffer.size(); x++)
		{
			max_b = std::max(max_b, diskBuffer[x].onDisk*queueSize);
			min_b = std::min(min_b, diskBuffer[x].onDisk*queueSize);
		}

		printf(" - Total states on disk: %llu; avg per bucket: %f; max: %d; min: %d\n", statesOnDisk, diskBuffer.size()/(float)statesOnDisk,
			   max_b, min_b);
	}

	void Solver::FlushBiggestBuffer() 
	{
		int max_ = 0;
		for (int x = 0; x < diskBuffer.size(); x++)
		{
			if (diskBuffer[x].onDisk > diskBuffer[max_].onDisk)
				max_ = x;
		}
		FlushBuffer(max_);
	}
	
	void Solver::FlushBuffers()
	{
		for (int x = 0; x < diskBuffer.size(); x++)
			FlushBuffer(x);
	}

	void Solver::FlushBuffer(uint32_t which)
	{
		data.LoadLargeBufferForWrite(which);

		// flush memory
		for (int x = 0; x < diskBuffer[which].index; x++)
		{
			auto &i = diskBuffer[which].queue[x];
			uint32_t p1Rank = order[i.baseMemOffset+(which<<8)];
			uint32_t p2Rank = i.offset;

			//if (data.SetIf(groups[p1Rank].memoryOffset*r.getMaxP2Rank()+p2Rank, kWin, kDraw))
			if (data.SetIfLargeBuffer(groups[p1Rank].memoryOffset, p2Rank, kWin, kDraw))
			{
				proven++;
				groups[p1Rank].assignedCount++;
				groups[p1Rank].changed = true;
			}
		}
		diskBuffer[which].index = 0;
		
		// read and clear disk
		fseek(diskBuffer[which].f, 0, SEEK_SET);
		for (int y = 0; y < diskBuffer[which].onDisk; y++)
		{
			fread(diskBuffer[which].queue, sizeof(DiskEntry), queueSize, diskBuffer[which].f);
			statesOnDisk -= queueSize;
			for (int x = 0; x < queueSize; x++)
			{
				auto &i = diskBuffer[which].queue[x];
				uint32_t p1Rank = order[i.baseMemOffset+(which<<8)];
				uint32_t p2Rank = i.offset;
				
				//if (data.SetIf(groups[p1Rank].memoryOffset*r.getMaxP2Rank()+p2Rank, kWin, kDraw))
				if (data.SetIfLargeBuffer(groups[p1Rank].memoryOffset, p2Rank, kWin, kDraw))
				{
					proven++;
					groups[p1Rank].assignedCount++;
					groups[p1Rank].changed = true;
				}
			}
		}
		if (diskBuffer[which].onDisk > 20000) // clear file if it was large
		{
			fclose(diskBuffer[which].f);
			diskBuffer[which].f = fopen(GetTempFileName(which), "w+b"); // open and truncate file to keep it small on disk
		}
		else {
			fseek(diskBuffer[which].f, 0, SEEK_SET);
		}
		diskBuffer[which].onDisk = 0;

		//data.FlushLargeBufferWrites(which);

	}


	
#pragma mark - other utilities -
	
	void Solver::PrintStats() const
	{
		uint64_t w = 0, l = 0, d = 0, i = 0;
		for (uint64_t x = 0; x < data.Size(); x++)
		{
			switch (data.Get(x))
			{
				case EM::kWin: w++; break;
				case EM::kLoss: l++; break;
				case EM::kIllegal: i++; break;
				case EM::kDraw: d++; break;
			}
		}
		printf("--Cache Data Summary--\n");
		printf("%llu wins\n%llu losses\n%llu draws\n%llu illegal\n", w, l, d, i);
		std::cout << stat << "\n";
	}
	
	const char *Solver::GetFileName(const char *dataPath)
	{
		static std::string s;
		s = dataPath;
		s += "CC-SOLVE-EM-";
		s += std::to_string(NUM_SPOTS);
		s += "-";
		s += std::to_string(NUM_PIECES);
		s += "-";
		s += r.name();
		return s.c_str();
	}
	
	const char *Solver::GetTempFileName(int which)
	{
		static std::string s;
		s = scratchPath;
		s += "CC-TMP-"+std::to_string(which);
		s += ".dat";
		return s.c_str();
	}

	tResult Solver::Translate(int nextPlayer, tResult res) const
	{
		switch (nextPlayer)
		{
			case 0: return res;
			case 1:
			{
				// 	kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
				tResult inv[4] = {kDraw, kWin, kLoss, kIllegal};
				return inv[res];
			}
		}
		assert(false);
		return kIllegal;
	}
	
	tResult Solver::Translate(const CCState &s, tResult res) const
	{
		switch (s.toMove)
		{
			case 0: return res;
			case 1:
			{
				// 	kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
				tResult inv[4] = {kDraw, kWin, kLoss, kIllegal};
				return inv[res];
			}
		}
		assert(false);
		return kIllegal;
	}
	
	tResult Solver::Lookup(const CCState &s) const
	{
		static CCheckers cc;
		int64_t p1, p2;
		uint64_t v = r.rank(s, p1, p2);
		if (groups[p1].symmetryRedundant)
		{
			CCState tmp = s;
			cc.SymmetryFlipHoriz(tmp);
			v = r.rank(tmp, p1, p2);
			if (groups[p1].symmetryRedundant)
			{
				printf("Flipped state and still symmetry redundant!\n");
				exit(0);
			}
		}
		return Translate(s, (tResult)data.Get(groups[p1].memoryOffset, p2));
//		return Translate(s, (tResult)data.Get(v));
	}
	
	const char *Solver::LookupText(const CCState &s) const
	{
		return resultText[Lookup(s)];
	}

	void Solver::DebugState(CCState &s)
	{
		CCheckers cc;
		s.PrintASCII();
		CCMove *c = cc.getMoves(s);
		for (CCMove *n = c; n; n = n->next)
		{
			cc.ApplyMove(s, n);
			printf("{%d} ", Lookup(s) );
			s.PrintASCII();
			cc.UndoMove(s, n);
		}
	}

	void Solver::DebugState(int64_t r1, int64_t r2)
	{
		CCState s;
		r.unrank(r1, r2, s);
		DebugState(s);
	}

}


