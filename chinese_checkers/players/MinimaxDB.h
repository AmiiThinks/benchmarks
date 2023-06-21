//
//  MinimaxDB.h
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/17/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef __CC_UCT__MinimaxDB__
#define __CC_UCT__MinimaxDB__

#include <stdio.h>
#include <assert.h>
#include <string>
#include <string.h>
#include "CCheckers.h"
#include "CCUtils.h"
#include "FPUtil.h"
#include "Timer.h"
#include "Player.h"
#include "CCEndgameData.h"

namespace MinimaxDB {
	struct StateEval {
		StateEval(double val = 0) { value = val; m = 0; }
		double value;
		CCMove *m;
	};
	
	enum boundType {
		upperBound,
		lowerBound,
		exactValue
	};
	
	struct TTEntry {
		uint64_t hash;
		double value;
		int depth;
		boundType b;
	};

	const uint32_t TTSize = (1<<24)-1;
	const uint64_t moveHash[2] = {7691059789347120792ull, 6871216590221194721ull};
	const uint64_t rands[2][81] =
	{{4675161237689191092ull, 8004795258322021864ull, 6478254587141557080ull, 5922839388686883993ull, 5060718716316130659ull, 5827768806656723959ull, 4235715689756156486ull, 1549195859410184122ull, 724629090236635265ull, 6615907259034218288ull, 564417634765611607ull, 2112862335165942686ull, 4308642580878263215ull, 738451561808175508ull, 3025486290891311733ull, 4240502770267730464ull, 2416134102176500141ull, 4242989019739141987ull, 8146471891693225747ull, 2003771968734191074ull, 2253458390144465111ull, 660245571370364273ull, 5631218200250778438ull, 8193337635489743215ull, 2027072587594793413ull, 5899206021963240096ull, 9128976886711489976ull, 1645254491491795893ull, 5805027117214507015ull, 3162834577623107880ull, 1907461893467795479ull, 7307788895536578470ull, 1201891296328588113ull, 3994612648107577034ull, 3269862249992194823ull, 4108809722321618651ull, 4606553353230423686ull, 2789122676740054780ull, 6978496724472531320ull, 7370246804966356372ull, 7344961137339175179ull, 2373687499720437593ull, 2834777110349227635ull, 2241441376781123627ull, 6402648048807172184ull, 8246945596667293942ull, 3584241258997843100ull, 3410333901064487476ull, 8658768338065008551ull, 4911123708640952913ull, 4179959129159056811ull, 8228579191339087500ull, 4565616692094745977ull, 577539962161131005ull, 3553418108746872308ull, 2102938657072534062ull, 2770495520347532230ull, 8925573192132193731ull, 8495568152300172438ull, 5021704823553202355ull, 7347918484315289410ull, 4761582065377793374ull, 1153268806155008582ull, 6494920266079755211ull, 6353490476612085904ull, 3293923652125217947ull, 1979233334121211721ull, 873218014451071596ull, 4093460637833896113ull, 8865591410223426484ull, 175172702589946875ull, 98513918957264293ull, 6221784189381640042ull, 8550940860076689914ull, 1744357606293223766ull, 1945452430077346031ull, 8394552030140836951ull, 1577691764507061366ull, 5655578427190128095ull, 913113900363860107ull, 1031911931597973086ull },
		{4095774435863805099ull, 6506168361246091342ull, 332623257227850346ull, 6419750424779595649ull, 9023135710953463717ull, 1720000663185169410ull, 317219319894696766ull, 8922364701997926578ull, 4212393118391586669ull, 3994061785596760038ull, 1312254329485079201ull, 3641508007069296297ull, 3489690576176407259ull, 1739186087724181337ull, 2859411268647054071ull, 8609842037619280012ull, 5782243287188822878ull, 2700806120118443706ull, 7268895326383290842ull, 6089709177160657896ull, 6492107635412806072ull, 6966936589242963657ull, 6528122653910377302ull, 6454115091981662768ull, 8109279523675005316ull, 7674989279621153567ull, 4390046623901435476ull, 3088107910297449276ull, 2166839653682773584ull, 3059169644058301589ull, 7598392017343785036ull, 2758064587838427370ull, 8338278006833401340ull, 1048757777387213172ull, 5356055216000134624ull, 4229088253546668374ull, 5724804755891438918ull, 210745730863408271ull, 8638582056026003355ull, 4546699106246744928ull, 7889665665716201686ull, 7722651361299025220ull, 5091441605291247688ull, 1712705420280895577ull, 6414681232763329302ull, 3061160142301589472ull, 7217514709162002959ull, 8946964483077654286ull, 2054723401242112111ull, 3492849934352801763ull, 7910817178157962718ull, 812197390650668520ull, 471218952684583257ull, 7441805272289149888ull, 7687803946506631479ull, 7410918516849527316ull, 5763448694989208992ull, 187065793423626551ull, 7497585080330442714ull, 8889799520252218496ull, 6671646273748751998ull, 255223454784042579ull, 8546812353691565356ull, 1774837584046864537ull, 751166120322045411ull, 8788942333173664747ull, 3766253665149393334ull, 3850861163114825445ull, 1400710808483634250ull, 4123901593984754692ull, 4834426738082834886ull, 6219240295651030926ull, 8221645915273282930ull, 3006021657495465294ull, 8306077300631695957ull, 1961381200280653884ull, 5400036762296972848ull, 1196768448874359137ull, 2195420551776650225ull, 6101640797212664055ull, 8239293553468373623ull }};
	
	
	template <typename eval>
	class MinimaxDB : public Player {
	public:
		MinimaxDB(const char *prefix, int firstEntry, int firstPiece,
				  eval *e, bool verbose = true)
		:e(e), db(prefix, firstEntry)
		{
			numMoves = totalDepth = 0;
			historyHeuristic = true;
			transpositionTable = true;
			this->verbose = verbose;
			validateTranspositionTable = false;
			TT = new TTEntry[TTSize];
			memset(TT, 0, sizeof(TTEntry)*TTSize);
			perfectFromRoot = false;
			this->firstPiece = firstPiece;
		}
		~MinimaxDB()
		{ delete TT; }
		MinimaxDB(MinimaxDB const&) = delete;
		MinimaxDB& operator=(MinimaxDB const&) = delete;
		void Reset() { 	rootDepth0 = rootDepth1 = 27; }
		CCMove *GetNextAction(CCheckers *cc, const CCState &s, double &value, double timeLimit, int depthLimit);
		virtual const char *GetName()
		{
			name = "minimaxdb-";
			if (historyHeuristic)
				name += "hh-";
			if (transpositionTable)
				name += "tt-";
			name += e->GetName();
			return name.c_str();
		}
		uint64_t GetNodesExpanded() { return expanded; }
		bool historyHeuristic;
		bool transpositionTable;
		bool validateTranspositionTable;
	private:
		bool SearchDone(CCheckers *cc, CCState &s, StateEval &result, int depth, int p0Depth, int p1Depth);
		void ApplyMove(CCheckers *cc, CCState &s, CCMove *m);
		void UndoMove(CCheckers *cc, CCState &s, CCMove *m);
		StateEval MaxPlayer(CCheckers *cc, CCState &s, int depthRemaining,
							int p0Depth, int p1Depth,
							double alpha, double beta, bool debug);
		StateEval MinPlayer(CCheckers *cc, CCState &s, int depthRemaining,
							int p0Depth, int p1Depth,
							double alpha, double beta, bool debug);
		StateEval Dispatch(CCheckers *cc, CCState &s, int depthRemaining,
						   int p0Depth, int p1Depth,
						   double alpha, double beta, bool debug);
		void RenewDepthEstimates(CCState &currState);
		CCMove *ReorderMoves(CCMove *moves);
		
		Timer t;
		double timeLimit;
		eval *e;
		int rootPlayer;
		uint64_t expanded;
		uint64_t hh[81*81];
		int numMoves;
		int totalDepth;
		std::string name;
		bool verbose;
		uint64_t hash;
		uint64_t ttCollisions;
		uint64_t ttAdds;
		uint64_t ttHits;
		TTEntry *TT; // approximately 10 million entries
		int rootDepth0, rootDepth1;
		//static const uint64_t rands[2][81];
		StateEval tmp;
		bool perfectFromRoot;
		CCEndGameData db;
		int firstPiece;
	};
	
	
	template <typename eval>
	void MinimaxDB<eval>::RenewDepthEstimates(CCState &currState)
	{
		int curr0 = GetDepth(fullDataPrefix, currState, 0);
		int curr1 = GetDepth(fullDataPrefix, currState, 1);
		if (curr0 != -1)
		{
			while (curr0 < rootDepth0-10)
				curr0+=15;
			printf("P0: Was at depth %d, now predicting %d\n", rootDepth0, curr0);
			rootDepth0 = curr0;
		}
		if (curr1 != -1)
		{
			while (curr1 < rootDepth1-10)
				curr1+=15;
			printf("P1: Was at depth %d, now predicting %d\n", rootDepth1, curr1);
			rootDepth1 = curr1;
		}
	}
	
	
	template <typename eval>
	CCMove *MinimaxDB<eval>::GetNextAction(CCheckers *cc, const CCState &s, double &value, double timeLimit, int depthLimit)
	{
		std::cerr << GetName() << " average so far: " << double(totalDepth)/double(numMoves) << " ply per search.\n";
		t.StartTimer();
		//memset(TT, 0, sizeof(TTEntry)*TTSize);
		numMoves++;
		CCState currState = s;
		StateEval curr, best;
		ttCollisions = 0;
		ttAdds = 0;
		ttHits = 0;
		hash = moveHash[s.toMove];
		for (int x = 0; x < NUM_PIECES; x++)
		{
			hash ^= rands[0][s.pieces[0][x]];
			hash ^= rands[1][s.pieces[1][x]];
		}
		expanded = 0;
		rootPlayer = s.toMove;
		e->RootState(s);
		this->timeLimit = timeLimit;
		for (int x = 0; x < 81*81; x++)
			hh[x] = 0;
		if (GetBackPieceAdvancement(s, 0) >= firstPiece &&
			GetBackPieceAdvancement(s, 1) >= firstPiece)
		{
			perfectFromRoot = true;
		}
		else {
			perfectFromRoot = false;
		}
		
		// This code estimates the true depth of the current state
		RenewDepthEstimates(currState);
		
		for (int depth = 1; depth <= depthLimit; depth++)
		{
			std::cerr << "Starting search depth " << depth << std::endl;
			curr = MaxPlayer(cc, currState, depth,
							 rootDepth0, rootDepth1,
							 -1000, 1000, verbose);
			if (t.GetElapsedTime() >= timeLimit)
			{
				totalDepth += depth-1;
				cc->freeMove(curr.m);
				cc->freeMove(best.m->next);
				best.m->next = 0;
				value = best.value;
				if (transpositionTable)
					std::cerr << ttHits << " tt hits; " << ttCollisions << " collisions; " << ttAdds << " adds.\n";
				return best.m;
			}
			std::cerr << "Expanded " << expanded << " states" << std::endl;
			cc->freeMove(best.m);
			best = curr;
			std::cerr << "Best move value " << best.value << " ";
			best.m->Print(1);
			std::cerr << "\n";
			if (curr.value == 1000)
			{
				totalDepth += depth;
				std::cerr << "Got a win!\n";
				cc->freeMove(best.m->next);
				best.m->next = 0;
				value = best.value;
				return best.m;
			}
			if (curr.value == -1000)
			{
				totalDepth += depth;
				std::cerr << "Got a loss!\n";
				cc->freeMove(best.m->next);
				best.m->next = 0;
				value = best.value;
				return best.m;
			}
		}
		value = best.value;
		if (transpositionTable)
			std::cerr << ttHits << " tt hits; " << ttCollisions << " collisions; " << ttAdds << " adds.\n";
		return best.m;
	}
	
	template <typename eval>
	bool MinimaxDB<eval>::SearchDone(CCheckers *cc, CCState &s, StateEval &result,
									 int depth, int p0Depth, int p1Depth)
	{
		if (t.GetElapsedTime() > timeLimit)
		{
			result = StateEval();
			return true;
		}
		if (cc->Done(s))
		{
			result = StateEval((cc->Winner(s) == rootPlayer)?1000:-1000);
			return true;
		}
		if (depth == 0)
		{
			if (perfectFromRoot)
			{
				if (rootPlayer == 0)
				{
					result.value = p1Depth-p0Depth;
					result.m = 0;
				}
				else {
					result.value = p0Depth-p1Depth;
					result.m = 0;
				}
				return true;
			}
			result = StateEval(e->eval(cc, s, rootPlayer));
			return true;
		}
		return false;
	}

	
	template <typename eval>
	StateEval MinimaxDB<eval>::MaxPlayer(CCheckers *cc, CCState &s, int depthRemaining,
										 int p0Depth, int p1Depth,
										 double alpha, double beta, bool debug)
	{
		expanded++;
		
		if (SearchDone(cc, s, tmp, depthRemaining, p0Depth, p1Depth))
			return tmp;
		
		CCMove *m = cc->getMovesForward(s);
		m = ReorderMoves(m);
		StateEval best, curr;
		best.value = alpha;
		for (CCMove *t = m; t; t = t->next)
		{
			ApplyMove(cc, s, t);
			curr = Dispatch(cc, s, depthRemaining-1, p0Depth, p1Depth, best.value, beta, false);
			UndoMove(cc, s, t);
			
			if (debug)
			{
				std::cerr << "   * ";
				t->Print(0);
				std::cerr << " value " << curr.value << "\n";
			}
			if ((curr.value > best.value) || (curr.value == best.value && best.m == 0))
			{
				cc->freeMove(best.m);
				best = curr;
				best.m = t->clone(*cc);
				best.m->next = curr.m;
			}
			else {
				cc->freeMove(curr.m);
			}
			if (best.value >= beta)
				break;
		}
		cc->freeMove(m);
		
		// History Heuristic
		if (best.m)
			hh[best.m->from*81+best.m->to] += 1<<depthRemaining;
		
		return best;
	}
	
	template <typename eval>
	StateEval MinimaxDB<eval>::MinPlayer(CCheckers *cc, CCState &s, int depthRemaining,
										 int p0Depth, int p1Depth,
										 double alpha, double beta, bool debug)
	{
		expanded++;

		if (SearchDone(cc, s, tmp, depthRemaining, p0Depth, p1Depth))
			return tmp;
		
		CCMove *m = cc->getMovesForward(s);
		m = ReorderMoves(m);
		StateEval best, curr;
		best.value = beta;
		for (CCMove *t = m; t; t = t->next)
		{
			ApplyMove(cc, s, t);
			curr = Dispatch(cc, s, depthRemaining-1, p0Depth, p1Depth, alpha, best.value, false);
			UndoMove(cc, s, t);
			
			if (curr.value < best.value)
			{
				cc->freeMove(best.m);
				best = curr;
				best.m = t->clone(*cc);
				best.m->next = curr.m;
			}
			else {
				cc->freeMove(curr.m);
			}
			if (alpha >= best.value)
				break;
		}
		cc->freeMove(m);
		
		if (best.m)
			hh[best.m->from*81+best.m->to] += 1<<depthRemaining;
		
		return best;
	}
	
	template <typename eval>
	StateEval MinimaxDB<eval>::Dispatch(CCheckers *cc, CCState &s, int depthRemaining,
										int p0Depth, int p1Depth,
										double alpha, double beta, bool debug)
	{
		if (depthRemaining >= 2 && transpositionTable)
		{
			TTEntry &entry = TT[hash%TTSize];
			if (entry.depth == depthRemaining && entry.hash == hash)// && s.toMove == entry.toMove)
			{
				double returnValue = entry.value;
				bool usable = false;
				if (entry.b == exactValue)
				{
					if (returnValue < alpha)
						returnValue = alpha;
					if (returnValue > beta)
						returnValue = beta;
					usable = true;
				}
				else if (entry.b == upperBound && alpha >= entry.value)
				{
					returnValue = alpha;
					usable = true;
				}
				else if (entry.b == lowerBound && beta <= entry.value)
				{
					returnValue = beta;
					usable = true;
				}
				
				if (usable && validateTranspositionTable) // validate TT
				{
					StateEval ev;
					if (s.toMove == rootPlayer)
						ev = MaxPlayer(cc, s, depthRemaining,
									   p0Depth, p1Depth,
									   alpha, beta, debug);
					else
						ev = MinPlayer(cc, s, depthRemaining,
									   p0Depth, p1Depth,
									   alpha, beta, debug);
					if (ev.value != returnValue)
					{
						s.Print();
						printf("Alpha: %1.2f, Beta: %1.2f\n", alpha, beta);
						printf("TT: %1.2f, Search: %1.2f\n", returnValue, ev.value);
						exit(0);
					}
					cc->freeMove(ev.m);
				}
				
				if (usable)
				{
					ttHits++;
					return StateEval(returnValue);
				}
			}
		}
		

		StateEval ev;
		if (perfectFromRoot)
		{
			if (s.toMove == 0) // p1 distance just changed
			{
				int d1;
				bool valid = db.GetRawDepth(s, 1, d1);
				assert(valid);
				while (d1 < p1Depth-10)
					d1+=15;
				p1Depth = d1;
			}
			else { // p0 distance just changed
				int d0;
				bool valid = db.GetRawDepth(s, 0, d0);
				assert(valid);
				while (d0 < p1Depth-10)
					d0+=15;
				p0Depth = d0;
			}
		}
		
		if (s.toMove == rootPlayer)
			ev = MaxPlayer(cc, s, depthRemaining, p0Depth, p1Depth, alpha, beta, debug);
		else
			ev = MinPlayer(cc, s, depthRemaining, p0Depth, p1Depth, alpha, beta, debug);

		
		// Transposition table
		if (depthRemaining >= 2 && transpositionTable)
		{
			TTEntry &entry = TT[hash%TTSize];
			if (entry.depth != 0)
				ttCollisions++;
			entry.value = ev.value;
			if (fgreater(ev.value, alpha) && fless(ev.value, beta))
				entry.b = exactValue;
			else if (!fgreater(ev.value, alpha)) // leq
				entry.b = upperBound;
			else if (!fless(ev.value, beta))
				entry.b = lowerBound;
			else { assert(!"Failed alpha-beta bounds cases"); }
			
			entry.hash = hash;
			entry.depth = depthRemaining;
			ttAdds++;
		}
		
		return ev;
	}
	
	template <typename eval>
	CCMove *MinimaxDB<eval>::ReorderMoves(CCMove *moves)
	{
		if (!historyHeuristic)
			return moves;
		CCMove *orderedList = 0;
		
		while (moves != 0)
		{
			CCMove *tmp = moves;
			moves = moves->next;
			tmp->next = 0;
			
			if (orderedList == 0)
			{
				orderedList = tmp;
				continue;
			}
			else if (hh[tmp->from*81+tmp->to] > hh[orderedList->from*81+orderedList->to])
			{
				tmp->next = orderedList;
				orderedList = tmp;
			}
			else {
				CCMove *ol = orderedList;
				while (true)
				{
					if (ol->next == 0)
					{
						ol->next = tmp;
						break;
					}
					else if (hh[tmp->from*81+tmp->to] > hh[ol->next->from*81+ol->next->to])
					{
						tmp->next = ol->next;
						ol->next = tmp;
						break;
					}
					else if (hh[tmp->from*81+tmp->to] == hh[ol->next->from*81+ol->next->to] && 0 == random()%2)
					{
						tmp->next = ol->next;
						ol->next = tmp;
						break;
					}
					ol = ol->next;
				}
			}
		}
		return orderedList;
	}
	
	template <typename eval>
	void MinimaxDB<eval>::ApplyMove(CCheckers *cc, CCState &s, CCMove *t)
	{
		hash ^= rands[s.toMove][t->from];
		hash ^= rands[s.toMove][t->to];
		hash ^= moveHash[s.toMove];
		hash ^= moveHash[1-s.toMove];
		cc->ApplyMove(s, t);

	}
	
	template <typename eval>
	void MinimaxDB<eval>::UndoMove(CCheckers *cc, CCState &s, CCMove *t)
	{
		cc->UndoMove(s, t);
		hash ^= rands[s.toMove][t->from];
		hash ^= rands[s.toMove][t->to];
		hash ^= moveHash[s.toMove];
		hash ^= moveHash[1-s.toMove];
	}

}

#endif /* defined(__CC_UCT__MinimaxDB__) */
