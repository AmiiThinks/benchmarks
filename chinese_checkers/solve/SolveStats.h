//
//  SolveStats.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 4/30/16.
//  Copyright © 2016 NS Software. All rights reserved.
//

#ifndef SolveStats_h
#define SolveStats_h

#include <mutex>
#include <iostream>
#include <string.h>

struct stats
{
	stats() { memset(this, 0, sizeof(stats)-sizeof(std::mutex)); }
	stats &operator+=(const stats &s)
	{
		lock.lock();
		legalStates += s.legalStates;
		winningStates += s.winningStates;
		losingStates += s.losingStates;
		illegalStates += s.illegalStates;
		forwardExpansions += s.forwardExpansions;
		backwardExpansions += s.backwardExpansions;
		forwardChildren += s.forwardChildren;
		backwardChildren += s.backwardChildren;
		apply += s.apply;
		undo += s.undo;
		rank += s.rank;
		unrank += s.unrank;
		changed += s.changed;
		lock.unlock();
		return *this;
	}
	uint64_t legalStates;
	uint64_t winningStates;
	uint64_t losingStates;
	uint64_t illegalStates;
	uint64_t forwardExpansions;
	uint64_t backwardExpansions;
	uint64_t forwardChildren;
	uint64_t backwardChildren;
	uint64_t apply;
	uint64_t undo;
	uint64_t rank;
	uint64_t unrank;
	uint64_t changed;
	std::mutex lock;
};

static std::ostream &operator<<(std::ostream &out, const stats &s)
{
	out << "Changed: " << s.changed << " ";
	out << "Legal: " << s.legalStates << " ";
	out << "Win: " << s.winningStates << " ";
	out << "Loss: " << s.losingStates << " ";
	out << "Illeg: " << s.illegalStates << " ";
	out << "ForExp: " << s.forwardExpansions << " ";
	out << "BckExp: " << s.backwardExpansions << " ";
	out << "ForCld: " << s.forwardChildren << " ";
	out << "BckCld: " << s.backwardChildren << " ";
	out << "AvgForChd: " << ((s.forwardExpansions>0)?((float)s.forwardChildren/s.forwardExpansions):(0)) << " ";
	out << "AvgBackChd: " << ((s.backwardExpansions>0)?((float)s.backwardChildren/s.backwardExpansions):(0)) << " ";
	out << "Apply: " << s.apply << " ";
	out << "Undo: " << s.undo << " ";
	out << "Rank: " << s.rank << " ";
	out << "Unrank: " << s.unrank;
	return out;
}

#endif /* SolveStats_h */
