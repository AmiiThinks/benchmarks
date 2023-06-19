/* 
 *
 * ccwrapper.cpp
 *
 * C++ Wrapper for functions necessary to use CC simulator in Python
 * Author: Zaheen Ahmad
 *
*/

#include </Users/bigyankarki/opt/anaconda3/envs/cc2/include/pybind11/pybind11.h>
#include <iostream>
#include "CCheckers.h"
#include "DistEval.h"
#include "UCT.h"
#include "FasterSymmetrySolver.h"
#include "CCRankings.h"

using namespace std;
//using namespace FasterSymmetry;
namespace py = pybind11;

//
// Creating Python binding for necessary classes/methods
//

PYBIND11_MODULE(ccwrapper, m)
{
    // Wrapper for CCMove class
    // Added getter/setter to access attributes
    m.attr("BOARD_SIZE") = DIAGONAL_LEN;
    m.attr("NUM_PIECES") = NUM_PIECES;

    py::class_<CCMove>(m, "CCMove")
        .def(py::init<CCMove*>())
        .def(py::init<int, int, CCMove*>())
        .def("length", &CCMove::length)
        .def_readwrite("from_", &CCMove::from)
        .def_readwrite("to_", &CCMove::to)
        .def("getFrom", &CCMove::getFrom)
        .def("getTo", &CCMove::getTo)
        .def("getWhich", &CCMove::getWhich)
        .def("getNextMove", &CCMove::getNextMove,
                py::return_value_policy::reference)
        .def("setNextMove", &CCMove::setNextMove)
        .def("clone", &CCMove::clone,
             py::return_value_policy::reference)
        .def("Print", &CCMove::Print);

    // Wrapper for CCState class
 
    py::class_<CCState>(m, "CCState")
        .def(py::init<>())
        .def("Print", &CCState::Print)
        .def("PrintASCII", &CCState::PrintASCII) // first: function name in python, second is function name in the class
        // .def("convertToCCState", &CCState::convertToCCState)
        .def_readonly("board", &CCState::board)
        .def("getBoard", &CCState::getBoard,
                py::return_value_policy::reference)
        .def("getPieces", &CCState::getPieces,
                 py::return_value_policy::reference)
        .def("getToMove", &CCState::getToMove);

        m.def("list_to_ccstate", &listToCCState, "Convert a Python list to a CCState object", py::arg("list"), py::arg("toMove"));

    // Wrapper for CCheckers class

    py::class_<CCheckers>(m, "CCheckers")
        .def(py::init<>())
        .def("Reset", &CCheckers::Reset)
        .def("Done", &CCheckers::Done)
        .def("Winner", &CCheckers::Winner)
        .def("getNewMove", &CCheckers::getNewMove)
        .def("freeMove", &CCheckers::freeMove)
        .def("getMoves", &CCheckers::getMoves,
                py::return_value_policy::reference)
        .def("getMovesForward", &CCheckers::getMovesForward,
                py::return_value_policy::reference)
        .def("getReverseMoves", &CCheckers::getReverseMoves,
                py::return_value_policy::reference)
        .def("ApplyMove", &CCheckers::ApplyMove)
        .def("applyState", &CCheckers::applyState)
        .def("UndoMove", &CCheckers::UndoMove)
        .def("delMove", &CCheckers::delMove);


    py::class_<DistEval>(m, "DistEval")
            .def(py::init<>());


    py::class_<BestPlayout>(m, "BestPlayout")
            .def(py::init<>());
            

    py::class_<Node>(m, "Node")
            .def(py::init<>())
            .def_readonly("numSamples", &Node::numSamples)
            .def_readonly("fromCell", &Node::from)
            .def_readonly("toCell", &Node::to);


    py::class_<UCT<BestPlayout, DistEval>>(m, "UCT")
            .def(py::init<double, BestPlayout*, DistEval*>())
            .def("get_next_action", &UCT<BestPlayout, DistEval>::GetNextAction,
                    py::return_value_policy::reference)
            .def_readonly("stats", &UCT<BestPlayout, DistEval>::stats);

    py::enum_<FasterSymmetry::tResult>(m, "tResult")
            .value("kWin", FasterSymmetry::tResult::kWin)
            .value("kLoss", FasterSymmetry::tResult::kLoss)
            .value("kDraw", FasterSymmetry::tResult::kDraw)
            .value("kIllegal", FasterSymmetry::tResult::kIllegal)
            .export_values();


    py::class_<FasterSymmetry::Solver>(m, "Solver")
            .def(py::init<const char*, bool, bool>())
            .def("lookup", &FasterSymmetry::Solver::Lookup);

    py::class_<CCLocalRank12>(m, "CCLocalRank12")
            .def(py::init<>())
            .def("getMaxRank", &CCLocalRank12::getMaxRank)
            .def("unrank", &CCLocalRank12::unrank);

}
